from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import torch.nn.functional as F
import comfy.model_management as mm
import os
import gc
import weakref
import math

torch.set_float32_matmul_precision(["high", "highest"][0])

transform_image = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

current_path  = os.getcwd()

## ComfyUI portable standalone build for Windows 
model_path = os.path.join(current_path, "ComfyUI"+os.sep+"models"+os.sep+"BiRefNet")

# 全局模型缓存字典
_model_cache = {}
_device_cache = {}

def get_available_memory(device):
    """获取可用显存"""
    if device.startswith('cuda'):
        try:
            current_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            available_memory = total_memory - current_memory
            return available_memory
        except:
            return 4.0  # 默认4GB
    return 16.0  # CPU或其他设备默认大内存

def calculate_optimal_resolution(image_size, device, target_memory_gb=2.0):
    """根据可用显存计算最优分辨率"""
    w, h = image_size
    available_memory = get_available_memory(device)
    
    # 如果可用显存充足，使用较高分辨率
    if available_memory > 6.0:
        max_res = 1024
    elif available_memory > 4.0:
        max_res = 768
    elif available_memory > 2.0:
        max_res = 512
    else:
        max_res = 384  # 极端情况下的最小分辨率
    
    # 确保不超过原图尺寸太多
    original_max = max(w, h)
    if original_max < max_res:
        max_res = min(max_res, original_max + 128)  # 适当放大但不过度
    
    return max_res

def create_dynamic_transform(target_size):
    """创建动态的图像变换"""
    return transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

def get_cached_model(model_name, local_model_path, load_local_model, device):
    """获取缓存的模型，避免重复加载"""
    cache_key = f"{model_name}_{local_model_path}_{load_local_model}"
    
    # 检查是否有缓存的模型
    if cache_key in _model_cache:
        model = _model_cache[cache_key]
        # 检查设备是否匹配
        if cache_key in _device_cache and _device_cache[cache_key] == device:
            return model
        else:
            # 设备不匹配，移动模型到新设备
            model = model.to(device)
            _device_cache[cache_key] = device
            return model
    
    # 加载新模型前，检查显存并清理
    if device.startswith('cuda'):
        available_mem = get_available_memory(device)
        print(f"\033[93m可用显存: {available_mem:.1f}GB\033[0m")
        
        if available_mem < 3.0:  # 如果可用显存不足3GB
            torch.cuda.empty_cache()
            gc.collect()
            print("\033[93m显存不足，已执行清理\033[0m")
    
    # 加载新模型
    print(f"\033[93m正在加载BiRefNet模型到设备: {device}\033[0m")
    
    if load_local_model:
        model = AutoModelForImageSegmentation.from_pretrained(
            local_model_path, trust_remote_code=True
        )
    else:
        model = AutoModelForImageSegmentation.from_pretrained(
            model_name, trust_remote_code=True
        )
    
    # 使用ComfyUI的显存管理
    model = model.to(device)
    
    # 缓存模型
    _model_cache[cache_key] = model
    _device_cache[cache_key] = device
    
    return model

def clear_model_cache():
    """清理模型缓存"""
    global _model_cache, _device_cache
    for model in _model_cache.values():
        del model
    _model_cache.clear()
    _device_cache.clear()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def resize_image_smart(image, max_size=1024, device="cuda"):
    """超级智能resize，根据显存动态调整"""
    image = image.convert('RGB')
    w, h = image.size
    
    # 根据显存情况动态调整最大分辨率
    optimal_max_size = calculate_optimal_resolution((w, h), device)
    final_max_size = min(max_size, optimal_max_size)
    
    print(f"\033[96m原始分辨率: {w}x{h}, 目标最大分辨率: {final_max_size}\033[0m")
    
    # 如果图片已经很小，不需要resize
    if max(w, h) <= final_max_size:
        # 保持原始尺寸，但确保是32的倍数（对模型更友好）
        w = ((w + 31) // 32) * 32
        h = ((h + 31) // 32) * 32
        model_input_size = (w, h)
    else:
        # 等比缩放到final_max_size
        if w > h:
            new_w = final_max_size
            new_h = int(h * final_max_size / w)
        else:
            new_h = final_max_size
            new_w = int(w * final_max_size / h)
        
        # 确保尺寸是32的倍数
        new_w = ((new_w + 31) // 32) * 32
        new_h = ((new_h + 31) // 32) * 32
        model_input_size = (new_w, new_h)
    
    image = image.resize(model_input_size, Image.BILINEAR)
    print(f"\033[96m调整后分辨率: {model_input_size[0]}x{model_input_size[1]}\033[0m")
    return image

def safe_model_inference(model, input_tensor, device):
    """安全的模型推理，包含多级降级策略"""
    try:
        # 第一次尝试：正常推理
        with torch.no_grad():
            result = model(input_tensor)[-1].sigmoid()
            return result.cpu()
    except torch.cuda.OutOfMemoryError as e:
        print(f"\033[91m显存不足，尝试清理后重试...\033[0m")
        
        # 清理显存
        del input_tensor
        torch.cuda.empty_cache()
        gc.collect()
        
        # 重新创建输入张量，但使用更小的分辨率
        raise e  # 重新抛出异常，让上层处理

colors = ["transparency", "green", "white", "red", "yellow", "blue", "black", "pink", "purple", "brown", "violet", "wheat", "whitesmoke", "yellowgreen", "turquoise", "tomato", "thistle", "teal", "tan", "steelblue", "springgreen", "snow", "slategrey", "slateblue", "skyblue", "orange"]

def get_device_by_name(device):
    """
    "device": (["auto", "cuda", "cpu", "mps", "xpu", "meta"],{"default": "auto"}), 
    """
    if device == 'auto':
        try:
            device = "cpu"
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            elif torch.xpu.is_available():
                device = "xpu"
        except:
                raise AttributeError("What's your device(到底用什么设备跑的)？")
    print("\033[93mUse Device(使用设备):", device, "\033[0m")
    return device


class BiRefNet_Hugo:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        
        return {
            "required": {
                "image": ("IMAGE",),
                "model": (["ZhengPeng7/BiRefNet", "ZhengPeng7/BiRefNet_HR", "ZhengPeng7/BiRefNet-portrait"],{"default": "ZhengPeng7/BiRefNet"}),
                "load_local_model": ("BOOLEAN", {"default": False}),
                "background_color_name": (colors,{"default": "transparency"}), 
                "device": (["auto", "cuda", "cpu", "mps", "xpu", "meta"],{"default": "auto"}),
                "max_resolution": ("INT", {"default": 768, "min": 384, "max": 1024, "step": 64}),
                "enable_memory_efficient": ("BOOLEAN", {"default": True}),
                "auto_resolution": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "local_model_path": ("STRING", {"default":model_path}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "background_remove"
    CATEGORY = "🔥BiRefNet"
  
    def background_remove(self, 
                          image, 
                          model,
                          load_local_model,
                          device, 
                          background_color_name,
                          max_resolution=768,
                          enable_memory_efficient=True,
                          auto_resolution=True,
                          *args, **kwargs
                          ):
        processed_images = []
        processed_masks = []
       
        device = get_device_by_name(device)
        
        # 显示当前显存状态
        if device.startswith('cuda'):
            available_mem = get_available_memory(device)
            print(f"\033[96m当前可用显存: {available_mem:.1f}GB\033[0m")
        
        try:
            # 获取缓存的模型
            local_model_path = kwargs.get("local_model_path", model_path)
            birefnet = get_cached_model(model, local_model_path, load_local_model, device)
            
            # 强制内存效率模式进行单张处理
            for idx, img_tensor in enumerate(image):
                print(f"\033[96m正在处理第 {idx+1}/{len(image)} 张图片\033[0m")
                
                retry_count = 0
                max_retries = 3
                current_max_res = max_resolution
                
                while retry_count < max_retries:
                    try:
                        orig_image = tensor2pil(img_tensor)
                        w, h = orig_image.size
                        
                        # 智能resize - 根据显存自动调整分辨率
                        if auto_resolution:
                            resized_image = resize_image_smart(orig_image, current_max_res, device)
                        else:
                            # 手动resize
                            if max(w, h) > current_max_res:
                                if w > h:
                                    new_w = current_max_res
                                    new_h = int(h * current_max_res / w)
                                else:
                                    new_h = current_max_res
                                    new_w = int(w * current_max_res / h)
                                # 确保是32的倍数
                                new_w = ((new_w + 31) // 32) * 32
                                new_h = ((new_h + 31) // 32) * 32
                                resized_image = orig_image.resize((new_w, new_h), Image.BILINEAR)
                            else:
                                resized_image = orig_image
                        
                        # 创建动态transform
                        dynamic_transform = create_dynamic_transform(resized_image.size)
                        im_tensor = dynamic_transform(resized_image).unsqueeze(0)
                        im_tensor = im_tensor.to(device)
                        
                        # 推理前检查显存
                        if device.startswith('cuda'):
                            torch.cuda.empty_cache()
                        
                        # 安全推理
                        print(f"\033[96m开始推理，分辨率: {resized_image.size}\033[0m")
                        with torch.no_grad():
                            result = birefnet(im_tensor)[-1].sigmoid()
                            
                            # 立即移到CPU释放GPU显存
                            result = result.cpu()
                            
                            # 清理中间张量
                            del im_tensor
                            if device.startswith('cuda'):
                                torch.cuda.empty_cache()
                        
                        print(f"\033[92m推理成功!\033[0m")
                        
                        # 后处理
                        result = torch.squeeze(F.interpolate(result, size=(h, w), mode='bilinear', align_corners=False))
                        ma = torch.max(result)
                        mi = torch.min(result)
                        result = (result - mi) / (ma - mi + 1e-8)
                        
                        im_array = (result * 255).data.numpy().astype(np.uint8)
                        pil_im = Image.fromarray(np.squeeze(im_array))
                        
                        # 生成最终图像
                        if background_color_name == 'transparency':
                            color = (0, 0, 0, 0)
                            mode = "RGBA"
                        else:
                            color = background_color_name
                            mode = "RGB"
                        
                        new_im = Image.new(mode, pil_im.size, color)
                        new_im.paste(orig_image, mask=pil_im)
                        
                        new_im_tensor = pil2tensor(new_im)
                        pil_im_tensor = pil2tensor(pil_im)
                        
                        processed_images.append(new_im_tensor)
                        processed_masks.append(pil_im_tensor)
                        
                        # 清理中间变量
                        del result, im_array, pil_im, new_im
                        
                        # 成功处理，跳出重试循环
                        break
                        
                    except torch.cuda.OutOfMemoryError as e:
                        retry_count += 1
                        print(f"\033[91m显存不足 (尝试 {retry_count}/{max_retries})，降低分辨率重试...\033[0m")
                        
                        # 清理显存
                        if 'im_tensor' in locals():
                            del im_tensor
                        if 'result' in locals():
                            del result
                        torch.cuda.empty_cache()
                        gc.collect()
                        
                        # 降低分辨率
                        current_max_res = max(384, current_max_res - 128)
                        print(f"\033[93m降低分辨率到: {current_max_res}\033[0m")
                        
                        if retry_count >= max_retries:
                            print(f"\033[91m重试次数已达上限，跳过此图片\033[0m")
                            # 创建一个空白的结果
                            orig_image = tensor2pil(img_tensor)
                            empty_mask = Image.new('L', orig_image.size, 0)
                            processed_images.append(pil2tensor(orig_image))
                            processed_masks.append(pil2tensor(empty_mask))
                            break
                    
                    except Exception as e:
                        print(f"\033[91m处理图片时出错: {str(e)}\033[0m")
                        # 在出错时也要清理显存
                        if device.startswith('cuda'):
                            torch.cuda.empty_cache()
                        raise e
                
                # 每张图片处理后清理
                if device.startswith('cuda'):
                    torch.cuda.empty_cache()
                    gc.collect()

            new_ims = torch.cat(processed_images, dim=0)
            new_masks = torch.cat(processed_masks, dim=0)

            print(f"\033[92m所有图片处理完成!\033[0m")
            return new_ims, new_masks
            
        except Exception as e:
            print(f"\033[91mBiRefNet处理失败: {str(e)}\033[0m")
            # 发生错误时清理显存
            if device.startswith('cuda'):
                torch.cuda.empty_cache()
            gc.collect()
            raise e


# 添加清理函数节点
class BiRefNet_ClearCache:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trigger": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("success",)
    FUNCTION = "clear_cache"
    CATEGORY = "🔥BiRefNet"
    
    def clear_cache(self, trigger):
        """手动清理BiRefNet模型缓存"""
        if trigger:
            clear_model_cache()
            print("\033[92mBiRefNet模型缓存已清理\033[0m")
        return (True,)


NODE_CLASS_MAPPINGS = {
    "BiRefNet_Hugo": BiRefNet_Hugo,
    "BiRefNet_ClearCache": BiRefNet_ClearCache
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "BiRefNet_Hugo": "🔥BiRefNet",
    "BiRefNet_ClearCache": "🔥BiRefNet清理缓存"
}
