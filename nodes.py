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
    
    # 加载新模型
    print(f"\033[93m正在加载BiRefNet模型到设备: {device}\033[0m")
    
    # 在加载新模型前，确保有足够显存
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
        gc.collect()
    
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

def resize_image(image, max_size=1024):
    """智能resize，避免过大的张量"""
    image = image.convert('RGB')
    w, h = image.size
    
    # 如果图片已经很小，不需要resize到1024
    if max(w, h) <= max_size:
        # 保持原始尺寸，但确保是偶数
        w = w if w % 2 == 0 else w + 1
        h = h if h % 2 == 0 else h + 1
        model_input_size = (w, h)
    else:
        # 等比缩放到max_size
        if w > h:
            new_w = max_size
            new_h = int(h * max_size / w)
        else:
            new_h = max_size
            new_w = int(w * max_size / h)
        
        # 确保尺寸是偶数
        new_w = new_w if new_w % 2 == 0 else new_w + 1
        new_h = new_h if new_h % 2 == 0 else new_h + 1
        model_input_size = (new_w, new_h)
    
    image = image.resize(model_input_size, Image.BILINEAR)
    return image

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
                "max_resolution": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 64}),
                "enable_memory_efficient": ("BOOLEAN", {"default": True}),
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
                          max_resolution=1024,
                          enable_memory_efficient=True,
                          *args, **kwargs
                          ):
        processed_images = []
        processed_masks = []
       
        device = get_device_by_name(device)
        
        try:
            # 获取缓存的模型
            local_model_path = kwargs.get("local_model_path", model_path)
            birefnet = get_cached_model(model, local_model_path, load_local_model, device)
            
            # 如果启用内存效率模式，分批处理
            batch_size = 1 if enable_memory_efficient else len(image)
            
            for i in range(0, len(image), batch_size):
                batch_images = image[i:i+batch_size]
                
                for img_tensor in batch_images:
                    try:
                        orig_image = tensor2pil(img_tensor)
                        w, h = orig_image.size
                        
                        # 智能resize
                        resized_image = resize_image(orig_image, max_resolution)
                        im_tensor = transform_image(resized_image).unsqueeze(0)
                        im_tensor = im_tensor.to(device)
                        
                        # 推理
                        with torch.no_grad():
                            result = birefnet(im_tensor)[-1].sigmoid()
                            
                            # 立即移到CPU释放GPU显存
                            result = result.cpu()
                            
                            # 清理中间张量
                            del im_tensor
                            if device.startswith('cuda'):
                                torch.cuda.empty_cache()
                        
                        # 后处理
                        result = torch.squeeze(F.interpolate(result, size=(h, w), mode='bilinear', align_corners=False))
                        ma = torch.max(result)
                        mi = torch.min(result)
                        result = (result - mi) / (ma - mi + 1e-8)  # 添加小值避免除零
                        
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
                        
                    except Exception as e:
                        print(f"\033[91m处理图片时出错: {str(e)}\033[0m")
                        # 在出错时也要清理显存
                        if device.startswith('cuda'):
                            torch.cuda.empty_cache()
                        raise e
                
                # 批处理间清理
                if enable_memory_efficient and device.startswith('cuda'):
                    torch.cuda.empty_cache()
                    gc.collect()

            new_ims = torch.cat(processed_images, dim=0)
            new_masks = torch.cat(processed_masks, dim=0)

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
