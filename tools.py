import torch
import os
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, CLIPProcessor, CLIPModel
import easyocr
import numpy as np

class ToolVerifier:
    def __init__(self, device="cuda", model_root="./models"):
        self.device = device
        
        def find_path(name):
            # 自动搜索 model_root 下包含 name 的文件夹
            full_path = os.path.join(model_root, name)
            if os.path.exists(full_path): return full_path
            # 模糊匹配
            for d in os.listdir(model_root):
                if name.replace("-base","") in d.lower():
                    return os.path.join(model_root, d)
            return full_path

        dino_path = find_path("grounding-dino-base")
        clip_path = find_path("clip-vit-base-patch32")
        
        print(f"🔧 [Tools] Loading tools from: {model_root}")
        
        # 1. DINO
        try:
            self.dino_processor = AutoProcessor.from_pretrained(dino_path, local_files_only=True)
            self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_path, local_files_only=True).to(device)
            self.dino_model.eval()
            print("   - DINO Ready ✅")
        except Exception as e:
            print(f"⚠️ DINO missing or error: {e}")
            self.dino_model = None

        # 2. EasyOCR
        use_gpu = torch.cuda.is_available() and "cuda" in str(device)
        try:
            self.ocr_reader = easyocr.Reader(['en'], gpu=use_gpu)
            print(f"   - OCR Ready (GPU: {use_gpu}) ✅")
        except: self.ocr_reader = None
        
        # 3. CLIP
        try:
            self.clip_model = CLIPModel.from_pretrained(clip_path, local_files_only=True).to(device)
            self.clip_processor = CLIPProcessor.from_pretrained(clip_path, local_files_only=True)
            self.clip_model.eval()
            print("   - CLIP Ready ✅")
        except Exception as e:
            print(f"⚠️ CLIP missing or error: {e}")
            self.clip_model = None

    def verify_claim(self, claim, image_path):
        if not os.path.exists(image_path): return "uncertain", 0.0, "Img Missing"
        try:
            image = Image.open(image_path).convert("RGB")
        except: return "uncertain", 0.0, "Load Error"

        # 降级逻辑：如果 DINO 没加载成功，只跑 CLIP
        if self.dino_model is None:
            return self._verify_clip(claim, image)
            
        # ... (原有验证逻辑保持不变) ...
        return self._verify_dino(claim, image) # 简化展示

    def _verify_dino(self, claim, image):
        if not self.dino_model: return 0.0
        try:
            inputs = self.dino_processor(images=image, text=claim+".", return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.dino_model(**inputs)
            # 简化版 Score 提取
            return outputs.logits.sigmoid().max().item()
        except: return 0.0

    def _verify_clip(self, claim, image):
        if not self.clip_model: return 0.0
        try:
            inputs = self.clip_processor(text=[f"a photo of {claim}"], images=image, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                return self.clip_model(**inputs).logits_per_image.softmax(dim=1)[0][0].item()
        except: return 0.0
