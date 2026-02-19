import torch
import os
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, CLIPProcessor, CLIPModel
import easyocr
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class ToolVerifier:
    # 各工具模型的估计显存占用 (MiB)
    TOOL_MEM_MIB = {"dino": 850, "clip": 650, "ocr": 200}

    @staticmethod
    def auto_assign_devices():
        """查询所有可见 GPU 空闲显存，按空闲降序分配 DINO/CLIP/OCR 到最空闲的卡"""
        if not torch.cuda.is_available():
            return {"dino": "cpu", "clip": "cpu", "ocr": "cpu"}

        n_gpus = torch.cuda.device_count()
        if n_gpus == 0:
            return {"dino": "cpu", "clip": "cpu", "ocr": "cpu"}

        # 查询各卡空闲显存
        free = {}
        for i in range(n_gpus):
            f, _ = torch.cuda.mem_get_info(i)
            free[i] = f / (1024 * 1024)  # MiB

        # 按显存需求从大到小分配：DINO > CLIP > OCR
        assignment = {}
        for tool in ["dino", "clip", "ocr"]:
            need = ToolVerifier.TOOL_MEM_MIB[tool]
            # 选空闲最多的卡
            best_gpu = max(free, key=free.get)
            if free[best_gpu] >= need:
                assignment[tool] = f"cuda:{best_gpu}"
                free[best_gpu] -= need  # 扣减预估占用
            else:
                assignment[tool] = "cpu"

        return assignment

    def __init__(self, model_root="./models", device=None, devices=None):
        """
        devices: 显式设备列表 (轮询分配)，用于 CLI 覆盖。
        device:  单设备回退。
        两者都不指定时自动按显存分配。
        """
        if devices and len(devices) > 0:
            dev_list = [torch.device(d) for d in devices]
            assign = {
                "dino": dev_list[0 % len(dev_list)],
                "clip": dev_list[2 % len(dev_list)],
                "ocr":  dev_list[1 % len(dev_list)],
            }
        elif device is not None:
            d = torch.device(device)
            assign = {"dino": d, "clip": d, "ocr": d}
        else:
            raw = self.auto_assign_devices()
            assign = {k: torch.device(v) for k, v in raw.items()}

        self.dino_device = assign["dino"]
        self.ocr_device = assign["ocr"]
        self.clip_device = assign["clip"]
        
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
        print(f"   Devices: DINO→{self.dino_device}, OCR→{self.ocr_device}, CLIP→{self.clip_device}")

        # 1. DINO
        self.dino_processor = AutoProcessor.from_pretrained(dino_path, local_files_only=True)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_path, local_files_only=True).to(self.dino_device)
        self.dino_model.eval()
        print("   - DINO Ready ✅")

        # 2. EasyOCR
        use_gpu = torch.cuda.is_available() and "cuda" in str(self.ocr_device)
        self.ocr_reader = easyocr.Reader(['en'], gpu=use_gpu)
        print(f"   - OCR Ready (GPU: {use_gpu}) ✅")

        # 3. CLIP (权重文件为 .bin 格式，需要绕过 PyTorch 的 torch.load 安全限制)
        _orig_load = torch.load
        torch.load = lambda *args, **kwargs: _orig_load(*args, **{k: v for k, v in kwargs.items() if k != 'weights_only'}, weights_only=False)
        self.clip_model = CLIPModel.from_pretrained(clip_path, local_files_only=True).to(self.clip_device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_path, local_files_only=True)
        torch.load = _orig_load
        self.clip_model.eval()
        print("   - CLIP Ready ✅")

    def verify_claim(self, claim, image_path):
        if not os.path.exists(image_path): return "uncertain", 0.0, "Img Missing"
        try:
            image = Image.open(image_path).convert("RGB")
        except: return "uncertain", 0.0, "Load Error"

        # 降级逻辑：如果 DINO 没加载成功，只跑 CLIP
        if self.dino_model is None:
            score = self._verify_clip(claim, image)
        else:
            score = self._verify_dino(claim, image)

        # 根据置信度分数判断 verdict
        if score <= 0.0:  # 工具失败或无法判断
            return "uncertain", score, "tool failed"
        elif score > 0.5:
            return "correct", score, ""
        else:
            return "incorrect", score, ""

    def _verify_dino(self, claim, image):
        if not self.dino_model: return 0.0
        try:
            inputs = self.dino_processor(images=image, text=claim+".", return_tensors="pt").to(self.dino_device)
            with torch.no_grad():
                outputs = self.dino_model(**inputs)
            # 简化版 Score 提取
            return outputs.logits.sigmoid().max().item()
        except: return 0.0

    def _verify_clip(self, claim, image):
        if not self.clip_model: return 0.0
        try:
            inputs = self.clip_processor(text=[f"a photo of {claim}"], images=image, return_tensors="pt", padding=True).to(self.clip_device)
            with torch.no_grad():
                return self.clip_model(**inputs).logits_per_image.softmax(dim=1)[0][0].item()
        except: return 0.0

    def _batch_verify_dino(self, claims, image):
        """批量 DINO 验证：一张图的多个 claims 拼成一个 query"""
        if not self.dino_model:
            return [0.0] * len(claims)
        try:
            # DINO 支持用 ". " 分隔多个 text query
            combined_text = ". ".join(claims) + "."
            inputs = self.dino_processor(images=image, text=combined_text, return_tensors="pt").to(self.dino_device)
            with torch.no_grad():
                outputs = self.dino_model(**inputs)
            # outputs.logits shape: [1, num_boxes, num_queries]
            # 每个 claim 对应一个 query，取各 query 的最大 box score
            logits = outputs.logits.sigmoid()  # [1, num_boxes, num_queries]
            if logits.dim() == 3 and logits.shape[2] >= len(claims):
                scores = [logits[0, :, q].max().item() for q in range(len(claims))]
            else:
                # fallback: 所有 claims 共享一个 score
                s = logits.max().item()
                scores = [s] * len(claims)
            return scores
        except:
            return [0.0] * len(claims)

    def _batch_verify_clip(self, claims, image):
        """批量 CLIP 验证：一张图的多个 claims 作为 text list"""
        if not self.clip_model:
            return [0.0] * len(claims)
        try:
            texts = [f"a photo of {c}" for c in claims]
            inputs = self.clip_processor(text=texts, images=image, return_tensors="pt", padding=True).to(self.clip_device)
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
            # logits_per_image: [1, num_claims]
            scores = outputs.logits_per_image.softmax(dim=1)[0].tolist()
            return scores
        except:
            return [0.0] * len(claims)

    def verify_claims_batch(self, claims_by_image, preloaded_images=None):
        """
        批量验证 claims，按图片分组，每张图只加载一次。
        claims_by_image: [(image_path, [claim1, claim2, ...]), ...]
        preloaded_images: 可选 dict {image_path: PIL.Image}，避免重复磁盘 IO
        返回: {(image_path, claim): verdict}
        """
        results = {}
        with ThreadPoolExecutor(max_workers=2) as executor:
            for img_path, claims in claims_by_image:
                if not claims:
                    continue
                # 尝试使用预加载图片
                image = None
                if preloaded_images and img_path in preloaded_images:
                    image = preloaded_images[img_path]
                else:
                    if not os.path.exists(img_path):
                        for c in claims:
                            results[(img_path, c)] = "uncertain"
                        continue
                    try:
                        image = Image.open(img_path).convert("RGB")
                    except:
                        for c in claims:
                            results[(img_path, c)] = "uncertain"
                        continue

                # 并行执行 DINO 和 CLIP（它们在不同设备上）
                dino_future = executor.submit(self._batch_verify_dino, claims, image)
                clip_future = executor.submit(self._batch_verify_clip, claims, image)
                dino_scores = dino_future.result()
                clip_scores = clip_future.result()

                for i, c in enumerate(claims):
                    score = max(dino_scores[i], clip_scores[i])
                    if score <= 0.0:
                        results[(img_path, c)] = "uncertain"
                    elif score > 0.5:
                        results[(img_path, c)] = "correct"
                    else:
                        results[(img_path, c)] = "incorrect"
        return results
