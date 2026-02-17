import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from transformers import CLIPProcessor, CLIPModel
import easyocr
import numpy as np

class ToolVerifier:
    def __init__(self, device="cuda", model_root="./models"):
        self.device = device
        self.model_root = model_root
        print(f"🔧 Initializing Verification Tools (Offline Mode: {model_root})...")
        
        # 1. Grounding DINO
        dino_path = os.path.join(model_root, "grounding-dino-base")
        if not os.path.exists(dino_path): dino_path = "IDEA-Research/grounding-dino-base"
        
        try:
            self.dino_processor = AutoProcessor.from_pretrained(dino_path)
            self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(dino_path).to(device)
        except Exception as e:
            print(f"Error DINO: {e}")

        # 2. EasyOCR
        # EasyOCR needs models in ~/.EasyOCR/model, we will handle this in run script
        self.ocr_reader = easyocr.Reader(['en'], gpu=(device == "cuda"))
        
        # 3. CLIP
        clip_path = os.path.join(model_root, "clip-vit-base-patch32")
        if not os.path.exists(clip_path): clip_path = "openai/clip-vit-base-patch32"
        
        try:
            self.clip_model = CLIPModel.from_pretrained(clip_path).to(device)
            self.clip_processor = CLIPProcessor.from_pretrained(clip_path)
        except Exception as e:
            print(f"Error CLIP: {e}")
        
        print("✅ Tools Ready.")

    def verify_claim(self, claim, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            return "uncertain", 0.0, f"Image Error: {e}"

        claim_lower = claim.lower()
        
        # Routing
        ocr_keywords = ["text", "sign", "written", "letter", "word", "says", "read"]
        if any(w in claim_lower for w in ocr_keywords):
            return self._verify_ocr(claim, image_path)
            
        dino_score = self._verify_dino(claim, image)
        clip_score = self._verify_clip(claim, image)
        
        # Ensemble Logic
        final_score = 0.0
        reason = ""
        
        if dino_score > 0.35:
            final_score = 0.5 * dino_score + 0.5 * clip_score
            reason = f"DINO({dino_score:.2f}) + CLIP({clip_score:.2f})"
        else:
            final_score = clip_score * 0.9
            reason = f"CLIP Only ({clip_score:.2f})"
            
        if final_score > 0.65:
            return "correct", final_score, reason
        elif final_score < 0.4:
            return "incorrect", final_score, reason
        else:
            return "uncertain", final_score, reason

    def _verify_dino(self, claim, image):
        prompt = claim if claim.endswith(".") else claim + "."
        try:
            inputs = self.dino_processor(images=image, text=prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.dino_model(**inputs)
            target_sizes = torch.tensor([image.size[::-1]])
            results = self.dino_processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, box_threshold=0.25, text_threshold=0.2, target_sizes=target_sizes
            )[0]
            if len(results['scores']) > 0:
                return results['scores'].max().item()
            return 0.0
        except:
            return 0.0

    def _verify_clip(self, claim, image):
        prompts = [f"A photo of {claim}", "A photo of something else", "Noise", "Opposite"]
        try:
            inputs = self.clip_processor(text=prompts, images=image, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
            return probs[0][0].item()
        except:
            return 0.0

    def _verify_ocr(self, claim, image_path):
        try:
            results = self.ocr_reader.readtext(image_path)
            detected_texts = " ".join([res[1].lower() for res in results])
            target_words = [w for w in claim.lower().split() if len(w) > 3 and w not in ["text","says"]]
            match_count = sum(1 for w in target_words if w in detected_texts)
            if not target_words: return "uncertain", 0.0, "No targets"
            score = match_count / len(target_words)
            if score > 0.8: return "correct", score, "OCR Match"
            elif score < 0.2: return "incorrect", score, "OCR Mismatch"
            return "uncertain", score, "Partial"
        except:
            return "uncertain", 0.0, "OCR Error"
