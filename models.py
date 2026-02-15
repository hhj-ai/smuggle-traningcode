import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

class VerifierModel:
    """
    Wrapper for Verifier (DeepSeek-R1-Distill-Qwen-7B).
    """
    def __init__(self, model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", device="cuda"):
        self.device = device
        print(f"Loading Verifier: {model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            # FIXED: Use dict for device_map to avoid accelerate conflicts
            device_map={"": device},
            trust_remote_code=True,
            # H200 Optimization:
            attn_implementation="flash_attention_2"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def verify_claims(self, description):
        prompt = f"Extract distinct, verifiable visual claims from the following description. Format as a bulleted list.\n\nDescription: {description}\n\nClaims:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.6
        )
        
        raw_response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        clean_text = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL).strip()
        
        claims = []
        for line in clean_text.split('\n'):
            cleaned = line.strip().lstrip('-').lstrip('*').strip()
            if len(cleaned) > 5:
                claims.append(cleaned)
                
        return claims, raw_response

    def compute_sequence_log_prob(self, prompt, completion):
        full_prompt = f"Extract distinct, verifiable visual claims from the following description. Format as a bulleted list.\n\nDescription: {prompt}\n\nClaims:"
        full_text = full_prompt + completion
        
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        labels = inputs.input_ids.clone()
        
        prompt_ids = self.tokenizer(full_prompt, return_tensors="pt").input_ids
        prompt_len = prompt_ids.shape[1]
        safe_len = min(prompt_len, labels.shape[1])
        labels[:, :safe_len] = -100
        
        outputs = self.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            labels=labels
        )
        
        valid_token_count = (labels != -100).sum().item()
        if valid_token_count == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
            
        total_log_prob = -outputs.loss * valid_token_count
        return total_log_prob

class VLMModel:
    """
    Wrapper for VLM (Qwen3-VL-8B-Instruct).
    """
    def __init__(self, model_name="Qwen/Qwen3-VL-8B-Instruct", device="cuda"):
        self.device = device
        print(f"Loading VLM: {model_name} with Flash Attention 2...")
        try:
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                # FIXED: Device map safe mode
                device_map={"": device},
                trust_remote_code=True,
                # H200 Optimization:
                attn_implementation="flash_attention_2"
            )
        except Exception as e:
            raise RuntimeError(f"VLM Load Error: {e}")
        self.tokenizer = self.processor.tokenizer

    def generate_description_batch(self, image_inputs, num_generations=4):
        # Qwen-VL Instruct format requires proper structure
        # Simplified here for generation
        text_prompts = ["Describe this image in detail."] * len(image_inputs)
        
        # Note: Qwen processor usually handles 'text' and 'images' list directly
        # But for batch generation, we let the processor handle the padding
        
        inputs = self.processor(
            text=text_prompts,
            images=image_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=1.0,
                num_return_sequences=num_generations
            )
        
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        results = []
        for i in range(len(image_inputs)):
            start = i * num_generations
            results.append(generated_texts[start : start + num_generations])
            
        return results

    def compute_log_probs(self, input_ids, attention_mask, labels):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        valid_count = (labels != -100).sum().item()
        if valid_count == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
            
        return -outputs.loss * valid_count
