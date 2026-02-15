import torch
import re
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor

class VerifierModel:
    """
    Wrapper for Verifier (DeepSeek-R1-Distill-Qwen-7B).
    Standard loading with Flash Attention 2.
    """
    def __init__(self, model_name="./models/DeepSeek-R1-Distill-Qwen-7B", device="cuda"):
        self.device = device
        
        # 1. 路径检查 (本地优先，远程回退)
        if not os.path.exists(model_name):
            print(f"⚠️ Warning: Local model path '{model_name}' not found. Fallback to HF ID.")
            if "models/" in model_name: 
                model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        
        print(f"Loading Verifier from: {model_name} ...")
        
        # 2. 加载 Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # 3. 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": device}, # 适配 Accelerator
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )

    def verify_claims(self, description):
        """
        生成验证点 (Claims)
        """
        prompt = f"Extract distinct, verifiable visual claims from the following description. Format as a bulleted list.\n\nDescription: {description}\n\nClaims:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.6
        )
        
        raw_response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # 清理思维链 (DeepSeek R1 特性)
        clean_text = re.sub(r'<think>.*?</think>', '', raw_response, flags=re.DOTALL).strip()
        
        claims = []
        for line in clean_text.split('\n'):
            # 简单的清洗逻辑
            cleaned = line.strip().lstrip('-').lstrip('*').strip()
            if len(cleaned) > 5:
                claims.append(cleaned)
                
        return claims, raw_response

    def compute_sequence_log_prob(self, prompt, completion):
        """
        计算生成序列的对数概率 (用于 PPO/DPO 训练)
        """
        full_prompt = f"Extract distinct, verifiable visual claims from the following description. Format as a bulleted list.\n\nDescription: {prompt}\n\nClaims:"
        full_text = full_prompt + completion
        
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        labels = inputs.input_ids.clone()
        
        # Mask 掉 Prompt 部分的 Loss，只计算 Completion 部分
        prompt_ids = self.tokenizer(full_prompt, return_tensors="pt").input_ids
        safe_len = min(prompt_ids.shape[1], labels.shape[1])
        labels[:, :safe_len] = -100
        
        outputs = self.model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            labels=labels
        )
        
        # 计算平均 Log Prob
        valid_token_count = (labels != -100).sum().item()
        if valid_token_count == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
            
        return -outputs.loss * valid_token_count

class VLMModel:
    """
    Wrapper for VLM (Qwen3-VL-8B-Instruct).
    Standard loading (Requires latest transformers library).
    """
    def __init__(self, model_name="./models/Qwen3-VL-8B-Instruct", device="cuda"):
        self.device = device
        
        # 1. 路径检查
        if not os.path.exists(model_name):
            print(f"⚠️ Warning: Local model path '{model_name}' not found. Fallback to HF ID.")
            if "models/" in model_name:
                model_name = "Qwen/Qwen3-VL-8B-Instruct"

        print(f"Loading VLM from: {model_name} ...")
        
        try:
            # 2. 加载 Processor
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            
            # 3. 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map={"": device},
                trust_remote_code=True,
                attn_implementation="flash_attention_2"
            )
            
        except Exception as e:
            print(f"❌ VLM Load Error: {e}")
            raise RuntimeError(f"VLM Load Error: {e}")
            
        self.tokenizer = self.processor.tokenizer

    def generate_description_batch(self, image_inputs, num_generations=4):
        """
        批量生成图片描述
        """
        text_prompts = ["Describe this image in detail."] * len(image_inputs)
        
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
        
        # 重组结果: [batch_size, num_generations]
        results = []
        for i in range(len(image_inputs)):
            start = i * num_generations
            results.append(generated_texts[start : start + num_generations])
            
        return results

    def compute_log_probs(self, input_ids, attention_mask, labels):
        """
        计算 VLM 的 Loss (用于 PPO Update)
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        valid_count = (labels != -100).sum().item()
        if valid_count == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
            
        return -outputs.loss * valid_count
