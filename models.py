import torch
import re
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    AutoConfig
)

# Flash Attention 2 检测
try:
    import flash_attn  # noqa: F401
    _ATTN_IMPL = "flash_attention_2"
except ImportError:
    _ATTN_IMPL = "sdpa"


def _unwrap_model(model):
    """Unwrap a model from DDP / accelerate / torch.compile wrappers."""
    try:
        from accelerate import unwrap_model
        model = unwrap_model(model)
    except ImportError:
        pass

    while hasattr(model, 'module') and model is not model.module:
        model = model.module

    if hasattr(model, '_orig_mod'):
        model = model._orig_mod

    return model


class VerifierModel:
    def __init__(self, model_name="./models/DeepSeek-R1-Distill-Qwen-7B", device="cuda",
                 tokenizer_name=None):
        self.device = device
        tokenizer_path = tokenizer_name or model_name
        print(f"Loading Verifier: {model_name} on {device}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            trust_remote_code=True,
            attn_implementation=_ATTN_IMPL,
            local_files_only=True
        )
        self.model.eval()

        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

    def verify_claims(self, description):
        prompt = f"Extract distinct, verifiable visual claims from the following description. Format as a bulleted list.\n\nDescription: {description}\n\nClaims:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        model_to_gen = _unwrap_model(self.model)

        with torch.no_grad():
            outputs = model_to_gen.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.6, pad_token_id=self.tokenizer.pad_token_id)
        raw = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        clean = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        claims = [line.strip().lstrip('-*').strip() for line in clean.split('\n') if len(line.strip()) > 5]
        return claims, raw

    def compute_sequence_log_prob(self, prompt, completion):
        full_text = f"Extract distinct, verifiable visual claims from the following description. Format as a bulleted list.\n\nDescription: {prompt}\n\nClaims:" + completion
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        labels = inputs.input_ids.clone()
        prompt_text = f"Extract distinct, verifiable visual claims from the following description. Format as a bulleted list.\n\nDescription: {prompt}\n\nClaims:"
        prompt_len = self.tokenizer(prompt_text, return_tensors="pt").input_ids.shape[1]
        labels[:, :min(prompt_len, labels.shape[1])] = -100

        model_to_use = _unwrap_model(self.model)

        outputs = model_to_use(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, labels=labels)
        valid_tokens = (labels != -100).sum()
        if valid_tokens == 0: return torch.tensor(0.0).to(self.device)
        return -outputs.loss * valid_tokens

    def verify_claims_batch(self, descriptions, max_batch=16):
        """批量提取 claims，内部按 max_batch 分片避免 OOM"""
        all_claims, all_raws = [], []
        model_to_gen = _unwrap_model(self.model)
        prompt_tpl = "Extract distinct, verifiable visual claims from the following description. Format as a bulleted list.\n\nDescription: {}\n\nClaims:"
        for i in range(0, len(descriptions), max_batch):
            batch_descs = descriptions[i:i + max_batch]
            prompts = [prompt_tpl.format(d) for d in batch_descs]
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                outputs = model_to_gen.generate(
                    **inputs, max_new_tokens=256, do_sample=True,
                    temperature=0.6, pad_token_id=self.tokenizer.pad_token_id
                )
            # 每个样本的 prompt 长度不同（因 padding），用 input_ids 长度截断
            for j in range(len(batch_descs)):
                prompt_len = (inputs.attention_mask[j] == 1).sum().item()
                gen_ids = outputs[j][prompt_len:]
                raw = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
                clean = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
                claims = [line.strip().lstrip('-*').strip() for line in clean.split('\n') if len(line.strip()) > 5]
                all_claims.append(claims)
                all_raws.append(raw)
        return all_claims, all_raws

    def compute_sequence_log_prob_batch(self, prompts, completions, max_batch=16):
        """批量计算多个 (prompt, completion) 对的 sequence log-prob"""
        prompt_tpl = "Extract distinct, verifiable visual claims from the following description. Format as a bulleted list.\n\nDescription: {}\n\nClaims:"
        model_to_use = _unwrap_model(self.model)
        log_probs = []
        for i in range(0, len(prompts), max_batch):
            batch_prompts = prompts[i:i + max_batch]
            batch_completions = completions[i:i + max_batch]
            full_texts = [prompt_tpl.format(p) + c for p, c in zip(batch_prompts, batch_completions)]
            prompt_texts = [prompt_tpl.format(p) for p in batch_prompts]
            # tokenize full texts with padding
            full_enc = self.tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            # get per-sample prompt lengths (without padding)
            prompt_lens = [self.tokenizer(pt, return_tensors="pt").input_ids.shape[1] for pt in prompt_texts]
            # build labels: mask prompt + padding with -100
            labels = full_enc.input_ids.clone()
            for j in range(len(batch_prompts)):
                labels[j, :min(prompt_lens[j], labels.shape[1])] = -100
            # mask padding tokens
            labels[full_enc.attention_mask == 0] = -100
            # forward pass
            outputs = model_to_use(
                input_ids=full_enc.input_ids,
                attention_mask=full_enc.attention_mask,
                labels=labels
            )
            # compute per-sample log-prob (not batch-averaged loss)
            logits = outputs.logits[:, :-1, :]  # [B, T-1, V]
            shift_labels = labels[:, 1:]  # [B, T-1]
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            per_token_loss = loss_fct(logits.reshape(-1, logits.size(-1)), shift_labels.reshape(-1))
            per_token_loss = per_token_loss.view(shift_labels.size())  # [B, T-1]
            for j in range(len(batch_prompts)):
                valid_mask = shift_labels[j] != -100
                valid_count = valid_mask.sum()
                if valid_count == 0:
                    log_probs.append(torch.tensor(0.0, device=self.device))
                else:
                    log_probs.append(-per_token_loss[j][valid_mask].sum())
        return log_probs


class VLMModel:
    def __init__(self, model_name="./models/Qwen3-VL-8B-Instruct", device="cuda",
                 processor_name=None):
        self.device = device
        processor_path = processor_name or model_name
        print(f"Loading VLM: {model_name} on {device}")

        self.processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)

        from transformers import AutoModel
        try:
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
            ModelClass = Qwen3VLForConditionalGeneration
        except ImportError:
            ModelClass = AutoModel

        self.model = ModelClass.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            trust_remote_code=True,
            attn_implementation=_ATTN_IMPL,
            local_files_only=True
        )

        self.model.eval()
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"

        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

    def generate_description_batch(self, image_inputs, num_generations=4):
        print(f"[DEBUG-VLM] Starting generate_description_batch for {len(image_inputs)} images, {num_generations} generations", flush=True)
        messages_batch = []
        for _ in image_inputs:
            messages = [{"role": "user", "content": [{"type": "image", "image": None}, {"type": "text", "text": "Describe this image in detail."}]}]
            messages_batch.append(messages)
        text_prompts = [self.processor.apply_chat_template(msg, add_generation_prompt=True) for msg in messages_batch]
        print(f"[DEBUG-VLM] Applied chat template, preparing inputs", flush=True)
        inputs = self.processor(text=text_prompts, images=image_inputs, padding=True, return_tensors="pt").to(self.device)
        print(f"[DEBUG-VLM] Inputs prepared, shape: {inputs.input_ids.shape}", flush=True)

        model_to_gen = _unwrap_model(self.model)

        print(f"[DEBUG-VLM] Model unpacked, starting generation", flush=True)
        with torch.no_grad():
            generated_ids = model_to_gen.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=1.0, num_return_sequences=num_generations)
        print(f"[DEBUG-VLM] Generation complete, shape: {generated_ids.shape}", flush=True)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids.repeat_interleave(num_generations, dim=0), generated_ids)]
        texts = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
        print(f"[DEBUG-VLM] Decoded {len(texts)} texts", flush=True)
        return [texts[i * num_generations : (i + 1) * num_generations] for i in range(len(image_inputs))]

    def compute_log_probs(self, input_ids, attention_mask, labels):
        model_to_use = _unwrap_model(self.model)

        outputs = model_to_use(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return -outputs.loss * (labels != -100).sum()

    def compute_log_probs_batch(self, images, texts, max_batch=8):
        """批量计算多个 (image, text) 对的 log-prob"""
        model_to_use = _unwrap_model(self.model)
        log_probs = []
        for i in range(0, len(images), max_batch):
            batch_imgs = images[i:i + max_batch]
            batch_txts = texts[i:i + max_batch]
            # 构造 messages
            msgs_list = []
            for img, txt in zip(batch_imgs, batch_txts):
                msg = [
                    {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "Describe."}]},
                    {"role": "assistant", "content": [{"type": "text", "text": txt}]}
                ]
                msgs_list.append(msg)
            text_inputs = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=False) for m in msgs_list]
            inputs = self.processor(text=text_inputs, images=batch_imgs, padding=True, return_tensors="pt").to(self.device)
            labels = inputs.input_ids.clone()
            # mask padding
            labels[inputs.attention_mask == 0] = -100
            # forward — 需要传入 pixel_values / image_grid_thw 等视觉输入
            forward_kwargs = {k: v for k, v in inputs.items()}
            forward_kwargs["labels"] = labels
            outputs = model_to_use(**forward_kwargs)
            # per-sample log-prob
            logits = outputs.logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            per_token_loss = loss_fct(logits.reshape(-1, logits.size(-1)), shift_labels.reshape(-1))
            per_token_loss = per_token_loss.view(shift_labels.size())
            for j in range(len(batch_imgs)):
                valid_mask = shift_labels[j] != -100
                if valid_mask.sum() == 0:
                    log_probs.append(torch.tensor(0.0, device=self.device))
                else:
                    log_probs.append(-per_token_loss[j][valid_mask].sum())
        return log_probs
