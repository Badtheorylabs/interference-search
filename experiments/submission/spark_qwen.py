"""Run the pinned original MLX 4-bit Qwen weights on Spark through Transformers.

The affine codes, scales, and biases are expanded to BF16. This preserves the
source checkpoint and quantized weight values; the kernels and accumulation
differ from MLX. Comparisons are internal to this backend, not legacy parity.
"""
import json
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from experiments.submission.common import digest, write_json

REVISION = "3b1b1768f8f8cf8351c712464f906e86c2b8269e"


class SparkQwen:
    def __init__(self, path, receipt, device="cuda"):
        path = Path(path)
        raw_config = json.loads((path / "config.json").read_text())
        quant = raw_config["quantization"]
        assert quant == {"group_size":64,"bits":4}
        packed = load_file(str(path / "model.safetensors"))
        weights = {}
        started=time.perf_counter()
        for name, value in packed.items():
            if name.endswith((".scales",".biases")):
                continue
            prefix=name.removesuffix(".weight")
            if prefix+".scales" in packed:
                codes=((value.to(torch.int64).unsqueeze(-1) >> (4*torch.arange(8))) & 15).reshape(value.shape[0],-1)
                scales=packed[prefix+".scales"].float().repeat_interleave(64,dim=-1)
                biases=packed[prefix+".biases"].float().repeat_interleave(64,dim=-1)
                value=(codes.float()*scales+biases).to(torch.bfloat16)
                assert value.shape[-1] == packed[prefix+".scales"].shape[-1]*64
            weights[name]=value.to(torch.bfloat16)
        config=AutoConfig.from_pretrained(path,local_files_only=True)
        # Prevent Transformers from requantizing already expanded values.
        if hasattr(config,"quantization_config"):
            del config.quantization_config
        self.model=AutoModelForCausalLM.from_config(config,dtype=torch.bfloat16,attn_implementation="sdpa")
        missing, unexpected=self.model.load_state_dict(weights,strict=False,assign=True)
        if unexpected or set(missing)-{"lm_head.weight"}:
            raise ValueError((missing,unexpected))
        self.model.tie_weights()
        self.model=self.model.to(device).eval()
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True,padding_side="left")
        self.tokenizer.pad_token_id=self.tokenizer.eos_token_id
        self.device=device
        self.receipt={"source_repo":"mlx-community/Qwen3-1.7B-4bit","source_revision":REVISION,
                      "source_safetensors_sha256":digest(path/"model.safetensors"),
                      "config_sha256":digest(path/"config.json"),"dtype":"bfloat16",
                      "backend":"Transformers SDPA; affine 4-bit weights expanded to BF16",
                      "legacy_mlx_kernel_parity":"not verified; do not compare absolute results with legacy counts",
                      "load_seconds":time.perf_counter()-started,"device":device,
                      "parameters":sum(p.numel() for p in self.model.parameters())}
        write_json(receipt,self.receipt)

    def prompt(self, messages):
        return self.tokenizer.apply_chat_template(messages,add_generation_prompt=True,
                                                 tokenize=False,enable_thinking=False)

    def generate(self, prompts, cap, seed, temperature=0.8):
        torch.manual_seed(seed)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(seed)
        batch=self.tokenizer(prompts,padding=True,return_tensors="pt",add_special_tokens=False).to(self.device)
        started=time.perf_counter()
        with torch.inference_mode():
            output=self.model.generate(**batch,max_new_tokens=cap,do_sample=temperature>0,
                       temperature=temperature if temperature>0 else None,top_p=0.95,top_k=20,
                       pad_token_id=self.tokenizer.pad_token_id,eos_token_id=self.tokenizer.eos_token_id,
                       use_cache=True)
        if self.device=="cuda":
            torch.cuda.synchronize()
        elapsed=time.perf_counter()-started
        generated=output[:,batch["input_ids"].shape[1]:].tolist()
        rows=[]
        for ids, prompt_count in zip(generated,batch["attention_mask"].sum(1).tolist()):
            if self.tokenizer.eos_token_id in ids:
                ids=ids[:ids.index(self.tokenizer.eos_token_id)+1]
            rows.append({"text":self.tokenizer.decode(ids,skip_special_tokens=True),
                         "token_ids":ids,"generated_tokens":len(ids),"prompt_tokens":prompt_count,
                         "batch_wall_seconds":elapsed,"batch_size":len(prompts)})
        return rows
