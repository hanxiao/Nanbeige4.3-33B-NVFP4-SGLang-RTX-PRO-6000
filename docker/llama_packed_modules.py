"""Let SGLang v0.5.22 serve a ModelOpt MIXED_PRECISION checkpoint of a Llama model.

ModelOptMixedPrecisionConfig looks up each layer's format in hf_quant_config's quantized_layers,
which names the checkpoint's modules (q_proj, k_proj, v_proj, gate_proj, up_proj). SGLang's Llama
fuses them into qkv_proj and gate_up_proj, and resolves the fused name through the model class's
packed_modules_mapping. Qwen3.5 declares one; LlamaForCausalLM does not, so the fused layers found
no entry, were built unquantized, and loading the packed NVFP4 weights failed with
`assert param_data.shape == loaded_weight.shape` in MergedColumnParallelLinear.weight_loader.
This declares the mapping on LlamaForCausalLM (inherited by its subclasses).

    python3 llama_packed_modules.py   # edits the installed package; fails if the text moved
"""
import pathlib

import sglang

P = pathlib.Path(sglang.__file__).parent / "srt/models/llama.py"
s = P.read_text()
assert "packed_modules_mapping" not in s, "llama.py already declares packed_modules_mapping"
old = "class LlamaForCausalLM(nn.Module):\n"
assert s.count(old) == 1, "llama.py text moved"
new = old + ('    packed_modules_mapping = {\n'
             '        "qkv_proj": ["q_proj", "k_proj", "v_proj"],\n'
             '        "gate_up_proj": ["gate_proj", "up_proj"],\n'
             '    }\n\n')
P.write_text(s.replace(old, new))
print("patched", P)
