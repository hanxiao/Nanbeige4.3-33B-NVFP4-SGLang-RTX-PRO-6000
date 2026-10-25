"""Edit a downloaded checkpoint before SGLang starts.

    prep.py <model dir> [steps]   steps: comma list of gen, stripkv, w4h (default gen)

gen      write Nanbeige's recommended agentic sampling (temperature 1.0, top_p 0.95) into
         generation_config.json, which --sampling-defaults=model serves as the default for requests
         that send none. The published generation_config.json carries no sampling fields.
stripkv  remove kv_cache_scheme / kv_cache_quant_algo (ModelOpt FP8 KV scales) so that a BF16 KV
         cache is safe; a checkpoint that keeps them produces garbage with --kv-cache-dtype=bf16.
w4h      serve an NVFP4 lm_head as W4A16_NVFP4 (Marlin, BF16 activations); config only.
w4mlp    serve the NVFP4 MLPs as W4A16_NVFP4 too (same weights, BF16 activations); config only.
"""
import json, os, sys

d = sys.argv[1]
steps = (sys.argv[2] if len(sys.argv) > 2 else "gen").split(",")


def load(f):
    p = os.path.join(d, f)
    return (p, json.load(open(p))) if os.path.exists(p) else (p, None)


def strip(o):
    if isinstance(o, dict):
        for k in ("kv_cache_scheme", "kv_cache_quant_algo"):
            o.pop(k, None)
        for v in o.values():
            strip(v)
    elif isinstance(o, list):
        for v in o:
            strip(v)


if "gen" in steps:
    p, g = load("generation_config.json")
    g = g or {}
    g.update(temperature=1.0, top_p=0.95, do_sample=True)
    json.dump(g, open(p, "w"), indent=2)
    print("prep: generation_config", g, flush=True)
for f in ("config.json", "hf_quant_config.json"):
    p, c = load(f)
    if c is None:
        continue
    if "stripkv" in steps:
        strip(c)
    if "w4h" in steps:
        q = c.get("quantization_config", c.get("quantization", c))
        ql = q.get("quantized_layers", {})
        if "lm_head" in ql:
            ql["lm_head"] = {"quant_algo": "W4A16_NVFP4", "group_size": ql["lm_head"].get("group_size", 16)}
    if "w4mlp" in steps:
        q = c.get("quantization_config", c.get("quantization", c))
        ql = q.get("quantized_layers", {})
        for k, v in ql.items():
            if ".mlp." in k and str(v.get("quant_algo", "")).upper() == "NVFP4":
                ql[k] = {"quant_algo": "W4A16_NVFP4", "group_size": v.get("group_size", 16)}
    json.dump(c, open(p, "w"), indent=2)
    print("prep:", f, steps, flush=True)
