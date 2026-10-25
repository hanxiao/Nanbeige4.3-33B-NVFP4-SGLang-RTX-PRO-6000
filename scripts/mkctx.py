"""Cut the coding-agent contexts (SGLang v0.5.22 source) to ~30k/60k/120k Nanbeige tokens.

    python3 mkctx.py <Nanbeige4.3-33B dir> <sglang v0.5.22 checkout>
"""
import os, sys, glob
from tokenizers import Tokenizer
tok = Tokenizer.from_file(sys.argv[1] + "/tokenizer.json")
ROOT = sys.argv[2] + "/python"
S = ROOT + "/sglang/srt/"
first = ["mem_cache/radix_cache.py", "function_call/qwen3_coder_detector.py", "mem_cache/base_prefix_cache.py",
         "function_call/base_format_detector.py", "mem_cache/evict_policy.py", "function_call/core_types.py"]
rest = []
for d in ["mem_cache", "function_call", "managers", "speculative"]:
    for f in sorted(glob.glob(S + d + "/**/*.py", recursive=True)):
        r = f[len(S):]
        if r not in first and os.path.getsize(f) > 2000 and os.path.getsize(f) < 60000:
            rest.append(r)
order = first + rest
for target in (30000, 60000, 120000):
    parts, n = [], 0
    for r in order:
        body = open(S + r).read()
        blk = "### File: python/sglang/srt/" + r + "\n```python\n" + body + "\n```\n\n"
        t = len(tok.encode(blk).ids)
        if n + t > target - 1500:
            continue
        parts.append(blk)
        n += t
        if n > target - 3000:
            break
    txt = "".join(parts)
    open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ctx_%d.txt" % (target // 1000)), "w").write(txt)
    print(target, n, len(txt), len(parts))
