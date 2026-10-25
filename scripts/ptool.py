#!/usr/bin/env python3
"""Parallel tool calls over the Anthropic /v1/messages endpoint: 8 prompts that need 2-4 independent
tool calls in one turn, each asked twice (thinking on, model sampling defaults). A case passes when
the reply has the expected number of tool_use blocks, the expected tool names, valid inputs that
contain every expected argument value, and stop_reason tool_use.

    SGLANG_API_KEY=... python3 ptool.py <url> <out.json>
"""
import json, os, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL, OUT = sys.argv[1], sys.argv[2]
KEY = os.environ["SGLANG_API_KEY"]
H = {"Content-Type": "application/json", "Authorization": " ".join(["Bearer", KEY]), "anthropic-version": "2023-06-01"}
MODEL = "Nanbeige4.3-33B"
STREAM = os.environ.get("STREAM", "0") == "1"
TOOLS = [
    {"name": "get_weather", "description": "Current weather for a city", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}},
    {"name": "read_file", "description": "Read a file from the repository", "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    {"name": "run_command", "description": "Run a shell command", "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "search_web", "description": "Search the web", "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "get_stock_price", "description": "Latest stock price for a ticker", "input_schema": {"type": "object", "properties": {"ticker": {"type": "string"}}, "required": ["ticker"]}},
]
SYS = "You are an agent. When several independent tool calls are needed, issue all of them in the same turn, in parallel."
CASES = [
    ("What's the weather right now in Paris, Tokyo and Berlin?", "get_weather", ["paris", "tokyo", "berlin"]),
    ("Read src/main.py and src/utils.py so we can compare them.", "read_file", ["src/main.py", "src/utils.py"]),
    ("Get the latest stock prices for AAPL, NVDA, MSFT and GOOG.", "get_stock_price", ["aapl", "nvda", "msft", "goog"]),
    ("同时查一下北京、上海和深圳现在的天气。", "get_weather", ["北京|beijing", "上海|shanghai", "深圳|shenzhen"]),
    ("Run `git status` and `ls -la` in the repo (they are independent).", "run_command", ["git status", "ls -la"]),
    ("Open README.md, package.json and tsconfig.json.", "read_file", ["readme.md", "package.json", "tsconfig.json"]),
    ("Search the web for 'SGLang DFlash' and for 'NVFP4 SM120 GEMM'.", "search_web", ["dflash", "nvfp4"]),
    ("I need weather for London and New York, please.", "get_weather", ["london", "new york"]),
]


def call_stream(body):
    """Stream /v1/messages and rebuild the non-streaming reply: content blocks and stop_reason."""
    body = dict(body, stream=True)
    r = urllib.request.Request(URL + "/v1/messages", json.dumps(body).encode(), H)
    blocks, stop = {}, None
    with urllib.request.urlopen(r, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            try:
                ev = json.loads(line[5:])
            except Exception:
                continue
            ty = ev.get("type")
            if ty == "content_block_start":
                cb = dict(ev["content_block"])
                cb["_json"] = ""
                blocks[ev["index"]] = cb
            elif ty == "content_block_delta":
                d = ev["delta"]
                b = blocks.setdefault(ev["index"], {"type": "?", "_json": ""})
                if d.get("type") == "input_json_delta":
                    b["_json"] += d.get("partial_json", "")
                elif d.get("type") == "text_delta":
                    b["text"] = b.get("text", "") + d.get("text", "")
            elif ty == "message_delta":
                stop = ev.get("delta", {}).get("stop_reason", stop)
    content = []
    for i in sorted(blocks):
        b = blocks[i]
        if b.get("type") == "tool_use":
            try:
                b["input"] = json.loads(b["_json"]) if b["_json"] else (b.get("input") or {})
            except Exception:
                b["input"] = {"_invalid_json": b["_json"]}
        b.pop("_json", None)
        content.append(b)
    return {"content": content, "stop_reason": stop}


def one(arg):
    (prompt, name, want), k = arg
    body = {"model": MODEL, "max_tokens": 8000, "system": SYS, "tools": TOOLS, "messages": [{"role": "user", "content": prompt}]}
    try:
        if STREAM:
            j = call_stream(body)
        else:
            r = urllib.request.Request(URL + "/v1/messages", json.dumps(body).encode(), H)
            j = json.loads(urllib.request.urlopen(r, timeout=600).read())
    except Exception as e:
        return {"prompt": prompt, "k": k, "pass": False, "err": repr(e)[:200]}
    tu = [c for c in j.get("content", []) if c.get("type") == "tool_use"]
    names = [c["name"] for c in tu]
    vals = [json.dumps(c.get("input", {}), ensure_ascii=False).lower() for c in tu]
    hit = all(any(any(a in v for a in w.split("|")) for v in vals) for w in want)
    ok = len(tu) == len(want) and all(n == name for n in names) and hit and j.get("stop_reason") == "tool_use"
    return {"prompt": prompt, "k": k, "pass": ok, "n_calls": len(tu), "names": names, "inputs": [c.get("input") for c in tu],
            "stop": j.get("stop_reason")}


jobs = [(c, k) for k in range(2) for c in CASES]
with ThreadPoolExecutor(4) as ex:
    res = list(ex.map(one, jobs))
npass = sum(r["pass"] for r in res)
json.dump({"pass": npass, "n": len(res), "rows": res}, open(OUT, "w"), indent=1, ensure_ascii=False)
print(f"ptool{' stream' if STREAM else ''}: {npass}/{len(res)} pass; calls per case {[r.get('n_calls') for r in res]}")
