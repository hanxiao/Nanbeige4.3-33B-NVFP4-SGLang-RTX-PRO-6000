#!/usr/bin/env python3
"""Code correctness: 41 HumanEval problems (every 4th), x2 samples, thinking on, max_tokens 24000
(so answers are not truncated), model sampling defaults. Extract the last python block, run the
official tests in a subprocess (10 s timeout).

    SGLANG_API_KEY=... python3 codetest.py <url> <out.json> [C]
"""
import json, os, re, subprocess, sys, tempfile, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

URL, OUT = sys.argv[1], sys.argv[2]
C = int(sys.argv[3]) if len(sys.argv) > 3 else 8
KEY = os.environ["SGLANG_API_KEY"]
H = {"Content-Type": "application/json", "Authorization": " ".join(["Bearer", KEY])}
PROBS = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "he40.json")))
MODEL = "Nanbeige4.3-33B"


def ask(p):
    msg = ("Complete the following Python function. Reply with the complete function (including the "
           "signature and any imports it needs) in a single ```python code block.\n\n```python\n" + p["prompt"] + "\n```")
    body = {"model": MODEL, "messages": [{"role": "user", "content": msg}], "max_tokens": 24000}
    t = time.time()
    r = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), H)
    j = json.loads(urllib.request.urlopen(r, timeout=3600).read())
    ch = j["choices"][0]
    return ch["message"].get("content") or "", ch.get("finish_reason"), j["usage"]["completion_tokens"], time.time() - t


def check(p, text):
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if not blocks:
        return False, "no code"
    code = blocks[-1]
    prog = code + "\n\n" + p["test"] + "\n\ncheck(" + p["entry_point"] + ")\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(prog)
    try:
        r = subprocess.run([sys.executable, f.name], capture_output=True, timeout=10, text=True)
        return r.returncode == 0, r.stderr[-300:]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        os.unlink(f.name)


def one(arg):
    p, k = arg
    try:
        text, fin, ntok, dt = ask(p)
        ok, err = check(p, text)
    except Exception as e:
        text, fin, ntok, dt, ok, err = "", "error", 0, 0, False, repr(e)[:300]
    return {"task": p["task_id"], "k": k, "pass": ok, "finish": fin, "ntok": ntok, "sec": round(dt, 1), "err": err if not ok else ""}


jobs = [(p, k) for k in range(2) for p in PROBS]
with ThreadPoolExecutor(C) as ex:
    res = list(ex.map(one, jobs))
n = len(res)
npass = sum(r["pass"] for r in res)
nlen = sum(r["finish"] == "length" for r in res)
summary = {"pass": npass, "n": n, "truncated": nlen, "errors": sum(r["finish"] == "error" for r in res),
           "mean_tokens": round(sum(r["ntok"] for r in res) / n)}
json.dump({"summary": summary, "rows": res}, open(OUT, "w"), indent=1)
print(f"code: {npass}/{n} pass, truncated {nlen}, errors {summary['errors']}, mean tokens {summary['mean_tokens']}")
