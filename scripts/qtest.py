#!/usr/bin/env python3
"""Quality / no-regression checks used for the 2026-10-23 consolidation (not part of the repo).

    SGLANG_API_KEY=... python3 qtest.py agree <url> <out.json>          # 120 greedy prompts, C=1, 256 tok
    SGLANG_API_KEY=... python3 qtest.py gsm <url> <out.json> [C] [N]    # GSM8K test[:N], greedy, thinking off
    SGLANG_API_KEY=... python3 qtest.py longloop <url> <out.json>       # 8 long greedy answers, 4000 tok
    python3 qtest.py cmp <a.json> <b.json>                              # agreement between two agree runs
"""
import json, os, re, sys, time, zlib, urllib.request
from concurrent.futures import ThreadPoolExecutor

MODEL = "Nanbeige4.3-33B"
HERE = os.path.dirname(os.path.abspath(__file__))

CONCEPTS = ["a hash map", "TCP congestion control", "garbage collection in Java", "the CAP theorem",
            "gradient descent", "public-key cryptography", "a B-tree", "the Fourier transform",
            "inflation", "natural selection", "the immune system", "plate tectonics",
            "the French Revolution", "the Silk Road", "quantum entanglement", "a compiler's register allocator",
            "attention in transformers", "the Raft consensus algorithm", "Bloom filters", "virtual memory"]
CODE = ["reverses a linked list", "merges two sorted arrays", "computes the edit distance of two strings",
        "parses an ISO 8601 date without libraries", "implements binary search on a rotated array",
        "finds all anagrams of a word in a list", "flattens a nested dictionary with dotted keys",
        "implements a token-bucket rate limiter", "returns the n-th Fibonacci number with memoization",
        "validates a Sudoku board", "computes the median of a stream", "serialises a binary tree to a string",
        "detects a cycle in a directed graph", "implements a simple LRU cache", "tokenises an arithmetic expression"]
ZH = ["解释一下什么是通货膨胀。", "写一段关于秋天的散文。", "介绍一下唐朝的诗歌成就。", "如何学习一门新的编程语言？",
      "什么是量子计算？", "简述人工智能的发展历史。", "写一封请假邮件。", "比较一下猫和狗作为宠物的优缺点。",
      "解释光合作用的过程。", "给初学者推荐五本经典小说并说明理由。", "怎样做红烧肉？", "简述丝绸之路的意义。",
      "什么是递归？举一个例子。", "解释一下区块链的共识机制。", "写一首七言绝句，主题是春雨。"]
MISC = ["Give five tips for a job interview.", "Write a limerick about a cat who learns to code.",
        "Explain the rules of chess in brief.", "Plan a three-day trip to Kyoto.",
        "What are the pros and cons of remote work?", "Write a product description for a smart water bottle.",
        "Describe the taste of coffee to someone who has never had it.", "Write a short dialogue between a robot and a child.",
        "List the planets of the solar system with one fact each.", "Explain how a bill becomes law in the United States.",
        "Write a cover letter opening for a data scientist role.", "Compare Python and Rust for systems programming.",
        "What is the difference between weather and climate?", "Draft a polite reminder email about an unpaid invoice.",
        "Explain the Monty Hall problem and its solution.", "Write a SQL query that finds duplicate emails in a users table.",
        "Return a JSON object with five countries, their capitals and populations.",
        "Translate 'Knowledge is power, but enthusiasm pulls the switch' into French, Spanish and German.",
        "Summarise the theory of relativity in simple terms.", "Write a bash one-liner that counts lines in all .py files.",
        "Solve 3x + 7 = 25 and explain each step.", "What is 17 * 23? Show the work.",
        "Describe a sunset over the ocean in vivid prose.", "Explain what an API is to a non-technical manager.",
        "Give a recipe for a vegetarian lasagna.", "Write a regex that matches US phone numbers and explain it.",
        "What were the main causes of World War I?", "Explain Big-O notation with examples.",
        "Write a motivational speech for a marathon team, 150 words.", "How do vaccines work?"]


def prompts():
    p = [f"Explain {c} in two short paragraphs." for c in CONCEPTS]
    p += [f"Write a Python function that {t}, with a docstring and one example." for t in CODE]
    p += ZH + MISC
    sys.path.insert(0, HERE + "/repo/scripts")
    import drift
    p += drift.PROMPTS
    seen, out = set(), []
    for x in p:
        if x not in seen:
            seen.add(x); out.append(x)
    return out


def chat(url, msgs, max_tokens, thinking=False, temperature=0, timeout=1800):
    key = os.environ["SGLANG_API_KEY"]
    body = {"model": MODEL, "messages": msgs, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": thinking}}
    if temperature is not None:
        body["temperature"] = temperature
    r = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                               {"Content-Type": "application/json", "Authorization": "Bearer " + key})
    for attempt in range(3):
        try:
            d = json.loads(urllib.request.urlopen(r, timeout=timeout).read())
            m = d["choices"][0]
            return {"text": m["message"].get("content") or "", "finish": m.get("finish_reason"),
                    "toks": d["usage"]["completion_tokens"]}
        except Exception as e:
            err = str(e); time.sleep(5)
    return {"text": "", "finish": "error: " + err, "toks": 0}


def agree(url, out):
    ps = prompts()
    res = []
    t0 = time.time()
    for p in ps:
        r = chat(url, [{"role": "user", "content": p}], 256)
        r["prompt"] = p
        res.append(r)
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"{out}: {len(res)} prompts in {time.time() - t0:.0f}s, errors {sum(1 for r in res if not r['text'])}")


def cmp(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    same = 0; firsts = []; pref = []
    for x, y in zip(A, B):
        x, y = x["text"], y["text"]
        if x == y:
            same += 1; pref.append(1.0)
            continue
        i = next((k for k, (c1, c2) in enumerate(zip(x, y)) if c1 != c2), min(len(x), len(y)))
        firsts.append(i); pref.append(i / max(len(x), len(y), 1))
    firsts.sort()
    med = firsts[len(firsts) // 2] if firsts else None
    print(f"{os.path.basename(a)} vs {os.path.basename(b)}: identical {same}/{len(A)}; "
          f"mean common-prefix fraction {sum(pref) / len(pref):.3f}; median first diff char {med}; "
          f"first diffs <50 chars: {sum(1 for f in firsts if f < 50)}")


def num(s):
    s = s.replace(",", "").replace("$", "")
    m = re.findall(r"Answer:\s*\**\s*(-?\d+(?:\.\d+)?)", s)
    if not m:
        m = re.findall(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    v = float(m[-1])
    return v


def gsm(url, out, c=8, n=200):
    rows = [json.loads(l) for l in open(HERE + "/data/gsm8k.jsonl")][:n]
    def one(r):
        q = r["question"] + "\n\nSolve step by step, then finish with a final line of the form 'Answer: <number>'."
        a = chat(url, [{"role": "user", "content": q}], 1024)
        gold = float(r["answer"].split("####")[-1].strip().replace(",", ""))
        got = num(a["text"])
        return {"q": r["question"], "gold": gold, "got": got, "ok": got is not None and abs(got - gold) < 1e-6,
                "text": a["text"], "finish": a["finish"], "toks": a["toks"]}
    t0 = time.time()
    with ThreadPoolExecutor(c) as ex:
        res = list(ex.map(one, rows))
    k = sum(r["ok"] for r in res)
    json.dump({"url": url, "n": len(res), "correct": k, "results": res}, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"{out}: GSM8K {k}/{len(res)} = {k / len(res) * 100:.1f}%  truncated {sum(1 for r in res if r['finish'] == 'length')}"
          f"  wall {time.time() - t0:.0f}s")


LONG = ["Write a complete, detailed tutorial on building a REST API in Python with FastAPI, covering setup, routing, validation, databases, auth, testing and deployment. Be thorough, at least 3000 words.",
        "Write a long short story (at least 3000 words) about a crew of a deep-sea research station who discover something unexpected.",
        "Write an exhaustive guide to the history of the Roman Empire from its founding to the fall of Constantinople, at least 3000 words.",
        "List and explain 60 common Linux command-line tools, one paragraph each with an example.",
        "用中文写一篇至少3000字的文章，详细介绍中国古代四大发明及其对世界的影响。",
        "Write a full implementation of a text-based adventure game in Python with rooms, items, inventory, save/load, and a parser, with comments.",
        "Write a detailed comparison of 20 programming languages, covering history, typing, performance, ecosystem and typical use, one section each.",
        "Explain, chapter by chapter, a 15-chapter textbook outline on machine learning, writing about 200 words per chapter."]


def rep_score(t):
    """Fraction of 40-char windows (stride 20) that occur more than twice: a loop detector."""
    if len(t) < 400:
        return 0.0
    w = [t[i:i + 40] for i in range(0, len(t) - 40, 20)]
    from collections import Counter
    c = Counter(w)
    return sum(1 for x in w if c[x] > 2) / len(w)


def longloop(url, out):
    res = []
    for p in LONG:
        a = chat(url, [{"role": "user", "content": p}], 4000)
        t = a["text"].encode()
        a.update(prompt=p, zlib=len(zlib.compress(t)) / max(len(t), 1), rep=rep_score(a["text"]))
        res.append(a)
        print(f"  toks {a['toks']:5d} finish {a['finish']:7s} zlib {a['zlib']:.3f} rep {a['rep']:.3f}  {p[:50]}", flush=True)
    json.dump(res, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"{out}: mean zlib {sum(r['zlib'] for r in res) / len(res):.3f}, max rep {max(r['rep'] for r in res):.3f}, "
          f"loops(rep>0.2) {sum(1 for r in res if r['rep'] > 0.2)}/{len(res)}")


if __name__ == "__main__":
    m = sys.argv[1]
    if m == "agree":
        agree(sys.argv[2], sys.argv[3])
    elif m == "cmp":
        cmp(sys.argv[2], sys.argv[3])
    elif m == "gsm":
        gsm(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 8, int(sys.argv[5]) if len(sys.argv) > 5 else 200)
    elif m == "longloop":
        longloop(sys.argv[2], sys.argv[3])
    elif m == "count":
        print(len(prompts()))
