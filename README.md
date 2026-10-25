# Nanbeige4.3-33B · own NVFP4/FP8 quant · SGLang + n-gram speculation · NVIDIA RTX PRO 6000

[`Nanbeige/Nanbeige4.3-33B`](https://huggingface.co/Nanbeige/Nanbeige4.3-33B), released on
2026-10-24 under Apache-2.0, served on one RTX PRO 6000 on Cloud Run. We quantize it ourselves with
NVIDIA ModelOpt and serve it with SGLang v0.5.22. The process, scripts and quality bar come from
[Qwen3.8-27B-abliterated-AWQ-SGLang-DFlash2-RTX-PRO-6000](https://github.com/hanxiao/Qwen3.8-27B-abliterated-AWQ-SGLang-DFlash2-RTX-PRO-6000).

| | |
| --- | --- |
| Model | [`Nanbeige/Nanbeige4.3-33B`](https://huggingface.co/Nanbeige/Nanbeige4.3-33B): dense Llama architecture, 33.1B parameters, 52 layers, hidden 7,168, 56 query and 8 KV heads of 128, vocabulary 166,912, text only. Served as `Nanbeige4.3-33B` |
| Weights | FP8 W8A8 attention, NVFP4 MLP weights and `lm_head`, both served as W4A16 (NVFP4 weights, BF16 activations, Marlin). Our own quantization with ModelOpt 0.47 (`quant/modelopt/`), 22.2 GB resident |
| Accuracy | KL 0.080 to BF16 and 90.1% top-1 agreement on our private corpus (FP8 W8A8 everywhere: 0.062; NVFP4 W4A4 MLPs: 0.122) |
| Drafter | none exists for this model; SGLang's n-gram speculation, 8 draft tokens |
| Engine | SGLang v0.5.22 (`lmsysorg/sglang:v0.5.22`) with four patches in `docker/`, FlashInfer attention |
| GPU | one whole NVIDIA RTX PRO 6000 Blackwell (96 GB, SM120) on Cloud Run, 20 vCPU, 80 GiB |
| Slots | 8 concurrent requests |
| Context | **262,144** tokens per request; one 593,794-token FP8 KV pool shared by the 8 slots (it also fits the full 524,288-token context of the model) |
| Prefill | **4,779 tok/s** at 9.5k tokens, 3,884 at 32k, 2,112 at 127k (the W4A4 build of the same weights: 9,855 / 7,103 / 2,889, see [Results](#results)) |
| Time to first token, coding agent | 6.8 s at 27k tokens of code context, 18.4 s at 58k, 54.5 s at 119k; a follow-up turn **0.38 s** at 27k, 0.34 s at 58k, 0.52 s at 119k |
| Decode, coding agent, thinking on, sampled | **70.7 tok/s** at 27k tokens of context, 85.3 at 58k, 63.7 at 119k with one stream; per stream at two streams 67.2 / 49.9 / 29.3, at three 53.7 / 36.9 / 21.0 |
| Decode, greedy, thinking off | **78.9 tok/s**, first pass (prose 59.9, chat 62.7, translation 67.7, summary 64.6, code 82.2, math 93.0, JSON 122.0); BF16 decodes 22.1 |
| Quality | KL 0.080 to BF16; needle retrieval 5/5 at 30k (twice) and 121k tokens; multi-step math 12/12 (BF16 11/12); GSM8K 189/200 (BF16 191); `qcon.py` 61/64 alone and 63/64 at three streams (BF16 63/64 and 63/64); HumanEval 80/82 (two answers ran past 24,000 tokens; the W4A4 build also 80/82); ten of ten facts across a conversation with 99.8% of the prompt from cache |
| Tool calls | `qwen3_coder` format; arguments stream as they are generated; earlier thinking reaches the template natively, so tool rounds and follow-up turns hit the prefix cache; parallel tool calls 16/16 on `/v1/messages`, streaming and non-streaming |
| Structured output | `response_format` (JSON schema) and `tool_choice: "required"` work (xgrammar) |
| Cold start | about 12 minutes from instance start to ready; one instance stays warm |

All figures were measured on 2026-10-25. The production rows come from the production revision
`nanbeige43-33b-00001-sdn` itself, before any client used it. The comparison rows come from test
services with the same image and flags except where a row says otherwise.

```bash
bash docker/build.sh                                            # lmsysorg/sglang:v0.5.22 + four patches, into Artifact Registry
gcloud run services replace quant/modelopt/service.yaml         # once: quantize into the bucket (ModelOpt), then delete it
gcloud run services replace cloudrun/service.yaml
gcloud run services add-iam-policy-binding nanbeige43-33b --member=allUsers --role=roles/run.invoker
```

Fill in `REGION`, `PROJECT`, `BUCKET` and `SERVICE_ACCOUNT` first. The service reads its bearer token
from the Secret Manager secret `NANBEIGE43_API_KEY`.

## Results

All rows are on one whole RTX PRO 6000. Decode in bench2 is greedy with thinking off; the coding
agent streams `/v1/messages` with thinking on and the default sampling of the model. KL is
`scripts/fid.py` against our own BF16 run on the same corpus.

| Build (weights / KV / speculation) | KV pool, tokens | bench2, tok/s | Coding agent, one stream, 27k / 58k / 119k tok/s | TTFT at 27k / 58k / 119k | Prefill at 9.6k / 32k / 127k, tok/s | KL to BF16, top-1 | Needle 30k ×2, ~120k | Math | GSM8K of 200 | Parallel tool calls |
| --- | ---: | ---: | --- | --- | --- | --- | --- | ---: | ---: | --- |
| BF16 / BF16 / none | 114,999 | 22.1 | 20.1 / 18.7 / does not fit | 6.6 / 17.5 s / – | 4,964 / 4,113 / does not fit | 0 (reference), PPL 8.255 | 5/5, 5/5, 5/5 at 112k | 11/12 | 191 | 16/16 + 16/16 |
| FP8 W8A8 / BF16 / none | 239,297 | 41.1 | | | 7,482 / 5,792 / 2,647 | 0.0624, 92.2% | | | | |
| NVFP4 W4A4 MLPs / BF16 / none (`molh`) | 297,567 | 63.6 | 50.1 / 41.1 / 30.4 | 3.7 / 11.7 / 39.3 s | 9,855 / 7,103 / 2,889 | 0.1221, 87.6% | 5/5, 5/5, 5/5 | 11/12 | 189 | 16/16 + 16/16 |
| NVFP4 W4A16 MLPs / BF16 / none (`w4m`) | 296,907 | 62.2 | | | 4,762 / 4,049 / 2,208 | 0.0796, 90.1% | 5/5, 5/5, 5/5 | 12/12 | | |
| W4A16 / FP8 / none | 593,794 | 59.8 | 52.5 / 47.0 / 39.1 | 6.8 / 18.5 / 54.6 s | 4,691 / 3,860 / 2,102 | same as above¹ | 5/5, 5/5, 5/5 | 12/12 | | |
| **W4A16 / FP8 / n-gram 8 (deployed)** | **593,794** | **78.1**² | **71.3 / 69.4 / 61.8** | 6.8 / 18.6 / 55.0 s | 4,662 / 3,898 / 2,092 | 0.0796, 90.1% | 5/5, 5/5, 5/5 | 12/12 | 189 | 16/16 + 16/16 |

¹ `fid.py` scores one prefill per document, and a prefill attends to the new tokens in BF16, so an
FP8 KV cache does not show in it. The decode-side checks (needle at 121k after chunked prefill
from the FP8 cache, math, `qcon.py`, GSM8K) cover it.

² bench2 sends the same prompts in each of its runs, and the n-gram tree remembers the earlier
answers: the three runs after the first gave 173, 243 and 263 tok/s. 78.1 is the first pass over
the prompts, the fair figure: prose 60.6, chat 63.1, translation 68.0, summary 65.6, code 77.9,
math 92.6, JSON 118.9.

How to read it:

- **BF16** does not fit a 120k-token request: the 66 GB of weights leave a 114,999-token pool.
  Two 58k-token agent runs at once already queue (time to first token 137 s at two streams).
- **Weights:** decode is bound by the bytes read per token. FP8 reads 36 GB per step and NVFP4 MLPs
  22 GB, so they decode 1.9 and 2.9 times as fast as BF16.
- **W4A4 against W4A16 MLPs:** the same NVFP4 weights, served once with FP4 activations (CuTe-DSL
  FP4 GEMM) and once with BF16 activations (Marlin), a config-only change
  (`cloudrun/prep.py w4mlp`). W4A16 lowers KL from 0.122 to 0.080 at the same decode, and halves
  prefill (time to first token 6.8 against 3.7 s at 27k tokens). A coding agent spends most of
  its time decoding, and each follow-up turn hits the prefix cache (0.3 s), so we chose the lower
  KL. Dropping `w4mlp` from the prep step, or serving `nanbeige43/q/molh` with `prep.py w4h`,
  gives back the fast prefill.
- **FP8 KV:** all 52 layers use full attention, so at long context a decode step reads more KV
  than weights (12 GB at 58k tokens in BF16). FP8 KV halves that: 47.0 against 41.1 tok/s at 58k
  and 39.1 against 30.4 at 119k, and doubles the pool. It costs 4% of decode at short context and
  about 5% of prefill.
- **n-gram speculation:** a coding agent copies from its context and from itself; with 8 draft
  tokens one stream commits 1.55 to 1.80 tokens per verify step: +36% at 27k, +48% at 58k and
  +58% at 119k tokens. Prose is not slower.

## Quantization

`quant/modelopt/` has the scripts, adapted from the Qwen repository.

- **Tool:** NVIDIA ModelOpt 0.47.0 (`mo.py`), on a Cloud Run service with one whole RTX PRO 6000,
  30 vCPU and 120 GiB (`job.sh`, `service.yaml`). The 66 GB BF16 checkpoint and the export sit in
  the in-memory `/tmp`, which is why the service needs 120 GiB.
- **Layout (`molh`):**
  - FP8 per-tensor W8A8 with static scales: `self_attn.q/k/v/o_proj`.
  - NVFP4, group 16: `mlp.gate/up/down_proj` and `lm_head`.
  - BF16: embeddings and norms.
  - FP8 KV scales, calibrated.
- **Weight scales:** `local_hessian` with the FP8 scale sweep for the NVFP4 block scales.
- **Calibration:** 583 samples and 1.41M tokens, at most 4,096 tokens each, rendered with the
  Nanbeige chat template (`build_calib.py`): 44% agentic and tool-call data (OpenHands/SWE-Gym and
  Nemotron-SWE trajectories, 70 tool-call conversations rendered through the template's `tools=`
  path), 28% code (SGLang source as review/refactor/test requests, windows of the coding-agent
  context), 21% Chinese chat, 6% English chat, 1% math and STEM. The KL corpus is held out.
- **Serving:** the NVFP4 MLPs and the head are served as `W4A16_NVFP4`, and the FP8 KV scales are
  kept for the FP8 KV cache (`cloudrun/prep.py gen,w4h,w4mlp`; the bucket prefix
  `nanbeige43/q/molh-w4m` has these edits baked in).
- **FP8 alternative (`fp8`):** ModelOpt `max`, FP8 per-tensor W8A8 in every linear layer, BF16
  head. KL 0.062, but 41 tok/s.

Served KL, `scripts/fid.py` over 46 private documents and 201,887 positions:

| Build | PPL | KL Chinese | KL code | KL docs | KL papers | KL all | Top-1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 (reference) | 8.255 | | | | | | |
| FP8 W8A8, BF16 head | 8.063 | 0.0389 | 0.0487 | 0.0996 | 0.0429 | 0.0624 | 92.22% |
| NVFP4 W4A4 MLPs, head W4A16 | 8.292 | 0.0671 | 0.0986 | 0.1853 | 0.1064 | 0.1221 | 87.57% |
| **NVFP4 W4A16 MLPs and head (deployed)** | 8.263 | 0.0340 | 0.0616 | 0.1296 | 0.0687 | **0.0796** | 90.11% |

What did not work:

- **The offline proxy KL in `mo.py`** (fake-quantized ModelOpt model against the BF16 top-64)
  reported 0.72 for `molh` and 0.41 for `fp8`, with 2.5 and 1.5 on code, while the served
  checkpoints measure 0.12 and 0.06. Putting all MLPs back in BF16 still left 0.39, and attention
  made no difference, so the error lies in the fake-quant forward of this model under ModelOpt
  0.47 with transformers 5.20, which ModelOpt does not list as tested. We judged every build by the
  served KL instead. The sensitivity sweep (`SENS=1`) therefore says nothing here.
- **SGLang v0.5.22 could not load a MIXED_PRECISION checkpoint of a Llama model:** its Llama class
  has no `packed_modules_mapping`, so the fused `qkv_proj` and `gate_up_proj` found no entry in
  `quantized_layers`, were built unquantized, and loading the packed NVFP4 weights failed with
  `assert param_data.shape == loaded_weight.shape`. `docker/llama_packed_modules.py` declares the
  mapping.

## Speculative decoding

No drafter exists for Nanbeige4.3-33B on Hugging Face as of 2026-10-25: no EAGLE, DFlash or MTP
head. `Nanbeige/Nanbeige4.2-3B` shares the tokenizer (identical `tokenizer.json`, vocabulary 166,144
against 166,912), but its `NanbeigeForCausalLM` architecture is not in SGLang v0.5.22, and the
Llama-based Nanbeige4.1-3B has a different tokenizer. We did not try a separate draft model.

SGLang's n-gram speculation needs no model: it drafts from a trie of n-grams of the prompts and
outputs it has seen (`--speculative-algorithm NGRAM`). With 8 draft tokens and the defaults of
v0.5.22 (breadth 10, trie depth 18) it gave the gains above. We did not tune the draft length.
It turns off mixed chunked prefill, and structured output still works with it.

## Tool calls and thinking

The chat template uses the `<tool_call><function=...><parameter=...>` format of Qwen3 Coder and
the `reasoning_content` field, so the patches of the Qwen repository apply unchanged:

- `docker/qwen3_coder_detector.py` streams string arguments as they are generated (sglang PR
  #25246, ported), drops whitespace text that rides with call items (sglang#42260, which broke
  parallel calls on `/v1/messages`), and closes calls the model leaves open (sglang#43150).
  `docker/test_stream.py` runs at build time against the stock parser, which must fail, and the
  patched one.
- `docker/anthropic_reasoning_history.py` passes earlier thinking on `/v1/messages` to the template
  as `reasoning_content`. The template keeps the thinking of every assistant turn by default
  (`preserve_thinking` undefined counts as true), so the re-rendered history equals what the model
  generated and follow-up turns hit the cache: 0.27 to 0.36 s to the first token at 27k to 119k
  tokens, with all but 61 or 62 prompt tokens from the cache.
- `docker/sched_admission.py` counts a prefilling request's row once at admission (PR #44401).

The template ends the generation prompt with `<think>\n`, and `--reasoning-parser qwen3` splits the
thinking off. `enable_thinking: false` in `chat_template_kwargs` turns it off. The template has no
effort levels, so "thinking on" is its maximum. The model card recommends temperature 1.0 and top_p
0.95 for agentic use (0.6 / 0.95 for chat); the served `generation_config.json` carries the agentic
values, which apply to requests that send none. Without a system message the template inserts its
own (a Chinese identity prompt, or a tool-calling prompt when tools are given).

## Production, 2026-10-25

The service `nanbeige43-33b` in us-central1 is new; nothing else moved. Revision
`nanbeige43-33b-00001-sdn` serves `cloudrun/service.yaml`. We ran every check against it right
after it came up:

| Check | Result |
| --- | --- |
| Smoke test (`/v1/chat/completions` text; `/v1/messages` streamed tool call) | pass; valid JSON arguments, `stop_reason: tool_use` |
| KL to BF16, top-1 | 0.0796, 90.11% |
| Needle 30k / 30k / 121k, math | 5/5, 5/5, 5/5, 12/12 |
| `qcon.py`, alone / three streams | 61/64 / 63/64 (no word from another request in any answer) |
| GSM8K, 200, thinking off | 189 |
| HumanEval, 41 × 2, thinking on | 80/82 |
| Parallel tool calls, `/v1/messages`, 8 cases × 2 | 16/16 non-streaming, 16/16 streaming |
| `toolstream.py`, 1,500 integers: longest silence | 0.2 s on `/v1/messages`, 0.1 s on chat completions; content exact |
| `prefix.py`, facts / prompt from cache | 10/10, 99.8% |
| `response_format` JSON schema; `tool_choice: "required"` | valid JSON; one `get_weather` call |
| bench2, first pass | 78.9 tok/s |
| Prefill at 9.5k / 32k / 127k | 4,779 / 3,884 / 2,112 tok/s |

Coding agent (`scripts/codebench.py`, medians; C1 three runs, C2 four, C3 six):

| Context | One stream | Two streams, per stream | Three streams, per stream | TTFT, one stream | Follow-up TTFT | Tokens per verify |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 27k | 70.7 | 67.2 | 53.7 | 6.8 s | 0.38 s | 1.57 |
| 58k | 85.3 | 49.9 | 36.9 | 18.4 s | 0.34 s | 1.98 |
| 119k | 63.7 | 29.3 | 21.0 | 54.5 s | 0.52 s | 1.93 |

The n-gram tree keeps what the server has generated, and the benchmark repeats its three tasks,
so later phases draft from earlier answers to the same task: the 58k figure (85.3) ran after
thirteen 27k answers. On a fresh test service, where 58k came after three, one stream gave 69.4.
An agent that rewrites code it has written before sees the same effect. At three streams of
119k tokens the prompts (357k tokens plus output) and six queued requests share one pool, so time
to first token includes waiting.

GPU budget for the study: 9.2 GPU-hours of test services and quantization on 2026-10-25 (BF16
2.6, quantization 1.4, the W4A4 build 2.3, FP8 0.7, W4A16 0.6, FP8 KV 0.9, n-gram 0.7). All test
services are deleted.

### Client configuration

An Anthropic-style client points at the service URL with the bearer token from
`NANBEIGE43_API_KEY`, model `Nanbeige4.3-33B`, context 262,138 input tokens (SGLang's
`max_req_input_len`), reasoning on. SGLang returns 400 when the prompt plus `max_tokens` exceeds
the context, so clients cap the completion below the window (the model card suggests 65,536 for
agentic work).

## Flags

`--attention-backend flashinfer` is required on SM120. `--kv-cache-dtype fp8_e4m3` uses the KV
scales from the ModelOpt export; with `bf16` the scales must be stripped first (`prep.py stripkv`),
or the output is garbage. `--tool-call-parser qwen3_coder` and `--reasoning-parser qwen3` are what
the model card names. `--sampling-defaults model` serves the sampling in `generation_config.json`.
`--enable-cache-report` reports cache hits (`cache_read_input_tokens` on `/v1/messages`,
`prompt_tokens_details` on chat completions). `--enable-http2` with the `h2c` port lets a client
disconnect abort its request. `--mem-fraction-static 0.86` leaves 10.5 GB free after startup.

## Cold start

From instance start to ready took 12 min 22 s on the production revision (20:34:56 to 20:47:18
UTC): 7 min 12 s to download the 23.0 GB checkpoint (`cloudrun/gcsdl.py`, ~53 MB/s), 15 s to load
it into the GPU, and about 3.5 minutes for the pools, CUDA graphs and warmup. BF16 needs 66.3 GB: 21 minutes of download alone.

## Cost

One RTX PRO 6000 with 20 vCPU and 80 GiB, always on (minScale 1): about $3.18 per hour at the
Cloud Run list prices we used ($1.31 per hour for the GPU, $0.0648 per vCPU-hour, $0.0072 per
GiB-hour), so about **$76 per day** or $2,300 per 30 days, before committed-use discounts. Check
the current price page; we did not read the bill.

## Reproduction

1. Copy the model to the bucket (any machine with network; we used a Cloud Build step):
   `huggingface_hub.snapshot_download('Nanbeige/Nanbeige4.3-33B')`, then
   `gcloud storage cp -r` into `gs://BUCKET/nanbeige43/bf16/`, and `cloudrun/gcsdl.py` to the bucket root.
2. `bash docker/build.sh`.
3. `python3 quant/modelopt/build_calib.py`, then upload `mo.py`, `job.sh`, `gcsul.py`, the parquet
   and the KL corpus to `gs://BUCKET/nanbeige43/q/job/`; `gcloud run services replace
   quant/modelopt/service.yaml`; delete the service when `nanbeige43/q/molh` is uploaded.
4. Bake the serving edits: `prep.py <dir> gen,w4h,w4mlp` on the three JSON files of the export, and
   upload them with the safetensors to `nanbeige43/q/molh-w4m/`.
5. `gcloud run services replace cloudrun/service.yaml` and grant `roles/run.invoker`.
6. Benchmarks: `scripts/` (`SGLANG_API_KEY=... python3 scripts/bench2.py URL`, and so on;
   `scripts/mkctx.py` cuts the coding-agent contexts with the Nanbeige tokenizer).

## Credit

Nanbeige team for the model; SGLang; NVIDIA ModelOpt; the Qwen3.8 repository for the process.
