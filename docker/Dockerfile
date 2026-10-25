# Stock SGLang v0.5.22 with four changes, each a script that fails the build if v0.5.22's text
# moved, and Granian for --enable-http2. Three come unchanged from
# github.com/hanxiao/Qwen3.8-27B-abliterated-AWQ-SGLang-DFlash2-RTX-PRO-6000: Nanbeige4.3-33B uses
# the same <tool_call><function=...><parameter=...> format and the same reasoning_content field in
# its chat template, so the Qwen3 Coder parser and the reasoning-history patch apply as they are;
# llama_packed_modules.py is new here.
#   qwen3_coder_detector.py  streaming tool-call arguments (sglang PR #25246, ported); drops
#                            whitespace text riding with call items (sglang#42260); closes calls the
#                            model leaves open (sglang#43150, PR #43467 plus the held-terminator case)
#   sched_admission.py       admission counts a prefilling request's row once (PR #44401)
#   llama_packed_modules.py  LlamaForCausalLM declares packed_modules_mapping, so a ModelOpt
#                            MIXED_PRECISION checkpoint (NVFP4 MLPs, FP8 attention) loads
#   anthropic_reasoning_history.py  earlier thinking on /v1/messages reaches the chat template as
#                            reasoning_content, so follow-up turns hit the prefix cache
#
# The parser step stops unless the stock parser fails test_stream.py (else the test proves
# nothing), the patched parser passes it, and the patched parser passes SGLang's own Qwen3 Coder
# tests. The last step prints the patched files' hashes, to match a build against a tested image.
FROM lmsysorg/sglang:v0.5.22
COPY qwen3_coder_detector.py test_stream.py sched_admission.py anthropic_reasoning_history.py \
     llama_packed_modules.py /tmp/patch/
RUN set -e; \
    T="$(python3 -c 'import sglang.srt.function_call.qwen3_coder_detector as m; print(m.__file__)')"; \
    S="$(python3 -c 'import pathlib, sglang; print(pathlib.Path(sglang.__file__).parents[2])')"; \
    cp "$T" /tmp/patch/qwen3_coder_detector.orig.py; \
    if python3 /tmp/patch/test_stream.py /tmp/patch/qwen3_coder_detector.orig.py; then \
        echo "the stock parser passed test_stream.py: it proves nothing on this version"; exit 1; fi; \
    cp /tmp/patch/qwen3_coder_detector.py "$T"; \
    python3 /tmp/patch/test_stream.py; \
    cd "$S/test/registered/unit/function_call"; \
    python3 -m unittest -v test_function_call_parser.TestQwen3CoderDetector; \
    python3 /tmp/patch/sched_admission.py; \
    python3 /tmp/patch/anthropic_reasoning_history.py; \
    python3 /tmp/patch/llama_packed_modules.py; \
    rm -rf /tmp/patch; \
    P="$(python3 -c 'import pathlib, sglang; print(pathlib.Path(sglang.__file__).parent)')"; \
    cd "$P" && sha256sum srt/function_call/qwen3_coder_detector.py srt/managers/scheduler.py \
        srt/entrypoints/anthropic/serving.py srt/models/llama.py \
        && pip install --no-cache-dir granian==2.8.4 && python3 -c 'import granian; print("granian", granian.__version__)'
