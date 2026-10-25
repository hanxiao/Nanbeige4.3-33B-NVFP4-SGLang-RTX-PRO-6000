#!/usr/bin/env bash
# Build the serving image into Artifact Registry, where Cloud Run pulls it in-region: stock
# lmsysorg/sglang:v0.5.22 with the four changes listed in the Dockerfile.
set -euo pipefail

PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-us-central1}"
REPO="${REPO:-sglang}"
TAG="${TAG:-nanbeige43-sglang:v0.5.22-patched4}"

here="$(cd "$(dirname "$0")" && pwd)"
stage="$(mktemp -d)"; trap 'rm -rf "$stage"' EXIT
cp "$here/Dockerfile" "$here/qwen3_coder_detector.py" "$here/test_stream.py" \
   "$here/sched_admission.py" "$here/anthropic_reasoning_history.py" \
   "$here/llama_packed_modules.py" "$stage/"

gcloud artifacts repositories create "$REPO" --repository-format=docker \
  --location="$REGION" --project="$PROJECT" 2>/dev/null || true

gcloud builds submit "$stage" --project="$PROJECT" --region="$REGION" \
  --tag="$REGION-docker.pkg.dev/$PROJECT/$REPO/$TAG" --timeout=3600s \
  --machine-type=e2-highcpu-8 --disk-size=200

echo "built $REGION-docker.pkg.dev/$PROJECT/$REPO/$TAG"
echo "set it as the image in cloudrun/service.yaml"
