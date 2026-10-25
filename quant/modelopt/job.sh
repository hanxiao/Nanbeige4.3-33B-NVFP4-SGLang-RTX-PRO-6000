#!/bin/bash
# Cloud Run GPU entry for the own ModelOpt quant of Nanbeige4.3-33B (mo.py). Bucket mounted at /gcs.
# VARIANTS="NAME=a ALGO=max ...;NAME=b ..." run one after another on one download; afterwards the
# container polls /gcs/nanbeige43/q/queue/$K_SERVICE/*.v (one variant line per file) so new variants
# reuse the local BF16 copy (66 GB, about 20 minutes from the bucket at ~55 MB/s).
set -uxo pipefail
pip install -q uv
uv venv -q /venv --python 3.12
uv pip install -q --python /venv/bin/python torch "transformers>=5" "nvidia-modelopt==0.47.0" accelerate pandas pyarrow safetensors requests sentencepiece protobuf
/venv/bin/python -c "import torch,transformers,modelopt,pandas,safetensors;import modelopt.torch.quantization as mtq;import modelopt.torch.export as mte;print('IMPORTS OK',torch.__version__, transformers.__version__, modelopt.__version__, torch.cuda.get_device_name(0))" || { echo "IMPORT FAIL"; exit 3; }
pip install -q 'google-cloud-storage==3.17.1' && python3 /gcs/gcsdl.py ${BUCKET:?set BUCKET} nanbeige43/bf16=/tmp/src || exit 4
export CALIB=/gcs/nanbeige43/q/job/${CALIBF:-nbmix.parquet} FIDC=/gcs/nanbeige43/q/job/fid_corpus.json
runv() {
  echo "=== VARIANT $1"
  env $1 /venv/bin/python -u /gcs/nanbeige43/q/job/mo.py /tmp/src /tmp/out
  echo "=== VARIANT RC $? $1"
  rm -rf /tmp/out /tmp/out-bh
}
IFS=';' read -ra VS <<< "${VARIANTS:-}"
for v in "${VS[@]}"; do runv "$v"; done
QD=/gcs/nanbeige43/q/queue/${K_SERVICE:-none}
mkdir -p $QD
echo "QUEUE $QD"
while true; do
  for f in $(ls $QD/*.v 2>/dev/null | sort); do
    v=$(cat $f); rm -f $f
    runv "$v"
  done
  sleep 20
done
