#!/bin/bash
if [[ $# -ne 0 ]]; then
    echo "$0 does not accept parameters"
    exit 2
fi
docker run \
-e GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/legacy_credentials/${USER}@private.com/adc.json \
-e USER=${USER} \
-v ~/.config/gcloud:/root/.config/gcloud \
--rm -i -t serenade-rust-optimized \
gs://my-google-cloud-project/train_full/csv/part-00000-2ec72272-c01e-4e80-bf3a-45b18fdbfff4-c000.csv \
gs://my-google-cloud-project/test_full/csv/part-00000-f3c1634a-3c41-4c78-ba2c-435fb52254dd-c000.csv \
gs://my-google-cloud-project/results/rust_1m_vsknn_predictions.txt \
gs://my-google-cloud-project/results/rust_1m_latencies.txt

