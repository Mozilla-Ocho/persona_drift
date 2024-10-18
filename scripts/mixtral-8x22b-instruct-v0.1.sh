#!/bin/bash

MODEL_NAME="mistralai/Mixtral-8x22B-Instruct-v0.1"

NTRIALS=200

if [ -z "${VIRTUAL_ENV}" ]; then
    echo "VIRTUAL_ENV is not set: run 'source venv/bin/activate' first" >&2
    exit 1
fi

source .env

for ((trial=0; trial<${NTRIALS}; trial++))
do
  echo "Trial: ${trial}"
  python run_updated.py \
  --api_base_url "http://216.153.62.238:8000/v1" \
  --model_name "${MODEL_NAME}" \
  --agent -1 \
  --user -1 \
  --turns 32 \
  --seed "${trial}" \
  --runs 1
done

echo "Done with ${NTRIALS} trials"