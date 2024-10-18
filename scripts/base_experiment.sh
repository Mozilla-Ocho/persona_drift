#!/bin/bash

MODEL_NAME="$1"

NTRIALS=100

if [ -z "${VIRTUAL_ENV}" ]; then
    echo "VIRTUAL_ENV is not set: run 'source venv/bin/activate' first" >&2
    exit 1
fi

if [ -z "${MODEL_NAME}" ]; then
    echo "MODEL_NAME is not set: pass the model name as the first argument" >&2
    exit 1
fi

source .env

for ((trial=0; trial<${NTRIALS}; trial++))
do
  echo "Trial: ${trial}"
  python run_updated.py \
  --model_name "${MODEL_NAME}" \
  --agent -1 \
  --user -1 \
  --turns 32 \
  --seed "${trial}" \
  --runs 1
done

echo "Done with ${NTRIALS} trials"
