# Model Drift

This repository provides the code for plotting persona drift in LLM-based chatbots, as discussed in [Measuring and Controlling Persona Drift in Language Model Dialogs](https://arxiv.org/html/2402.10962v1). 

## Abstract

> Prompting is a standard tool for customizing language-model chatbots, enabling them to take on a specific "persona". An implicit assumption in the use of prompts is that they will be stable, so the chatbot will continue to generate text according to the stipulated persona for the duration of a conversation. We propose a quantitative benchmark to test this assumption, evaluating persona stability via self-chats between two personalized chatbots. Testing popular models like LLaMA2-chat-70B, we reveal a significant persona drift within eight rounds of conversations. An empirical and theoretical analysis of this phenomenon suggests the transformer attention mechanism plays a role, due to attention decay over long exchanges. To combat attention decay and persona drift, we propose a lightweight method called split-softmax, which compares favorably against two strong baselines.

## Installation

To install with conda:
```
conda env create -f environment.yml
conda activate drift
python -m ipykernel install --user --name drift --display-name "drift"
cp .env.example .env
# Edit .env file to fill in your OpenAI API key
```

To install with pip:
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name=drift --display-name="drift"
cp .env.example .env
# Edit .env file to fill in your OpenAI API key
```

<!-- Then we download the [dataset](https://huggingface.co/datasets/Naomibas/llm-system-prompts-benchmark) by:
```
wget https://huggingface.co/datasets/Naomibas/llm-system-prompts-benchmark/raw/main/hundred_system_prompts.py
``` -->

## Generating Self-Chats

For example, `python run.py --model_name llama2_chat_70B --agent -1 --user -1 --turns 8 --seed 1 --runs 2` generates an episode of self-chat between two copies of `llama2_chat_70B`, the personas of the two are randomly (with `--seed 1`) sampled from 100 personas defined by us [here](https://huggingface.co/datasets/Naomibas/llm-system-prompts-benchmark). The conversation will go for `8 (--turns)` turns (or `4` rounds). At each turn for the agent (2, 4, ..., 8), the probe question is asked `2 (--runs)` times. Results will be saved into `selfchat` folder.

Note that the model can be from HuggingFace or API calls like `--model_name gpt-3.5-turbo-16k`. The code is easily hackable so that you can swap in your locally built model. 

You can also skip this process by downloading self-chat histories from [this google drive](https://drive.google.com/drive/folders/1Iho3KfDbpxrMzEBum_VriKaUuaMji7zu?usp=sharing) and put them into `selfchat` folder.

## Run full experiment as described in the paper

Use the scripts in the `scripts/` folder to run a full experiment (i.e. all 100 trial conversations). For example, to run the full experiment with `gpt-3.5-turbo-16k`:

```bash
source venv/bin/activate
./scripts/gpt-3.5-turbo-16k.sh
```

Or you can run with a different model using

```bash
source venv/bin/activate
./scripts/base_experiment.sh <MODEL NAME>
```

Note that currently only `llama2_7B`, `llama2_chat_7b`, and `llama2_chat_70b` are supported. @k8si will add support for more models soon.

## Plotting Persona Drift

Please check out `plot_convergence.ipynb`.

## How to Cite

```
@article{li2024measuring,
  title={Measuring and Controlling Persona Drift in Language Model Dialogs},
  author={Li, Kenneth and Liu, Tianle and Bashkansky, Naomi and Bau, David and Vi{\'e}gas, Fernanda and Pfister, Hanspeter and Wattenberg, Martin},
  journal={arXiv preprint arXiv:2402.10962},
  year={2024}
}
```
