from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")

import os

from tqdm import tqdm
import numpy as np
import pickle
import random
import time
from functools import partial
from pprint import pprint
import argparse
import copy
from openai import OpenAI
from collections import defaultdict
from pathlib import Path
import json
from typing import Optional
import time

import torch
import transformers

from utils import *
from hundred_system_prompts import *

index_list = [0, 0, 0, 0, 0]
personas = [_[__] for _, __ in zip([pattern_system_prompts, multiple_choice_system_prompts, persona_system_prompts, memorization_system_prompts, language_system_prompts], index_list)]
other_personas = [_[__:] for _, __ in zip([pattern_system_prompts, multiple_choice_system_prompts, persona_system_prompts, memorization_system_prompts, language_system_prompts], [1, 1, 1, 1, 1])]
for _ in other_personas:
    personas.extend(_)


def load_or_init_conversation(args, topic: str, persona: str, user: str) -> dict:
    model_name = args.model_name
    if "/" in model_name:
        # Handle case where model name is e.g. 'mistralai/Mistral-7B-Instruct-v0.3'
        model_name = model_name.replace("/", "-")
    file_name = f"{model_name}_agent_{args.agent}_user_{args.user}_turn_{args.turns}"
    file_name += ".pkl"
    path = Path(f"selfchat/{file_name}")
    if not path.exists():
        # If no conversation existed, initialize a new one
        return {
            "topic": topic,
            "history": [topic],
            "probed_history_per_turn": defaultdict(list),
            "seed": args.seed,
            "persona": persona,
            "user": user,
        }

    with open(path, "rb") as handle:
        old_pkl = pickle.load(handle)
    pkl = {
        "topic": topic,
        "history": old_pkl["history"],
        "probed_history_per_turn": old_pkl["probed_history_per_turn"],
        "seed": args.seed,
        "persona": persona,
        "user": user,
    }
    return pkl


def save_conversation(args, pkl):
    model_name = args.model_name
    if "/" in model_name:
        # Handle case where model name is e.g. 'mistralai/Mistral-7B-Instruct-v0.3'
        model_name = model_name.replace("/", "-")
    file_name = f"{model_name}_agent_{args.agent}_user_{args.user}_turn_{args.turns}"
    file_name += ".pkl"
    path = Path(f"selfchat/{file_name}")
    # TODO why are we pickling btw, why not just save as json...
    with open(path, "wb") as handle:
        pickle.dump(pkl, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pprint(f"Saved to  {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama2_chat_7B')
    parser.add_argument('--agent', type=int, default=-1, choices=[-1, ] + list(range(len(personas))))
    parser.add_argument('--user', type=int, default=-1, choices=[-1, ] + list(range(len(personas))))
    parser.add_argument('--topic', type=int, default=-1, choices=range(len(topics)))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--turns', type=int, default=16)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument(
        '--disable_load_in_8bit',
        action='store_true',
        help='Disable loading model in 8-bit, requires NVidia GPU and bitsandbytes (default: 8-bit enabled i.e. False)'
    )
    parser.add_argument(
        '--api_base_url', type=str, default=None,
        help='base url to use for API calls'
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        # loading with 8-bit quantization requires bitsandbytes which is
        # currently only available on nvidia gpus
        args.disable_load_in_8bit = True

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if args.agent == -1:
        args.agent = random.randint(0, len(personas)-1)
    if args.user == -1:
        args.user = random.randint(0, len(personas)-1)
    persona, probe_str, judge_func = personas[args.agent]
    user, probe_str_user, judge_func_user = personas[args.user]
    if args.topic == -1:
        args.topic = random.randint(0, len(topics)-1)
    topic = topics[args.topic]
    print(f"Now {args.model_name} chatting over {topic} with system prompts: (A) {persona} and (B) {user}")

    # load assistant
    client = None
    use_api = "gpt" in args.model_name
    if use_api:
        client = OpenAI()
    elif args.api_base_url is not None:
        client = OpenAI(base_url=args.api_base_url)
        print(f"USING BASE URL = {args.api_base_url}")
    else:
        # model = ENGINE_MAP[args.model_name]

        disable_8bit = args.disable_load_in_8bit
        if disable_8bit:
            load_in_8bit = False
        else:
            load_in_8bit = True

        tokenizer, intervened_model = load_model(args.model_name, load_in_8bit=load_in_8bit)
        pipeline = transformers.pipeline(
            "text-generation",
            model=intervened_model,
            tokenizer=tokenizer,
        )
        pipeline.tokenizer.encode = partial(pipeline.tokenizer.encode, add_special_tokens=False)
        
    # task management
    pkl = load_or_init_conversation(args, topic, persona, user)

    # TODO: handle case where we've already finished this conversation

    for turn in range(len(pkl["history"])+1, args.turns+1):
        pkl_copy = copy.deepcopy(pkl)
        tick = time.time()
        messages = pkl2dict(pkl_copy)
        # prompt = llama_v2_prompt(messages)
        print("@"*100)
        # print(
        #     f"Prompting for the {turn}-th (one-based) turn with prompt:\n{prompt}"
        # )
        print(
            f"Prompting for the {turn}-th (one-based) turn with prompt:\n{json.dumps(messages, indent=2)}"
        )
        if client is not None:
            completion = client.chat.completions.create(model=args.model_name, messages=messages)
            sequence = completion.choices[0].message.content
        else:
            sequences = pipeline(
                messages,
                # prompt,
                do_sample=True,
                top_p=0.9,
                temperature=1.0,
                num_return_sequences=1,
                eos_token_id=pipeline.tokenizer.eos_token_id,
                max_new_tokens=400,
                return_full_text=False,
                clean_up_tokenization_spaces=True,
            )
            sequence = sequences[0]['generated_text']
        pkl["history"].append(process_answer(sequence))
        tok = time.time()
        print(f"Time taken for turn {turn}: {tok-tick:.2f} seconds")
        if len(pkl["history"]) % 2 == 0:
            save_conversation(args, pkl)
            # with open(f"selfchat/{file_name}", "wb") as handle:
            #     pickle.dump(pkl, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if client is not None:
            time.sleep(1)

    for turn in range(2, args.turns+1, 2):  # for 2, 4, 6, 8, 10, ...
        runs_to_run = args.runs - len(pkl["probed_history_per_turn"][turn])
        for _ in range(runs_to_run):
            temp_pkl = copy.deepcopy(pkl)
            temp_pkl["history"] = temp_pkl["history"][:turn]
            temp_pkl["history"].append(probe_str)
            pkl_copy = copy.deepcopy(temp_pkl)
            tick = time.time()
            messages = pkl2dict(pkl_copy)
            # prompt = llama_v2_prompt(messages)
            if client is not None:
                completion = client.chat.completions.create(model=args.model_name, messages=messages)
                sequence = completion.choices[0].message.content
            else:
                sequences = pipeline(
                    messages,
                    # prompt,
                    do_sample=True,
                    top_p=0.9,
                    temperature=1.0,
                    num_return_sequences=1,
                    eos_token_id=pipeline.tokenizer.eos_token_id,
                    max_new_tokens=400,
                    return_full_text=False,
                    clean_up_tokenization_spaces=True,
                )
                sequence = sequences[0]['generated_text']
            pkl["probed_history_per_turn"][turn].append(process_answer(sequence))
            tok = time.time()
            print(f"Time taken for probe turn {turn} ({_+1}/{runs_to_run}): {tok-tick:.2f} seconds")

        save_conversation(args, pkl)
        # with open(f"selfchat/{file_name}", "wb") as handle:
        #     pickle.dump(pkl, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if client is not None:
            time.sleep(1)
            
    # pprint(f"Saved to selfchat/{file_name}")

if __name__ == '__main__':
    main()
