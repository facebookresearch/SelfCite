# Copyright (c) Meta Platforms, Inc. and affiliates.

import os, json, jsonlines
from tqdm import tqdm
import sys
import re
import random
import tqdm
from transformers import AutoTokenizer
from make_data_get_prompt import get_context_in_prompt
from length_balancing import create_edited_reject_prediction


tokenizer = AutoTokenizer.from_pretrained("THUDM/LongCite-llama3.1-8b", trust_remote_code=True)

ipts = json.load(open("trimed_LongCite-45k.json", 'r'))

# reject_json = json.load(open(sys.argv[1]))
import glob
chosen_dict = {}
for file in tqdm.tqdm(glob.glob(sys.argv[1]), desc="Loading chosen json"):
    with open(file, 'r') as f:
        if file.endswith("json"):
            chosen_json = json.load(f)
        elif file.endswith("jsonl"):
            chosen_json = [json.loads(line) for line in f]
        else:
            raise ValueError(f"Unknown file type: {file}")
        
        for item in chosen_json:
            chosen_dict[item['idx']] = item

output_file = sys.argv[2]
# os.makedirs(os.path.dirname(output_file), exist_ok=True)

output_data = []

for item in tqdm.tqdm(ipts):
    if item['idx'] not in chosen_dict:
        continue
    idx = item['idx']
    context = item['context']
    query = item['query']
    reject_prediction = chosen_dict[idx]['old_prediction'].strip()
    chosen_prediction = chosen_dict[idx]['prediction'].strip()
    if reject_prediction == chosen_prediction:
        continue
    # chosen_prediction = merge_citations_in_text(chosen_prediction)
    edited_reject, edited_chosen, coverages, reach_max_tries = create_edited_reject_prediction(reject_prediction, chosen_prediction)
    full_prompt = get_context_in_prompt(context, query, tokenizer=tokenizer, max_input_length=25600-1024, max_new_tokens=1024, truncated_from_last=True)
    # full_prompt = '''Please answer the user's question based on the following document. When a sentence S in your response uses information from some chunks in the document (i.e., <C{s1}>-<C_{e1}>, <C{s2}>-<C{e2}>, ...), please append these chunk numbers to S in the format "<statement>{S}<cite>[{s1}-{e1}][{s2}-{e2}]...</cite></statement>". You must answer in the same language as the user's question.\n\n[Document Start]\n%s\n[Document End]\n\n%s''' % (context, query)
    output_data.append({
        'idx': idx,
        'prompt': full_prompt,
        'chosen': edited_chosen,
        'reject': edited_reject,
    })

print(f"Output data length: {len(output_data)}")
with open(output_file, 'w') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)
    
