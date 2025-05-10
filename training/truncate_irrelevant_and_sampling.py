# Copyright (c) Meta Platforms, Inc. and affiliates.

import os, json, jsonlines
from tqdm import tqdm
from transformers import AutoTokenizer
import traceback
import torch.multiprocessing as mp
from collections import defaultdict
import time
import sys
import re
import datasets
from trim import trim_sents_by_key_distance

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="THUDM/LongCite-llama3.1-8b")
parser.add_argument("--save_dir", type=str, default="preds-best-of-n-longcite-45k-truncated-25k")
parser.add_argument("--shard_id", type=int, default=0)
parser.add_argument("--total_shards", type=int, default=400)
parser.add_argument("--num_gpus", type=int, default=1)
args = parser.parse_args()

model_path = args.model_path
save_dir = args.save_dir
shard_id = args.shard_id
total_shards = args.total_shards
num_gpus = args.num_gpus

mp.set_start_method("spawn", force=True)
gpus = list(range(num_gpus))

save_name = model_path.split('/')[-1]
save_dir = f"{save_dir}/shard_{shard_id}_out_of_{total_shards}"
os.makedirs(f'{save_dir}/tmp', exist_ok=True)
fout_path = f'{save_dir}/tmp/{save_name}.jsonl'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
max_length = 25600

def truncate_irrelevant(prompt, response, max_length):
    context, query = x['prompt'].split("\n[Document End]\n\n")
    system_prompt, context = context.split("\n\n[Document Start]\n")
    cleaned_context = re.sub(r'<C\d+>', '', context)
    cleaned_prompt = system_prompt + "\n\n[Document Start]\n" + cleaned_context + "\n[Document End]\n\n" + query
    len_prompt = len(tokenizer.encode(cleaned_prompt))
    len_response = len(tokenizer.encode(response))
    if len_prompt + len_response < max_length:
        return cleaned_context, query

    len_system_prompt = len(tokenizer.encode(system_prompt + "\n\n[Document Start]\n"))
    len_query = len(tokenizer.encode("\n[Document End]\n\n" + query))
    
    # get sents from the context by splitting the context by <C\d+> tags
    sents = re.split(r'<C\d+>', context)[1:]
    sent_lens = [len(tokenizer.encode(sent)) for sent in sents]
    constaint = max_length - 1024 - len_system_prompt - len_query

    cited = {}
    # find things like <cite>[3-5][7-9]...</cite> in the response, register 3,4,5,7,8,9 into `cited`
    c_texts = re.findall(r'<cite>(.*?)</cite>', response, re.DOTALL)
    for c_text in c_texts:
        for c in re.findall(r'\[(\d+)-(\d+)\]', c_text):
            for i in range(int(c[0]), int(c[1])+1):
                cited[i] = True

    cited = set(cited.keys())
    trimed_sents, removed = trim_sents_by_key_distance(sents, sent_lens, cited, constaint)
    trimed_context = "".join(trimed_sents)
    return trimed_context, query

data = datasets.load_dataset("THUDM/LongCite-45k", split="train")
ipts = []
dataset_cache_file = "trimed_LongCite-45k.json"
if os.path.exists(dataset_cache_file):
    ipts = json.load(open(dataset_cache_file))
else:
    multithread = False
    if not multithread:
        for i, x in enumerate(tqdm(data, desc=f"Trimming context to {max_length}")):
            context, query = truncate_irrelevant(x['prompt'], x['response'], max_length)
            ipts.append({
                'idx': i,
                'context': context,
                'query': query,
            })
    else:
        def trim_context(i, x):
            context, query = truncate_irrelevant(x['prompt'], x['response'], max_length)
            return {
                'idx': i,
                'context': context,
                'query': query,
            }
        with mp.Pool(32) as p:
            ipts = p.starmap(trim_context, enumerate(data))
    json.dump(ipts, open(dataset_cache_file, 'w'), indent=2, ensure_ascii=False)
# shard

ipts = [x for i, x in enumerate(ipts) if int(i) % total_shards == shard_id]


import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from longcite_modeling_llama import LlamaForCausalLM

for gpu_per_model in [num_gpus]:
    parallel_num = len(gpus) // gpu_per_model
    if os.path.exists(fout_path):
        with jsonlines.open(fout_path, 'r') as f:
            opts = [x for x in f if x['branches'] != {}]
    else:
        opts = []
    s = set(x['idx'] for x in opts)
    need_list = [x for x in ipts if x['idx'] not in s]
    print(f'Model: {model_path} | GPU per model: {gpu_per_model} | parallel num: {parallel_num}')
    print(f'Already predict: {len(opts)} | Remain to predict: {len(need_list)}')
    if len(need_list) == 0:
        break

    def get_pred(rank, data):
        os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(x) for x in gpus[rank*gpu_per_model:(rank+1)*gpu_per_model])
        # 4bit
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_storage=torch.uint8,
            bnb_4bit_use_double_quant=False,
        )
        model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto", quantization_config=quant_config, attn_implementation="flash_attention_2")
        for js in tqdm(data):
            try:
                context, query = js['context'], js['query']
                res, sequence, branches = model.query_best_of_n(context, query, tokenizer=tokenizer, max_input_length=25600-1024, max_new_tokens=1024, truncated_from_last=True)
                res = {
                    'idx': js['idx'],
                    'query': js['query'],
                    'prediction': res['answer'],
                    'statements': res['statements_with_citations'],
                    'branches': branches,
                    'sequence': sequence,
                }
                with open(fout_path, "a") as fout:
                    fout.write(json.dumps(res, ensure_ascii=False)+'\n')
                    fout.flush()
            except KeyboardInterrupt as e:
                raise e
            except:
                print(js['idx'])
                print(query)
                traceback.print_exc()
                print('-'*200)
        del model

    need_list_subsets = [need_list[i::parallel_num] for i in range(parallel_num)]
    # processes = []
    # for rank in range(parallel_num):
    #     p = mp.Process(target=get_pred, args=(rank, need_list_subsets[rank]))
    #     p.start()
    #     processes.append(p)
    # for p in processes:
    #     p.join()

    # sequential
    for rank in range(parallel_num):
        get_pred(rank, need_list_subsets[rank])

with jsonlines.open(fout_path, 'r') as f:
    opts = sorted([x for x in f], key=lambda x:x['idx'])
json.dump(opts, open(f'{save_dir}/{save_name}.json', 'w'), indent=2, ensure_ascii=False)
