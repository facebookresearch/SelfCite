# Copyright (c) Meta Platforms, Inc. and affiliates.

import os, json, jsonlines
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from longcite_modeling_llama import LlamaForCausalLM
import traceback
import torch.multiprocessing as mp
from collections import defaultdict
import time
import sys
import argparse
import random

torch.manual_seed(0)
random.seed(0)
torch.cuda.manual_seed_all(0)

parser = argparse.ArgumentParser()
parser.add_argument("--shard_id", type=int, default=0)
parser.add_argument("--total_shards", type=int, default=1)
parser.add_argument("--model_path", type=str, default="THUDM/LongCite-llama3.1-8b")
parser.add_argument("--tokenizer_path", type=str, default="THUDM/LongCite-llama3.1-8b")
parser.add_argument("--save_dir", type=str, default="preds")
parser.add_argument("--num_gpus", type=int, default=8)
parser.add_argument("--llama_chat_template", action="store_true")
parser.add_argument("--subset", type=str, default="all")
args = parser.parse_args()

model_path = args.model_path
num_gpus = args.num_gpus
gpus = list(range(num_gpus))
shard_id = args.shard_id
total_shards = args.total_shards


# if file not exists, download it from https://github.com/THUDM/LongCite/raw/refs/heads/main/LongBench-Cite/LongBench-Cite.json
if not os.path.exists("LongBench-Cite.json"):
    import requests
    url = "https://github.com/THUDM/LongCite/raw/refs/heads/main/LongBench-Cite/LongBench-Cite.json"
    with open("LongBench-Cite.json", "wb") as f:
        f.write(requests.get(url).content)

# mp.set_start_method("spawn", force=True)
ipts = json.load(open("LongBench-Cite.json"))
# shard
ipts = [x for i, x in enumerate(ipts) if int(i) % total_shards == shard_id]
if args.subset != 'all':
    ipts = [x for x in ipts if x['dataset'] == args.subset]
save_name = model_path.split('/')[-1]
if total_shards > 1:
    save_dir = f"{args.save_dir}/shard_{shard_id}_out_of_{total_shards}"
else:
    save_dir = args.save_dir
os.makedirs(f'{save_dir}/tmp', exist_ok=True)
fout_path = f'{save_dir}/tmp/{save_name}.jsonl'
tokenizer_name = args.tokenizer_path
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

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
                res, sequence, branches = model.query_best_of_n(context, query, tokenizer=tokenizer, max_input_length=128000, max_new_tokens=1024, llama_chat_template=args.llama_chat_template)
                res = {
                    'idx': js['idx'],
                    'dataset': js['dataset'],
                    'query': js['query'],
                    'answer': js['answer'],
                    'few_shot_scores': js['few_shot_scores'],
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
            except torch.OutOfMemoryError as e:
                print('OOM')
                print(js['idx'])
                print(query)
                print('-'*200)
                # cleanup
                del model
                torch.cuda.empty_cache()
                model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto", quantization_config=quant_config, attn_implementation="flash_attention_2")
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
