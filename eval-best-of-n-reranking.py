# Copyright (c) Meta Platforms, Inc. and affiliates.

import os, json, jsonlines
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from longcite_modeling_llama import postprocess_citations, LlamaForCausalLM
import traceback
import torch.multiprocessing as mp
from collections import defaultdict
import time
import sys
import copy
import argparse
import glob
import random

torch.manual_seed(0)
random.seed(0)
torch.cuda.manual_seed_all(0)

parser = argparse.ArgumentParser()
parser.add_argument("--shard_id", type=int, default=0)
parser.add_argument("--total_shards", type=int, default=1)
parser.add_argument("--rerank_method", type=str, default="log_prob_drop_and_hold", help="log_prob_drop, log_prob, log_prob_hold, log_prob_drop_and_hold")
parser.add_argument("--model_path", type=str, default="THUDM/LongCite-llama3.1-8b")
parser.add_argument("--tokenizer_path", type=str, default="THUDM/LongCite-llama3.1-8b")
parser.add_argument("--sampling_files", type=str, default="preds/shard_*/tmp/*.jsonl")
parser.add_argument("--save_dir", type=str, default="preds-reranked")
parser.add_argument("--num_gpus", type=int, default=8)
parser.add_argument("--llama_chat_template", action="store_true")
parser.add_argument("--subset", type=str, default="all")
parser.add_argument("--length_limit", type=int, default=384, help="the maximum length of a citation for candidates. Set to 0 to disable.")

args = parser.parse_args()

shard_id = args.shard_id
total_shards = args.total_shards
rerank_method = args.rerank_method
num_gpus = args.num_gpus
model_path = args.model_path


# if file not exists, download it from https://github.com/THUDM/LongCite/raw/refs/heads/main/LongBench-Cite/LongBench-Cite.json
if not os.path.exists("LongBench-Cite.json"):
    import requests
    url = "https://github.com/THUDM/LongCite/raw/refs/heads/main/LongBench-Cite/LongBench-Cite.json"
    with open("LongBench-Cite.json", "wb") as f:
        f.write(requests.get(url).content)

ipts = json.load(open("LongBench-Cite.json"))
mp.set_start_method("spawn", force=True)
gpus = list(range(num_gpus))
# shard
print(f"Total number of examples is {len(ipts)}")
ipts = [x for i, x in enumerate(ipts) if int(i) % total_shards == shard_id]
print(f"After shard, the number of examples is {len(ipts)}")
if args.subset != 'all':
    ipts = [x for x in ipts if x['dataset'] == args.subset]
sampling_files = args.sampling_files

sampling_dict = {}
for sampling_file in glob.glob(sampling_files):
    if '.jsonl' in sampling_file:
        with jsonlines.open(sampling_file, 'r') as f:
            for d in f:
                sampling_dict[d['idx']] = d
    elif '.json' in sampling_file:
        with open(sampling_file, 'r') as f:
            json_file = json.load(f)
            for d in json_file:
                sampling_dict[d['idx']] = d
    else:
        print(f"Warn: weird file is {sampling_file}")
        continue

save_name = model_path.split('/')[-1] if not model_path.endswith('/') else model_path.split('/')[-2]
if total_shards > 1:
    save_dir = f"{args.save_dir}/{rerank_method}/shard_{shard_id}_out_of_{total_shards}"
else:
    save_dir = f"{args.save_dir}/{rerank_method}"
os.makedirs(f'{save_dir}/tmp', exist_ok=True)
fout_path = f'{save_dir}/tmp/{save_name}.jsonl'
dump_path = f'{save_dir}/{save_name}.json'
tokenizer_name = args.tokenizer_path
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

if args.length_limit != 0:
    length_tokenizer = AutoTokenizer.from_pretrained('THUDM/glm-4-9b-chat', trust_remote_code=True)
else:
    length_tokenizer = None


for gpu_per_model in [num_gpus]:
    parallel_num = len(gpus) // gpu_per_model
    if os.path.exists(fout_path):
        if '.jsonl' in fout_path:
            with jsonlines.open(fout_path, 'r') as f:
                opts = [x for x in f]
        else:
            opts = json.load(open(fout_path))
    else:
        opts = []
    opts_dict = {x['idx']: x for x in opts}
    need_list = [x for x in ipts if x['idx'] not in opts_dict]
    print(f'Model: {model_path} | GPU per model: {gpu_per_model} | parallel num: {parallel_num}')
    print(f'Already predict: {len(opts)} | Remain to predict: {len(need_list)}')
    if len(need_list) == 0:
        break

    def get_pred(rank, data):
        if rerank_method != 'log_prob':
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
            if js['idx'] not in sampling_dict:
                continue
            try:
                context, query = js['context'], js['query']
                res = sampling_dict[js['idx']]
                # 1. iterate through branches
                edits_to_do = {}
                for pos in sorted(res['branches'].keys(), key=lambda x: int(x)):
                    branch = res['branches'][pos]
                    for key in branch.keys():
                        if '><' in key: # replace wth "]["
                            new_key = key.replace('><', '][')
                            branch[new_key] = branch.pop(key)
                        if '>' in key or '<' in key:
                            new_key = key.replace('>', '').replace('<', '')
                            branch[new_key] = branch.pop(key)
                    if '' in branch:
                        branch.pop('')
                    if len(branch.keys()) <= 1: # skip the branch with only one option
                        continue
                    if rerank_method == 'log_prob':
                        # 2. get the average log prob of the branch
                        average_log_probs = {
                            key: sum(branch[key]['log_prob'])/len(branch[key]['log_prob']) for key in branch.keys()
                        }
                        best_key = max(average_log_probs, key=average_log_probs.get)
                        edits_to_do[int(pos)] = {'text': best_key, 'output': branch[best_key]['output']}
                    elif rerank_method == 'log_prob_drop':
                        reward_dict = {}
                        for key in tqdm(branch.keys(), desc=f"idx: {js['idx']} | pos: {pos}"):
                            to_drop = {}
                            drop_ranges = key.split('][')
                            for drop_range in drop_ranges:
                                try:
                                    start, end = drop_range.split('-')
                                    start = int(start)
                                    end = int(end)
                                except:
                                    print(f"Warn: weird drop_range is {drop_range}")
                                    continue
                                for i in range(start, end+1):
                                    to_drop[i] = True
                            measure_log_prob_end = int(pos)
                            measure_log_prob_start = int(pos)
                            # trace back to the previous [1500, 25159], which is '></statement'
                            sequence = res['sequence']
                            while measure_log_prob_start > 0 and not (sequence[measure_log_prob_start] == 1500 and sequence[measure_log_prob_start+1] == 25159):
                                measure_log_prob_start -= 1
                            num_measure_tokens = measure_log_prob_end - measure_log_prob_start
                            ablated_log_prob, dropped_sentences_length, cited_text = model.query_log_prob_drop_ablating(sequence[:measure_log_prob_end], num_measure_tokens, to_drop, context, query, tokenizer=tokenizer, max_input_length=128000, length_tokenizer=length_tokenizer, return_cited_text=False, llama_chat_template=args.llama_chat_template)
                            torch.cuda.empty_cache()
                            reward_dict[key] = - ablated_log_prob # original_log_prob is ignored as a constant
                            if args.length_limit != 0 and dropped_sentences_length > args.length_limit and len(to_drop) > 1:
                                reward_dict[key] -= 1000.0
                        best_key = max(reward_dict, key=reward_dict.get)
                        edits_to_do[int(pos)] = {'text': best_key, 'output': branch[best_key]['output']}
                    elif rerank_method == 'log_prob_hold':
                        reward_dict = {}
                        for key in tqdm(branch.keys(), desc=f"idx: {js['idx']} | pos: {pos}"):
                            to_drop = {}
                            drop_ranges = key.split('][')
                            for drop_range in drop_ranges:
                                try:
                                    start, end = drop_range.split('-')
                                    start = int(start)
                                    end = int(end)
                                except:
                                    print(f"Warn: weird drop_range is {drop_range}")
                                    continue
                                for i in range(start, end+1):
                                    to_drop[i] = True
                            measure_log_prob_end = int(pos)
                            measure_log_prob_start = int(pos)
                            # trace back to the previous [1500, 25159]
                            sequence = res['sequence']
                            while measure_log_prob_start > 0 and not (sequence[measure_log_prob_start] == 1500 and sequence[measure_log_prob_start+1] == 25159):
                                measure_log_prob_start -= 1
                            num_measure_tokens = measure_log_prob_end - measure_log_prob_start
                            pruned_log_prob, dropped_sentences_length, cited_text = model.query_log_prob_hold_pruning(sequence[:measure_log_prob_end], num_measure_tokens, to_drop, context, query, tokenizer=tokenizer, max_input_length=128000, length_tokenizer=length_tokenizer, return_cited_text=False, llama_chat_template=args.llama_chat_template)
                            torch.cuda.empty_cache()
                            reward_dict[key] = pruned_log_prob #- original_log_prob is ignored as a constant
                            if args.length_limit != 0 and dropped_sentences_length > args.length_limit and len(to_drop) > 1:
                                reward_dict[key] -= 1000.0
                        best_key = max(reward_dict, key=reward_dict.get)
                        edits_to_do[int(pos)] = {'text': best_key, 'output': branch[best_key]['output']}
                    elif rerank_method == 'log_prob_drop_and_hold':
                        reward_dict = {}
                        for key in tqdm(branch.keys(), desc=f"idx: {js['idx']} | pos: {pos}"):
                            to_drop = {}
                            drop_ranges = key.split('][')
                            for drop_range in drop_ranges:
                                try:
                                    start, end = drop_range.split('-')
                                    start = int(start)
                                    end = int(end)
                                    if start > end:
                                        print(f"Warn: start is larger than end, {start} > {end}")
                                        continue
                                    if start < 0 or end < 0:
                                        print(f"Warn: start or end is negative, {start} or {end}")
                                        continue
                                except:
                                    print(f"Warn: weird drop_range is {drop_range}")
                                    continue
                                for i in range(start, end+1):
                                    to_drop[i] = True
                            measure_log_prob_end = int(pos)
                            measure_log_prob_start = int(pos)
                            # trace back to the previous [1500, 25159]
                            sequence = res['sequence']
                            while measure_log_prob_start > 0 and not (sequence[measure_log_prob_start] == 1500 and sequence[measure_log_prob_start+1] == 25159):
                                measure_log_prob_start -= 1
                            num_measure_tokens = measure_log_prob_end - measure_log_prob_start
                            ablated_log_prob, dropped_sentences_length, cited_text = model.query_log_prob_drop_ablating(sequence[:measure_log_prob_end], num_measure_tokens, to_drop, context, query, tokenizer=tokenizer, max_input_length=128000, length_tokenizer=length_tokenizer, return_cited_text=False, llama_chat_template=args.llama_chat_template)
                            torch.cuda.empty_cache()
                            pruned_log_prob, _, _ = model.query_log_prob_hold_pruning(sequence[:measure_log_prob_end], num_measure_tokens, to_drop, context, query, tokenizer=tokenizer, max_input_length=128000, length_tokenizer=length_tokenizer, return_cited_text=False, llama_chat_template=args.llama_chat_template)
                            torch.cuda.empty_cache()
                            reward_dict[key] = pruned_log_prob - ablated_log_prob
                            if args.length_limit != 0 and dropped_sentences_length > args.length_limit and len(to_drop) > 1:
                                reward_dict[key] -= 1000.0
                        if len(reward_dict) == 0:
                            print(f"Warn: no valid ranges for {js['idx']} | {pos}: {branch.keys()}")
                            continue
                        best_key = max(reward_dict, key=reward_dict.get)
                        edits_to_do[int(pos)] = {'text': best_key, 'output': branch[best_key]['output']}
                    else:
                        raise NotImplementedError(f"rerank_method is {rerank_method}")
                # Given the edits to do, update the sequence
                sequence = res['sequence']
                # edit from the last position to the first position
                edited_sequence = copy.deepcopy(sequence)
                for pos in sorted(edits_to_do.keys(), reverse=True):
                    edit = edits_to_do[pos]
                    edit_start = pos # end is the first 32061 after pos
                    edit_end = -1
                    offset = 0
                    insert_end_of_cite = []
                    if '[' not in tokenizer.decode(edited_sequence[edit_start:edit_start+6]):
                        # print(f"Warn: no '[' in the decoded sequence is {tokenizer.decode(edited_sequence[edit_start:edit_start+6])}")
                        while not tokenizer.decode(edited_sequence[:edit_start+offset]).endswith('<cite></') and offset < 5:
                            offset += 1
                        if offset != 3:
                            print(f"Warn: offset is not 3, but {offset}. The decoded sequence is {tokenizer.decode(edited_sequence[:edit_start+offset])}")
                        edited_sequence[edit_start+offset-1] = tokenizer.encode('>[', add_special_tokens=False)[0]
                        edit_end = edit_start+offset
                        insert_end_of_cite = tokenizer.encode(']</', add_special_tokens=False)
                        edited_sequence = edited_sequence[:edit_start+offset] + edit['output'] + insert_end_of_cite + edited_sequence[edit_end:]
                    else:
                        while not tokenizer.decode(edited_sequence[:edit_start+offset]).endswith('[') and offset < 5:
                            offset += 1
                        if offset != 3:
                            print(f"Warn: offset is not 3, but {offset}. The decoded sequence is {tokenizer.decode(edited_sequence[:edit_start+offset])}")
                        for i in range(pos, len(edited_sequence)):
                            if edited_sequence[i] == 32061:
                                edit_end = i
                                break
                        if edit_end == -1: # skip editing
                            print(f"Warn: skip editing {js['idx']} | {pos} | {edited_sequence[pos:pos+10]}")
                            continue
                        edited_sequence = edited_sequence[:edit_start+offset] + edit['output'] + edited_sequence[edit_end:]
                edited_text = tokenizer.decode(edited_sequence)
                new_res = postprocess_citations(edited_text, context, query, tokenizer=tokenizer)
                final_res = {
                    'idx': js['idx'],
                    'dataset': js['dataset'],
                    'query': js['query'],
                    'answer': js['answer'],
                    'few_shot_scores': js['few_shot_scores'],
                    'prediction': edited_text,
                    'statements': new_res['statements_with_citations'],
                }
                with open(fout_path, "a") as fout:
                    fout.write(json.dumps(final_res, ensure_ascii=False)+'\n')
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
json.dump(opts, open(dump_path, 'w'), indent=2, ensure_ascii=False)
