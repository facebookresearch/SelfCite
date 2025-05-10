# Copyright (c) Meta Platforms, Inc. and affiliates.

import os, json, jsonlines
from tqdm import tqdm
import torch
from transformers import AutoTokenizer

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from longcite_modeling_llama import postprocess_citations, LlamaForCausalLM
import traceback
import torch.multiprocessing as mp
from collections import defaultdict
import time
import sys
import copy
import argparse

from liger_kernel.transformers import apply_liger_kernel_to_llama
apply_liger_kernel_to_llama()

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="THUDM/LongCite-llama3.1-8b")
parser.add_argument("--shard_id", type=int, default=0)
parser.add_argument("--total_shards", type=int, default=400)
parser.add_argument("--rerank_method", type=str, default="log_prob_drop_and_hold")
parser.add_argument("--sampling_path", type=str, default="preds-best-of-n-longcite-45k-truncated-25k")
parser.add_argument("--save_path", type=str, default="preds-best-of-n-longcite-45k-truncated-25k-edited-by-log_prob_drop_and_hold")
parser.add_argument("--num_gpus", type=int, default=1)
args = parser.parse_args()

shard_id = args.shard_id
total_shards = args.total_shards
rerank_method = args.rerank_method
num_gpus = args.num_gpus

mp.set_start_method("spawn", force=True)
gpus = list(range(num_gpus))
# ipts = json.load(open("LongBench-Cite.json"))
dataset_cache_file = "trimed_LongCite-45k.json"
ipts = json.load(open(dataset_cache_file))
# shard
ipts = [x for i, x in enumerate(ipts) if int(i) % total_shards == shard_id]
model_path = args.model_path
save_name = model_path.split('/')[-1]
save_dir = f"{args.sampling_path}/shard_{shard_id}_out_of_{total_shards}"
dump_dir = f"{args.save_path}/shard_{shard_id}_out_of_{total_shards}"
# os.makedirs(f'{save_dir}/tmp', exist_ok=True)
# make sure save_dir exists
os.makedirs(f'{dump_dir}/tmp', exist_ok=True)
fout_path = f'{save_dir}/tmp/{save_name}.jsonl'
dump_path = f'{dump_dir}/tmp/{save_name}.jsonl'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

for gpu_per_model in [num_gpus]:
    parallel_num = len(gpus) // gpu_per_model
    if os.path.exists(fout_path):
        with jsonlines.open(fout_path, 'r') as f:
            opts = [x for x in f]
    else:
        opts = []
    if os.path.exists(dump_path):
        with jsonlines.open(dump_path, 'r') as f:
            dumps = [x for x in f]
    else:
        dumps = []
    s = set(x['idx'] for x in opts)
    opts_dict = {x['idx']: x for x in opts}
    dumps_dict = {x['idx']: x for x in dumps}
    need_list = [x for x in ipts if x['idx'] not in dumps_dict]
    print(f'Model: {model_path} | GPU per model: {gpu_per_model} | parallel num: {parallel_num}')
    print(f'Already predict: {len(opts)} | Remain to predict: {len(need_list)}')
    if len(need_list) == 0: # 46T --> 40T --> 19T, 46 - 19 = 27
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
            if js['idx'] not in opts_dict:
                continue
            # try:
            context, query = js['context'], js['query']
            res = opts_dict[js['idx']]
            try:
                # 1. iterate through branches
                edits_to_do = {}
                for pos in sorted(res['branches'].keys(), key=lambda x: int(x)):
                    branch = res['branches'][pos]
                    original_log_prob = None
                    if len(branch.keys()) == 1: # skip the branch with only one option
                        continue
                    # pop the items if the key == '' --> only for open_to_cite
                    if '' in branch:
                        branch.pop('')
                    if rerank_method == 'log_prob':
                        # 2. get the average log prob of the branch
                        average_log_probs = {
                            key: sum(branch[key]['log_prob'])/len(branch[key]['log_prob']) for key in branch.keys()
                        }
                        best_key = max(average_log_probs, key=average_log_probs.get)
                        edits_to_do[int(pos)] = {'text': best_key, 'output': branch[best_key]['output']}
                        # import ipdb; ipdb.set_trace()
                    elif rerank_method == 'log_prob_drop':
                        log_prob_drops = {}
                        for key in tqdm(branch.keys(), desc=f"idx: {js['idx']} | pos: {pos}"):
                            to_drop = {}
                            drop_ranges = key.split('][')
                            for drop_range in drop_ranges:
                                try:
                                    start, end = drop_range.split('-')
                                except:
                                    print(f"Warn: weird drop_range is {drop_range}")
                                    continue
                                for i in range(int(start), int(end)+1):
                                    to_drop[i] = True
                            measure_log_prob_end = int(pos)
                            measure_log_prob_start = int(pos)
                            # trace back to the previous [1500, 25159], which is '></statement'
                            sequence = res['sequence']
                            while measure_log_prob_start > 0 and not (sequence[measure_log_prob_start] == 1500 and sequence[measure_log_prob_start+1] == 25159):
                                measure_log_prob_start -= 1
                            num_measure_tokens = measure_log_prob_end - measure_log_prob_start
                            # if original_log_prob is None:
                            #     original_log_prob = model.query_log_prob_drop_ablating(sequence[:measure_log_prob_end], num_measure_tokens, {}, context, query, tokenizer=tokenizer, max_input_length=128000)
                            ablated_log_prob, _, _ = model.query_log_prob_drop_ablating(sequence[:measure_log_prob_end], num_measure_tokens, to_drop, context, query, tokenizer=tokenizer, max_input_length=128000)
                            log_prob_drops[key] = - ablated_log_prob # original_log_prob
                        best_key = max(log_prob_drops, key=log_prob_drops.get)
                        edits_to_do[int(pos)] = {'text': best_key, 'output': branch[best_key]['output']}
                    elif rerank_method == 'log_prob_hold':
                        log_prob_drops = {}
                        for key in tqdm(branch.keys(), desc=f"idx: {js['idx']} | pos: {pos}"):
                            to_drop = {}
                            drop_ranges = key.split('][')
                            for drop_range in drop_ranges:
                                try:
                                    start, end = drop_range.split('-')
                                except:
                                    print(f"Warn: weird drop_range is {drop_range}")
                                    continue
                                for i in range(int(start), int(end)+1):
                                    to_drop[i] = True
                            measure_log_prob_end = int(pos)
                            measure_log_prob_start = int(pos)
                            # trace back to the previous [1500, 25159]
                            sequence = res['sequence']
                            while measure_log_prob_start > 0 and not (sequence[measure_log_prob_start] == 1500 and sequence[measure_log_prob_start+1] == 25159):
                                measure_log_prob_start -= 1
                            num_measure_tokens = measure_log_prob_end - measure_log_prob_start
                            # if original_log_prob is None:
                            #     original_log_prob = model.query_log_prob_drop_ablating(sequence[:measure_log_prob_end], num_measure_tokens, {}, context, query, tokenizer=tokenizer, max_input_length=128000)
                            pruned_log_prob, _, _ = model.query_log_prob_hold_pruning(sequence[:measure_log_prob_end], num_measure_tokens, to_drop, context, query, tokenizer=tokenizer, max_input_length=128000)
                            log_prob_drops[key] = pruned_log_prob #- original_log_prob
                        best_key = max(log_prob_drops, key=log_prob_drops.get)
                        edits_to_do[int(pos)] = {'text': best_key, 'output': branch[best_key]['output']}
                    elif rerank_method == 'log_prob_drop_and_hold':
                        log_prob_drops = {}
                        for key in tqdm(branch.keys(), desc=f"idx: {js['idx']} | pos: {pos}"):
                            to_drop = {}
                            drop_ranges = key.split('][')
                            for drop_range in drop_ranges:
                                try:
                                    start, end = drop_range.split('-')
                                except:
                                    print(f"Warn: weird drop_range is {drop_range}")
                                    continue
                                try:
                                    start, end = int(start), int(end)
                                except:
                                    print(f"Warn: weird start, end is {start}, {end}")
                                    continue
                                for i in range(int(start), int(end)+1):
                                    to_drop[i] = True
                            measure_log_prob_end = int(pos)
                            measure_log_prob_start = int(pos)
                            # trace back to the previous [1500, 25159]
                            sequence = res['sequence']
                            while measure_log_prob_start > 0 and not (sequence[measure_log_prob_start] == 1500 and sequence[measure_log_prob_start+1] == 25159):
                                measure_log_prob_start -= 1
                            num_measure_tokens = measure_log_prob_end - measure_log_prob_start
                            ablated_log_prob, _, _ = model.query_log_prob_drop_ablating(sequence[:measure_log_prob_end], num_measure_tokens, to_drop, context, query, tokenizer=tokenizer, max_input_length=128000)
                            torch.cuda.empty_cache()
                            pruned_log_prob, _, _ = model.query_log_prob_hold_pruning(sequence[:measure_log_prob_end], num_measure_tokens, to_drop, context, query, tokenizer=tokenizer, max_input_length=128000)
                            torch.cuda.empty_cache()
                            log_prob_drops[key] = pruned_log_prob - ablated_log_prob
                        best_key = max(log_prob_drops, key=log_prob_drops.get)
                        edits_to_do[int(pos)] = {'text': best_key, 'output': branch[best_key]['output']}
                    # 2. get the log prob of the branch
                    # log_prob = model.query_log_prob(context, query, branch, tokenizer=tokenizer, max_input_length=128000, max_new_tokens=1024)
                    # # 3. update the log prob
                    # res['branches'][pos]['log_prob'] = log_prob
                # 4. Given the edits to do, update the sequence
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
                        print(f"Warn: no '[' in the decoded sequence is {tokenizer.decode(edited_sequence[edit_start:edit_start+6])}")
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
                # res, sequence, branches = model.query_best_of_n(context, query, tokenizer=tokenizer, max_input_length=128000, max_new_tokens=1024)
                final_res = {
                    'idx': js['idx'],
                    'query': js['query'],
                    'old_prediction': res['prediction'],
                    'prediction': edited_text,
                    'statements': new_res['statements_with_citations'],
                }
                with open(dump_path, "a") as fout:
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

with jsonlines.open(dump_path, 'r') as f:
    opts = sorted([x for x in f], key=lambda x:x['idx'])
json.dump(opts, open(f'{dump_dir}/{save_name}.json', 'w'), indent=2, ensure_ascii=False)
