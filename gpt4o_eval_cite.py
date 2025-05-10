# The code is a combination of:
# - https://github.com/THUDM/LongCite/blob/main/LongBench-Cite/eval_cite.py
# - https://github.com/THUDM/LongCite/blob/main/LongBench-Cite/auto_scorer.py
# - https://github.com/THUDM/LongCite/blob/main/utils/llm_api.py
# Please always refer to the original code for the most up-to-date version.
import os, json, jsonlines
import sys
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
import re
import requests
import time
import traceback
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--openai_key", type=str, default=None)
# pred files joined by ,
parser.add_argument("--pred_paths", type=str, default=None)
# datasets to evaluate, joined by ,
parser.add_argument("--datasets", type=str, default=None)
args = parser.parse_args()

pred_paths = args.pred_paths.split(',')

defined_datasets = [
    "longbench-chat", 
    "multifieldqa_en", 
    "multifieldqa_zh", 
    "multifieldqa", 
    "hotpotqa", 
    "dureader", 
    "gov_report"
]

datasets = args.datasets.split(',') if args.datasets else defined_datasets

for dataset in datasets:
    if dataset not in defined_datasets:
        raise ValueError(f"Unknown dataset: {dataset}")

api_key = args.openai_key
if not api_key:
    raise ValueError("Please provide openai_key")

api_url = 'https://api.openai.com/v1/chat/completions'

GPT_MODEL = 'gpt-4o-2024-05-13'

need_citation_prompt_template = """You are an expert in evaluating text quality. You will receive a user's question regarding their uploaded document (due to the length of the document, it is not shown to you), an AI assistant's response based on the document, and a sentence from the response. Your task is to determine whether this sentence is a factual statement made based on the information in the document that requires citation, rather than an introductory sentence, transition sentence, or a summary, reasoning, or inference based on the previous response.
Ensure that you do not use any other external information during your evaluation. 
Please first provide your judgment (answer with [[Yes]] or [[No]]), then provide your analysis in the format "Need Citation: [[Yes/No]] Analysis: ...".\n\n{}
"""

def query_llm(messages, model, temperature=1.0, max_new_tokens=1024, stop=None, return_usage=False):
    tries = 0
    while tries < 5:
        tries += 1
        try:
            if 'claude' not in model:
                headers = {
                    'Authorization': "Bearer {}".format(api_key),
                }
            else:
                headers = {
                    'x-api-key': api_key,
                    'anthropic-version': "2023-06-01",
                }
                
            resp = requests.post(api_url, json = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_new_tokens,
                "stop" if 'claude' not in model else 'stop_sequences': stop,
            }, headers=headers, timeout=600)
            # print(resp.text)
            # print(resp.status_code)
            if resp.status_code != 200:
                raise Exception(resp.text)
            resp = resp.json()
            break
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            if "maximum context length" in str(e):
                raise e
            elif "triggering" in str(e):
                return 'Trigger OpenAI\'s content management policy.'
            print("Error Occurs: \"%s\"        Retry ..."%(str(e)))
            time.sleep(1)
    else:
        print("Max tries. Failed.")
        return None
    try:
        if 'content' not in resp["choices"][0]["message"] and 'content_filter_results' in resp["choices"][0]:
            resp["choices"][0]["message"]["content"] = 'Trigger OpenAI\'s content management policy.'
        if return_usage:
            return resp["choices"][0]["message"]["content"], resp['usage']
        else:
            return resp["choices"][0]["message"]["content"]
    except: 
        return None

def cat_qa_and_statement(question, answer, statement):
    prompt = f"<question>\n{question.strip()}\n</question>\n\n<response>\n{answer.strip()}\n</response>\n\n<sentence>\n{statement.strip()}\n</sentence>"
    return prompt

def need_citation_to_score(s):
    l = re.findall(r'\[\[([ /a-zA-Z]+)\]\]', s)
    if l:
        l = l[0]
        if "yes".lower() in l.lower():
            return 1
        else:
            return 0
    else:
        return None

def need_citation(question, answer, sentence):
    prompt = need_citation_prompt_template.format(cat_qa_and_statement(question, answer, sentence))
    for t in range(5):
        msg = [{'role': 'user', 'content': prompt}]
        output = query_llm(msg, model=GPT_MODEL, temperature=0 if t == 0 else 1, max_new_tokens=10, stop="Analysis:", return_usage=True)
        if isinstance(output, tuple):
            output, usage = output
            score = need_citation_to_score(output)
        else:
            score, usage = None, None
        if score is None:
            print("Unexcept need_citation output: ", output)
            if 'Trigger' in output:
                break
            continue
        else:
            break
    return score, output, usage
    
support_prompt_template = """You are an expert in evaluating text quality. You will receive a user's question about an uploaded document, a factual statement from an AI assistant's response based on that document, and a snippet from the document (since the document is too long to display in full). Your task is to carefully assess whether this statement is supported by the snippet. Please use the following scale to generate your rating:
- [[Fully supported]] - Most information in the statement is supported by or extracted from the snippet. This applies only to cases where the statement and parts of the snippet are almost identical. 
- [[Partially supported]] - More than half of the content in the statement is supported by the snippet, but a small portion is either not mentioned or contradicts the snippet. For example, if the statement has two key points and the snippet supports only one of them, it should be considered [Partially supported].
- [[No support]] - The statement is largely unrelated to the snippet, or most key points in the statement do not align with the content of the snippet.
Ensure that you do not use any information or knowledge outside of the snippet when evaluating. 
Please provide the rating first, followed by the analysis, in the format "Rating: [[...]] Analysis: ...". \n\n{}"""

def cat_question_statement_context(question, statement, context):
    prompt = f"<question>\n{question.strip()}\n</question>\n\n<statement>\n{statement.strip()}\n</statement>\n\n<snippet>\n{context.strip()}\n</snippet>\n\n"
    return prompt

def support_level_to_score(s):
    l = re.findall(r'\[\[([ /a-zA-Z]+)\]\]', s)
    if l:
        l = l[0]
        if "fully".lower() in l.lower():
            return 1
        elif "partially".lower() in l.lower():
            return 0.5
        else:
            return 0
    else:
        return None
    
def is_support(question, statement, context):
    if context == "":
        return 0, "No matched citation", None
    prompt = support_prompt_template.format(cat_question_statement_context(question, statement, context))
    for t in range(5):
        msg = [{'role': 'user', 'content': prompt}]
        output = query_llm(msg, model=GPT_MODEL, temperature=0 if t == 0 else 1, max_new_tokens=10, stop="Analysis:", return_usage=True)
        if isinstance(output, tuple):
            output, usage = output
            score = support_level_to_score(output)
        else:
            score, usage = None, None
        if score is None:
            print("Unexcept support output: ", output)
            if 'Trigger' in output:
                break
            continue
        else:
            break
    return score, output, usage
    
def score_recall(question, answer, statements_with_citations):
    scores, usages = [], []
    for js in statements_with_citations:
        statement, citations = js['statement'], js['citation']
        matched_citations = [c['cite'] for c in citations]
        
        if len(matched_citations) > 0:
            context = '\n\n'.join(matched_citations).strip()
            score, output, usage = is_support(question, statement, context)
            usages.append(usage)
            js.update({
                "support_output": output,
                "support_score": score,
            })
            if score is None:
                print("ERROR\tUnexcept support output: ", statement + '\t>>\t' + output)
                raise NotImplementedError
            else:
                scores.append(score)
        else:
            score, output, usage = need_citation(question, answer, statement)
            usages.append(usage)
            js.update({
                "support_output": output,
                "support_score": 1 - score if score is not None else None,
            })
            if score is None:
                print("ERROR\tUnexcept need_citation output: ", output)
                raise NotImplementedError
            else:
                scores.append(1-score)
    if len(scores) == 0:
        return 0, statements_with_citations, usages
    else:
        return np.mean(scores), statements_with_citations, usages
        
relevant_prompt_template = """You are an expert in evaluating text quality. You will receive a user's question about an uploaded document, a factual statement from an AI assistant's response based on that document, and a snippet from the document (since the document is too long to display in full). Your task is to carefully assess whether the snippet contains some key information of the statement. Please use the following grades to generate the rating:
- [[Relevant]] - Some key points of the statement are supported by the snippet or extracted from it.
- [[Unrelevant]] - The statement is almostly unrelated to the snippet, or all key points of the statement are inconsistent with the snippet content.
Ensure that you do not use any information or knowledge outside of the snippet when evaluating. 
Please provide the rating first, followed by the analysis, in the format "Rating: [[...]] Analysis: ...". \n\n{}"""

def relevant_level_to_score(s):
        l = re.findall(r'\[\[([ /a-zA-Z]+)\]\]', s)
        if l:
            l = l[0]
            if "unrelevant".lower() in l.lower():
                return 0
            else:
                return 1
        else:
            return None
        
def is_relevant(question, statement, citation):
    prompt = relevant_prompt_template.format(cat_question_statement_context(question, statement, citation))
    for t in range(5):
        msg = [{'role': 'user', 'content': prompt}]
        output = query_llm(msg, model=GPT_MODEL, temperature=0 if t == 0 else 1, max_new_tokens=10, stop="Analysis:", return_usage=True)
        if isinstance(output, tuple):
            output, usage = output
            score = relevant_level_to_score(output)
        else:
            score, usage = None, None
        if score is None:
            print("Unexcept relevant output: ", output)
            if 'Trigger' in output:
                break
            continue
        else:
            break
    return score, output, usage

def score_precision(question, answer, statements_with_citations):
    scores, usages = [], []
    for js in statements_with_citations:
        statement, citations = js['statement'], js['citation']
        for c in citations:
            score, output, usage = is_relevant(question, statement, c['cite'])
            usages.append(usage)
            c.update({
                "relevant_output": output,
                "relevant_score": score,
            })
            if score is None:
                print("ERROR\tUnexcept relevant output: ", output)
                raise NotImplementedError
            else:
                scores.append(score)
        
    if len(scores) == 0:
        return 0, statements_with_citations, usages
    else:
        return np.mean(scores), statements_with_citations, usages

def get_citation_score(js, max_statement_num=None):
    question, answer, statements_with_citations = js['query'], js['prediction'], js['statements']
    answer = re.sub(r"<cite>.*?</cite>", "", answer, flags=re.DOTALL)
    answer = answer.replace('<statement>', '').replace('</statement>', '')
    answer = re.sub(r"<\|reserved_special_token_0\|>.*?<\|reserved_special_token_1\|>", " ", answer, flags=re.DOTALL)
    answer = re.sub(r"<\|reserved_special_token_\d+\|>", " ", answer, flags=re.DOTALL)
    if max_statement_num and len(statements_with_citations) > max_statement_num:
        print(f"Too many statments, only evaluate {max_statement_num} of {len(statements_with_citations)} statements.")
        statements_with_citations = statements_with_citations[:max_statement_num]

    recall, _, usages1 = score_recall(question, answer, statements_with_citations)
    precision, _, usages2 = score_precision(question, answer, statements_with_citations)
    js['citation_recall'] = recall
    js['citation_precision'] = precision
    js['citation_f1'] = 0.0 if recall + precision == 0 else (2 * recall * precision) / (recall + precision)
    try:
        js['gpt_usage'] = {
            'prompt_tokens': sum(x['prompt_tokens'] for x in (usages1 + usages2)),
            'completion_tokens': sum(x['completion_tokens'] for x in (usages1 + usages2)),
        }
    except:
        import ipdb; ipdb.set_trace()
    return js

def process(item):
    try:
        js, fout_path = item
        js = get_citation_score(js, max_statement_num=40)
        del js['answer'], js['few_shot_scores']
        with open(fout_path, "a") as fout:
            fout.write(json.dumps(js, ensure_ascii=False)+'\n')
            fout.flush()
        return js
    except:
        print(js['query'])
        traceback.print_exc()
        print('-'*200)
        return None

if __name__ == "__main__":
    os.makedirs(f"./scores_cite/tmp", exist_ok=True)
    for path in pred_paths:
        print(path)
        save_name = path.split('/')[-1]
        fout_path = f"./scores_cite/tmp/{save_name}.jsonl"
        ipts = [x for x in json.load(open(path)) if x['dataset'] in datasets] 
        for trie in range(1):
            if os.path.exists(fout_path):
                with jsonlines.open(fout_path, 'r') as f:
                    opts = [x for x in f]
            else:
                opts = []
            s = set(x['idx'] for x in opts)
            need_list = [(x, fout_path) for x in ipts if x['idx'] not in s]#[:1]
            print(f"Need to process {len(need_list)}")

            with Pool(4) as p:
                rst = list(tqdm(p.imap(process, need_list), total=len(need_list)))
            opts = opts + [x for x in rst if x is not None]
            opts = sorted(opts, key=lambda x:x['idx'])
            
            result = {'scores': {}}
            for dataset in datasets:
                parts = [x for x in opts if dataset in x['dataset']]
                recall, precision, f1 = np.mean([x['citation_recall'] for x in parts]), np.mean([x['citation_precision'] for x in parts]), np.mean([x['citation_f1'] for x in parts])
                finish = sum([1 for x in opts if dataset in x['dataset']]) == sum([1 for x in ipts if dataset in x['dataset']])
                result['scores'][dataset] = {
                    "citation_recall": float(recall),
                    "citation_precision": float(precision),
                    "citation_f1": float(f1),
                    'finish': finish,
                }
            
            finish = len(opts) == len(ipts)
            gpt_usage = {
                'prompt_tokens': sum(x['gpt_usage']['prompt_tokens'] for x in opts),
                'completion_tokens': sum(x['gpt_usage']['completion_tokens'] for x in opts),
                'gpt_model': GPT_MODEL
            }
            result.update({
                "avg_citation_recall": float(np.mean([x['citation_recall'] for k, x in result['scores'].items() if k not in ['multifieldqa_en', 'multifieldqa_zh']])),
                "avg_citation_precision": float(np.mean([x['citation_precision'] for k, x in result['scores'].items() if k not in ['multifieldqa_en', 'multifieldqa_zh']])),
                "avg_citation_f1": float(np.mean([x['citation_f1'] for k, x in result['scores'].items() if k not in ['multifieldqa_en', 'multifieldqa_zh']])),
                "finish": finish,
                "gpt_usage": gpt_usage,
            })
            opts.append(result)
            if finish:
                break

        print(result)
        json.dump(opts, open(f"./scores_cite/{save_name}.json", "w"), indent=2, ensure_ascii=False)
