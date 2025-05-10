# Copyright (c) Meta Platforms, Inc. and affiliates.

import re
from nltk.tokenize.punkt import PunktSentenceTokenizer

def get_context_in_prompt(context, query, tokenizer, max_input_length=128000, max_new_tokens=1024, truncated_from_last=False):
    
    def text_split_by_punctuation(original_text, return_dict=False):
        # text = re.sub(r'([a-z])\.([A-Z])', r'\1. \2', original_text)  # separate period without space
        text = original_text
        custom_sent_tokenizer = PunktSentenceTokenizer()
        punctuations = r"([。；！？])"  # For Chinese support

        separated = custom_sent_tokenizer.tokenize(text)
        separated = sum([re.split(punctuations, s) for s in separated], [])
        # Put the punctuations back to the sentence
        for i in range(1, len(separated)):
            if re.match(punctuations, separated[i]):
                separated[i-1] += separated[i]
                separated[i] = ''

        separated = [s for s in separated if s != ""]
        if len(separated) == 1:
            separated = original_text.split('\n\n')
        separated = [s.strip() for s in separated if s.strip() != ""]
        if not return_dict:
            return separated
        else:
            pos = 0
            res = []
            for i, sent in enumerate(separated):
                st = original_text.find(sent, pos)
                assert st != -1, sent
                ed = st + len(sent)
                res.append(
                    {
                        'c_idx': i,
                        'content': sent,
                        'start_idx': st,
                        'end_idx': ed,
                    }
                )
                pos = ed
            return res 

    def get_prompt(context, question):
        sents = text_split_by_punctuation(context, return_dict=True)
        splited_context = ""
        for i, s in enumerate(sents):
            st, ed = s['start_idx'], s['end_idx']
            assert s['content'] == context[st:ed], s
            ed = sents[i+1]['start_idx'] if i < len(sents)-1 else len(context)
            sents[i] = {
                'content': context[st:ed],
                'start': st,
                'end': ed,
                'c_idx': s['c_idx'],
            }
            splited_context += f"<C{i}>"+context[st:ed]
        prompt = '''Please answer the user's question based on the following document. When a sentence S in your response uses information from some chunks in the document (i.e., <C{s1}>-<C_{e1}>, <C{s2}>-<C{e2}>, ...), please append these chunk numbers to S in the format "<statement>{S}<cite>[{s1}-{e1}][{s2}-{e2}]...</cite></statement>". You must answer in the same language as the user's question.\n\n[Document Start]\n%s\n[Document End]\n\n%s''' % (splited_context, question)
        return prompt, sents, splited_context
    
    def get_citations(statement, sents):
        c_texts = re.findall(r'<cite>(.*?)</cite>', statement, re.DOTALL)
        spans = sum([re.findall(r"\[([0-9]+\-[0-9]+)\]", c_text, re.DOTALL) for c_text in c_texts], [])
        statement = re.sub(r'<cite>(.*?)</cite>', '', statement, flags=re.DOTALL)
        merged_citations = []
        for i, s in enumerate(spans):
            try:
                st, ed = [int(x) for x in s.split('-')]
                if st > len(sents) - 1 or ed < st:
                    continue
                st, ed = max(0, st), min(ed, len(sents)-1)
                assert st <= ed, str(c_texts) + '\t' + str(len(sents))
                if len(merged_citations) > 0 and st == merged_citations[-1]['end_sentence_idx'] + 1:
                    merged_citations[-1].update({
                        "end_sentence_idx": ed,
                        'end_char_idx': sents[ed]['end'],
                        'cite': ''.join([x['content'] for x in sents[merged_citations[-1]['start_sentence_idx']:ed+1]]),
                    })
                else:
                    merged_citations.append({
                        "start_sentence_idx": st,
                        "end_sentence_idx": ed,
                        "start_char_idx":  sents[st]['start'],
                        'end_char_idx': sents[ed]['end'],
                        'cite': ''.join([x['content'] for x in sents[st:ed+1]]),
                    })
            except:
                print(c_texts, len(sents), statement)
                raise
        return statement, merged_citations[:3]
    
    def postprocess(answer, sents, splited_context):
        res = []
        pos = 0
        new_answer = ""
        while True:
            st = answer.find("<statement>", pos)
            if st == -1:
                st = len(answer)
            ed = answer.find("</statement>", st)
            statement = answer[pos:st]
            if len(statement.strip()) > 5:
                res.append({
                    "statement": statement,
                    "citation": []
                })
                new_answer += f"<statement>{statement}<cite></cite></statement>"
            else:
                res.append({
                    "statement": statement,
                    "citation": None,
                })
                new_answer += statement
            
            if ed == -1:
                break

            statement = answer[st+len("<statement>"):ed]
            if len(statement.strip()) > 0:
                statement, citations = get_citations(statement, sents)
                res.append({
                    "statement": statement,
                    "citation": citations
                })
                c_str = ''.join(['[{}-{}]'.format(c['start_sentence_idx'], c['end_sentence_idx']) for c in citations])
                new_answer += f"<statement>{statement}<cite>{c_str}</cite></statement>"
            else:
                res.append({
                    "statement": statement,
                    "citation": None,
                })
                new_answer += statement
            pos = ed + len("</statement>")
        return {
            "answer": new_answer.strip(),
            "statements_with_citations": [x for x in res if x['citation'] is not None],
            "splited_context": splited_context.strip(),
            "all_statements": res,
        }
    
    def truncate_from_middle(prompt, max_input_length=None, tokenizer=None):
        if max_input_length is None:
            return prompt
        else:
            assert tokenizer is not None
            tokenized_prompt = tokenizer.encode(prompt, add_special_tokens=False)
            if len(tokenized_prompt) > max_input_length:
                half = int(max_input_length/2)
                prompt = tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)+tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            return prompt
    
    prompt, sents, splited_context = get_prompt(context, query)
    if not truncated_from_last:
        prompt = truncate_from_middle(prompt, max_input_length, tokenizer)
    else:
        prompt_ctx, prompt_query = prompt.split("\n[Document End]\n\n")
        prompt_query = "\n[Document End]\n\n" + prompt_query
        len_ctx = len(tokenizer.encode(prompt_ctx))
        len_query = len(tokenizer.encode(prompt_query))
        if len_ctx + len_query > max_input_length:
            target_len = max_input_length - len_query
            prompt_ctx = tokenizer.decode(tokenizer.encode(prompt_ctx)[:target_len], skip_special_tokens=True)
            prompt = prompt_ctx + prompt_query
            print(f"Truncated from last: {len_ctx} + {len_query} > {max_input_length}")

    return prompt
