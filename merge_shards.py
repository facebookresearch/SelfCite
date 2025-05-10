# Copyright (c) Meta Platforms, Inc. and affiliates.

import sys
import os
import json
import jsonlines
import glob

jsonl_files = glob.glob(sys.argv[1])

data = {}

for jsonl_file in jsonl_files:
    if jsonl_file.endswith('json'):
        with open(jsonl_file, 'r') as f:
            for x in json.load(f):
                data[int(x['idx'])] = x
    else:
        with jsonlines.open(jsonl_file, 'r') as f:
            for x in f:
                data[int(x['idx'])] = x

# check missing
missing = []
for i in range(1000):
    if i not in data:
        missing.append(i)

print(f'--> {len(missing)} examples missing: {missing}')

print(f'--> finally {len(data)} examples')

with open(sys.argv[2], 'w') as f:
    # dump to json
    json.dump([data[x] for x in sorted(data)], f, indent=2, ensure_ascii=False)