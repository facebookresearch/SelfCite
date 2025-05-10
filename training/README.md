# SelfCite Preference Optimization with SimPO

The training code is based on [SimPO](https://github.com/princeton-nlp/SimPO), which is modified from [alignment-handbook](https://github.com/huggingface/alignment-handbook).


## Setup

Install the dependencies with conda and pip:
```
conda env create -n simpo python=3.10.15
pip install torch==2.4.1
pip install flashinfer-python -i https://flashinfer.ai/whl/cu121/torch2.4
pip install -r training-requirements.txt
```

or try to create a conda env from `training-env.yaml`: (notice that sometimes the installation fails due to the support of `torch` and some version conflicts)
```
conda env create -f training-env.yaml -n simpo
conda activate simpo
```

## Generate training data with BoN

### Step 1: Sampling candidates

If you have multiple nodes, e.g. 4 nodes, you can add the following arguments to split the examples into 100 shards and run them independently.

```bash
HF_DATASETS_CACHE=[path to cache] python truncate_irrelevant_and_sampling.py \
    --shard_id $TASK_ID \ # 0-3
    --total_shards 4 \
    --model_path THUDM/LongCite-llama3.1-8b \
    --save_dir $YOUR_SAMPLING_OUTPUT_DIR \
    --num_gpus $NUM_GPUS
```

### Step 2: Reranking candidates

```bash
HF_DATASETS_CACHE=[path to cache] python truncate_irrelevant_and_reranking.py \
    --shard_id $TASK_ID \ # 0-3
    --total_shards 4 \
    --model_path THUDM/LongCite-llama3.1-8b \
    --sampling_path $YOUR_SAMPLING_OUTPUT_DIR \
    --save_path $YOUR_RERANKING_OUTPUT_DIR \
    --num_gpus $NUM_GPUS
```

## Step 3: Generate preference pairs

```bash
python make_preference_data.py "$YOUR_RERANKING_OUTPUT_DIR/shard_*_out_of_*/tmp/*.jsonl" [output_file]
```

The output file will be used as the input for the following training step.

## Training

Let's say we have `selfcite-train-2k.json` as the output file from the data
generation.

```bash
HF_DATASETS_CACHE=[path to cache] ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file accelerate_configs/deepspeed_zero3.yaml scripts/run_simpo.py \
    training_configs/longcite-8b-simpo.yaml \
    --dataset_mixer="selfcite-train-2k.json:1.0" \
    --run_name=debug \
    --learning_rate=3e-7 \
    --output_dir=tmp_debug
```

`"selfcite-train-2k.json:1.0"` means using the whole dataset. If you want to use a subset of the data, you can change the ratio to a smaller floating point number, e.g. `"selfcite-train-2k.json:0.5"`. If you want to use `n` examples, you can change the ratio to an integer `n`, e.g. `"selfcite-train-2k.json:1000"`.


