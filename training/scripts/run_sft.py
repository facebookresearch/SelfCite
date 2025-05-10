#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import random
import sys
import os

import torch
import transformers
from transformers import AutoModelForCausalLM, set_seed

# make ../alignment accessible
# sys.path.append("../alignment")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alignment import (
    DataArguments,
    DPOConfig,
    H4ArgumentParser,
    ModelArguments,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
    is_adapter_model,
    __file__,
)
print(f"alignment.__file__={__file__}")
from alignment.data import maybe_insert_system_message, is_openai_format
from peft import PeftConfig, PeftModel
from sft_trainer import SFTTrainer
# from simpo_trainer_liger import SimPOTrainer
from simpo_config import SimPOConfig
from dataclasses import dataclass, field
from typing import Optional, Literal
import datetime

import accelerate

# Save original __init__ so we can wrap it
_orig_init = accelerate.InitProcessGroupKwargs.__init__

def _patched_init(
    self,
    backend: str = "nccl",
    init_method: str = None,
    timeout: datetime.timedelta = None,
):
    """
    Patched __init__ for InitProcessGroupKwargs that:
      - sets a default backend of 'nccl'
      - sets a larger default timeout if no timeout is explicitly provided
    """
    if timeout is None:
        # Example: set to 2 hours, adjust as needed
        timeout = datetime.timedelta(seconds=7200)
    return _orig_init(self, backend=backend, init_method=init_method, timeout=timeout)

# Apply the monkey patch
accelerate.InitProcessGroupKwargs.__init__ = _patched_init

logger = logging.getLogger(__name__)

def redirect_output_to_file(log_file_path):
    # check if the output directory exists
    if not os.path.exists(os.path.dirname(log_file_path)):
        os.makedirs(os.path.dirname(log_file_path))
    log_file = open(log_file_path, 'w')  # Open file in write mode
    sys.stdout = log_file  # Redirect stdout to the file
    sys.stderr = log_file  # Redirect stderr to the file

MISTRAL_CHAT_TEMPLATE = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'].strip() + '\n\n' %}{% else %}{% set loop_messages = messages %}{% set system_message = '' %}{% endif %}{% for message in loop_messages %}{% if loop.index0 == 0 %}{% set content = system_message + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"

def apply_chat_template(
    example,
    tokenizer,
    task: Literal["sft", "generation", "rm", "simpo"],
    auto_insert_empty_system_msg: bool = True,
    change_template = None,
    llama_chat_template = False,
):
    if change_template == "mistral":
        tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE
    if task in ["sft", "generation"]:
        if llama_chat_template:
            messages = [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["chosen"]},
            ]
            full_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            if not "<|start_header_id|>assistant<|end_header_id|>\n\n" in full_text:
                print(f"Warning: <|start_header_id|>assistant<|end_header_id|> not found in full_text: {full_text}")
            text_prompt, text_chosen = full_text.split("<|start_header_id|>assistant<|end_header_id|>\n\n")
            text_prompt = text_prompt + "<|start_header_id|>assistant<|end_header_id|>\n\n"
            example["text_prompt"] = text_prompt
            example["text_chosen"] = text_chosen
        else:
            example["text_prompt"] = "<|user|>\n" + example["prompt"] + "<|assistant|>\n"
            example["text_chosen"] = example["chosen"].strip()
    elif task == "simpo":
        if llama_chat_template:
            # chosen
            messages = [
                {"role": "user", "content": example["prompt"].strip()},
                {"role": "assistant", "content": example["chosen"].strip()},
            ]
            full_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            if not "<|start_header_id|>assistant<|end_header_id|>\n\n" in full_text:
                print(f"Warning: <|start_header_id|>assistant<|end_header_id|> not found in full_text: {full_text}")
            text_prompt, text_chosen = full_text.split("<|start_header_id|>assistant<|end_header_id|>\n\n")
            text_prompt = text_prompt + "<|start_header_id|>assistant<|end_header_id|>\n\n"

            # reject
            messages = [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["rejected"]},
            ]
            full_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            if not "<|start_header_id|>assistant<|end_header_id|>\n\n" in full_text:
                print(f"Warning: <|start_header_id|>assistant<|end_header_id|> not found in full_text: {full_text}")
            text_prompt, text_rejected = full_text.split("<|start_header_id|>assistant<|end_header_id|>\n\n")
            example["text_prompt"] = text_prompt
            example["text_chosen"] = text_chosen
            example["text_rejected"] = text_rejected
        else:
            example["text_prompt"] = "<|user|>\n" + example["prompt"] + "<|assistant|>\n"
            example["text_chosen"] =  example["chosen"].strip()
            example["text_rejected"] = example["rejected"].strip()
    else:
        raise ValueError(
            f"Task {task} not supported, please ensure that the provided task is one of ['sft', 'generation', 'rm', 'dpo', 'orpo']"
        )
    return example


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, SimPOConfig))
    model_args, data_args, training_args = parser.parse()

    global_rank = -1
    for v in ('RANK', 'PMI_RANK', 'SLURM_PROCID', 'OMPI_COMM_WORLD_RANK'):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break

    if global_rank != -1:
        redirect_output_to_file(f"{training_args.output_dir}/output.{global_rank}.log")

    device_string = global_rank % torch.cuda.device_count()
    
    print(f"global_rank: {global_rank}, device_string: {device_string}")

    #######
    # Setup
    #######
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Load datasets
    ###############
    raw_datasets = get_datasets(
        data_args,
        splits=data_args.dataset_splits,
        configs=data_args.dataset_configs,
        columns_to_keep=["messages", "chosen", "rejected", "reject", "prompt", "completion", "label"],
    )
    logger.info(
        f"Training on the following splits: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)

    #####################################
    # Load tokenizer and process datasets
    #####################################
    data_args.truncation_side = "left"  # Truncate from left to ensure we don't lose labels in final turn
    tokenizer = get_tokenizer(model_args, data_args)

    if "mistral" in model_args.model_name_or_path.lower():
        change_template = "mistral"
    else:
        change_template = None
    #####################
    # Apply chat template
    #####################
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": "sft",
            "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
            "change_template": change_template,
            "llama_chat_template": data_args.llama_chat_template,
        },
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Formatting comparisons with prompt template",
    )

    # Replace column names with what TRL needs, text_chosen -> chosen and text_rejected -> rejected
    for split in ["train", "test"]:
        if split in raw_datasets and raw_datasets[split] is not None:
            raw_datasets[split] = raw_datasets[split].rename_columns(
                {"text_prompt": "prompt", "text_chosen": "chosen"}
            )

    # Log a few random samples from the training set:
    for index in random.sample(range(len(raw_datasets["train"])), 3):
        logger.info(f"Prompt sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['prompt']}")
        logger.info(f"Chosen sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['chosen']}")
        # logger.info(f"Rejected sample {index} of the raw training set:\n\n{raw_datasets['train'][index]['rejected']}")

    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        attn_implementation=model_args.attn_implementation,
    )

    model = model_args.model_name_or_path

    training_args.model_init_kwargs = model_kwargs
    #########################
    # Instantiate SimPO trainer
    #########################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["test"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    ###############
    # Training loop
    ###############
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(raw_datasets["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    logger.info("*** Training complete ***")

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": list(data_args.dataset_mixer.keys()) if type(data_args.dataset_mixer) == dict else [data_args.dataset_mixer],
        "dataset_tags": list(data_args.dataset_mixer.keys()) if type(data_args.dataset_mixer) == dict else [data_args.dataset_mixer],
        "tags": ["alignment-handbook"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(raw_datasets["test"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

    logger.info("*** Training complete! ***")


if __name__ == "__main__":
    main()
