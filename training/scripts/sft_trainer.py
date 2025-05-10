# Copyright (c) Meta Platforms, Inc. and affiliates.

# This code is based on https://github.com/princeton-nlp/SimPO/blob/main/scripts/simpo_trainer.py

import inspect
import random
import warnings
from collections import defaultdict
from contextlib import nullcontext
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollator, PreTrainedModel, PreTrainedTokenizerBase, Trainer
from liger_kernel.transformers import AutoLigerKernelForCausalLM as AutoModelForCausalLM
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from transformers.utils import is_torch_fx_proxy

from trl.import_utils import is_peft_available, is_wandb_available
from simpo_config import SimPOConfig  # You can rename or replace this config as needed.

from dataclasses import dataclass
from typing import Dict, Literal, Optional

from transformers import TrainingArguments

from trl.trainer.utils import (
    DPODataCollatorWithPadding,
    disable_dropout_in_model,
    pad_to_length,
    peft_module_casting_to_bf16,
    trl_sanitze_kwargs_for_tagging,
)

if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

if is_wandb_available():
    import wandb


class SFTTrainer(Trainer):
    r"""
    A minimally modified version of a trainer for supervised fine-tuning (SFT).

    This trainer:
      - Uses only the "chosen" responses to compute a standard language-modeling (LM) loss.
      - Ignores the prompt tokens by masking them out with `label_pad_token_id`.
      - If you pass a `peft_config`, the model will be wrapped in a PEFT model.
    """

    _tag_names = ["trl", "sft"]

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: Optional[SimPOConfig] = None,  # or a custom config
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional[Dict] = None,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
    ):
        # Handle model init kwargs
        if args.model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError("You passed model_kwargs to the SFTTrainer, but your model is already instantiated.")
        else:
            model_init_kwargs = args.model_init_kwargs
            model_init_kwargs["torch_dtype"] = (
                model_init_kwargs["torch_dtype"]
                if model_init_kwargs["torch_dtype"] in ["auto", None]
                else getattr(torch, model_init_kwargs["torch_dtype"])
            )

        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id to the SFTTrainer. This will automatically create an "
                "`AutoModelForCausalLM` or a `PeftModel` (if you passed a `peft_config`) for you."
            )
            model_init_kwargs["trust_remote_code"] = True
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        # Initialize this variable to False. This helps tracking the case when `peft_module_casting_to_bf16`
        # has been called in order to properly call autocast if needed.
        self._peft_has_been_casted_to_bf16 = False

        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use PEFT."
            )
        elif is_peft_available() and peft_config is not None:
            # if model is a peft model and we have a peft_config, we merge and unload it first
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()

            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                _support_gc_kwargs = hasattr(
                    args, "gradient_checkpointing_kwargs"
                ) and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )

                prepare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

                if _support_gc_kwargs:
                    prepare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

                model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)
            elif getattr(args, "gradient_checkpointing", False):
                # For backward compatibility with older versions of transformers
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:

                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)

                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            model = get_peft_model(model, peft_config)
            if args.bf16 and getattr(model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(model)
                self._peft_has_been_casted_to_bf16 = True

        elif getattr(args, "gradient_checkpointing", False):
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if args.generate_during_eval and not is_wandb_available():
            raise ValueError(
                "`generate_during_eval=True` requires Weights and Biases to be installed."
                " Please install `wandb` to resolve."
            )

        if model is not None:
            self.is_encoder_decoder = model.config.is_encoder_decoder
        elif args.is_encoder_decoder is None:
            raise ValueError("When no model is provided, you need to pass the parameter is_encoder_decoder.")
        else:
            self.is_encoder_decoder = args.is_encoder_decoder

        if self.is_encoder_decoder:
            self.decoder_start_token_id = model.config.decoder_start_token_id
            self.pad_token_id = model.config.pad_token_id

        if tokenizer is None:
            raise ValueError("tokenizer must be specified for SFTTrainer.")
        if args.max_length is None:
            warnings.warn(
                "`max_length` is not set. It will default to `512`, but you should set it explicitly.",
                UserWarning,
            )
            max_length = 512
        else:
            max_length = args.max_length
        if args.max_prompt_length is None:
            warnings.warn(
                "`max_prompt_length` is not set. It will default to `128`, but you should set it explicitly.",
                UserWarning,
            )
            max_prompt_length = 128
        else:
            max_prompt_length = args.max_prompt_length

        if args.max_target_length is None and self.is_encoder_decoder:
            warnings.warn(
                "When using an encoder-decoder architecture, `max_target_length` should be set. "
                "It will default to `128`.",
                UserWarning,
            )
            max_target_length = 128
        else:
            max_target_length = args.max_target_length

        # By default, reuse DPODataCollator for SFT if none is specified
        if data_collator is None:
            data_collator = DPODataCollatorWithPadding(
                pad_token_id=tokenizer.pad_token_id,
                label_pad_token_id=args.label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False`. "
                    "We've done this for you.",
                    UserWarning,
                )
            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        if args.disable_dropout:
            disable_dropout_in_model(model)

        self.max_length = max_length
        self.generate_during_eval = args.generate_during_eval
        self.label_pad_token_id = args.label_pad_token_id
        self.padding_value = args.padding_value if args.padding_value is not None else tokenizer.pad_token_id
        self.max_prompt_length = max_prompt_length
        self.truncation_mode = args.truncation_mode
        self.max_target_length = max_target_length
        self.tokenizer = tokenizer

        # For backward compat

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # Tokenize the dataset
        with PartialState().local_main_process_first():
            train_dataset = train_dataset.map(self.tokenize_row, num_proc=args.dataset_num_proc) if train_dataset else None
            if eval_dataset is not None:
                if isinstance(eval_dataset, dict):
                    for k, dataset in eval_dataset.items():
                        eval_dataset[k] = dataset.map(self.tokenize_row, num_proc=args.dataset_num_proc)
                else:
                    eval_dataset = eval_dataset.map(self.tokenize_row, num_proc=args.dataset_num_proc)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Add tags if the model supports it
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

    def build_tokenized_answer(self, prompt, answer):
        """
        Handle the special token merging that can happen with some tokenizers (e.g., Llama).
        We'll still mask out the prompt tokens with label_pad_token_id later.
        """
        full_tokenized_input_ids = self.tokenizer.encode(prompt + answer, add_special_tokens=False)
        full_tokenized_attention_mask = [1] * len(full_tokenized_input_ids)
        full_tokenized = dict(input_ids=full_tokenized_input_ids, attention_mask=full_tokenized_attention_mask)
        prompt_input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)

        answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids) :]
        answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids) :]

        full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])
        full_input_ids = np.array(full_tokenized["input_ids"])

        if len(full_input_ids) != len(full_concat_input_ids):
            raise ValueError("Prompt input ids and answer input ids should have the same length.")

        response_token_ids_start_idx = len(prompt_input_ids)
        if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
            response_token_ids_start_idx -= 1

        prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
        prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]

        answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
        answer_attention_mask = full_tokenized["attention_mask"][response_token_ids_start_idx:]

        return dict(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            input_ids=answer_input_ids,
            attention_mask=answer_attention_mask,
        )

    def tokenize_row(self, feature, model: Optional[Union[PreTrainedModel, nn.Module]] = None) -> Dict:
        """
        Tokenize a single row for SFT training:
          - We create labels for the "chosen" response, masking out prompt tokens with label_pad_token_id.
        """
        batch = {}
        prompt = feature["prompt"]
        chosen = feature["chosen"]

        if not self.is_encoder_decoder:
            if not isinstance(prompt, str):
                raise ValueError(f"prompt should be an str but got {type(prompt)}")

            prompt_tokens_input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            prompt_tokens_attention_mask = [1] * len(prompt_tokens_input_ids)
            prompt_tokens = dict(input_ids=prompt_tokens_input_ids, attention_mask=prompt_tokens_attention_mask)
            prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

            if not isinstance(chosen, str):
                raise ValueError(f"chosen should be an str but got {type(chosen)}")
            chosen_tokens = self.build_tokenized_answer(prompt, chosen)

            # Potentially merged tokens in the last position
            prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])

            for k, v in prompt_tokens.items():
                prompt_tokens[k] = v[:prompt_len_input_ids]

            # Add BOS if needed
            bos_token_id = self.tokenizer.bos_token_id
            if bos_token_id is not None:
                if prompt_len_input_ids == 0 or bos_token_id != prompt_tokens["prompt_input_ids"][0]:
                    prompt_tokens["prompt_input_ids"] = [bos_token_id] + prompt_tokens["prompt_input_ids"]
                    prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
                if prompt_len_input_ids == 0 or bos_token_id != chosen_tokens["prompt_input_ids"][0]:
                    chosen_tokens["prompt_input_ids"] = [bos_token_id] + chosen_tokens["prompt_input_ids"]
                    chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens["prompt_attention_mask"]

            # Add EOS if needed
            eos_token_id = self.tokenizer.eos_token_id
            if len(chosen_tokens["input_ids"]) == 0 or eos_token_id != chosen_tokens["input_ids"][-1]:
                chosen_tokens["input_ids"].append(eos_token_id)
                chosen_tokens["attention_mask"].append(1)

            # Truncation
            longer_response_length = len(chosen_tokens["input_ids"])
            for answer_tokens in [chosen_tokens, prompt_tokens]:
                if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                    if self.truncation_mode == "keep_start":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][: self.max_prompt_length]
                    elif self.truncation_mode == "keep_end":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][-self.max_prompt_length :]
                    else:
                        raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")
            if len(chosen_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                for k in ["input_ids", "attention_mask"]:
                    chosen_tokens[k] = chosen_tokens[k][: self.max_length - self.max_prompt_length]

            # Create final sequence (prompt + chosen)
            chosen_sequence_tokens = {
                k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k] for k in ["input_ids", "attention_mask"]
            }
            # Mask out the prompt
            chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
            chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [
                self.label_pad_token_id
            ] * len(chosen_tokens["prompt_input_ids"])
            # chosen_sequence_tokens["labels"] = chosen_sequence_tokens["labels"][1:] + [self.label_pad_token_id]

            for k, toks in {
                "chosen_": chosen_sequence_tokens,
                "": prompt_tokens,  # we keep the prompt in the batch if needed
            }.items():
                for type_key, tokens in toks.items():
                    if type_key == "token_type_ids":
                        continue
                    batch[f"{k}{type_key}"] = tokens

        else:
            # Encoder-decoder case
            chosen_tokens = self.tokenizer(
                chosen, truncation=True, max_length=self.max_target_length, add_special_tokens=True
            )
            prompt_tokens = self.tokenizer(
                prompt, truncation=True, max_length=self.max_prompt_length, add_special_tokens=True
            )

            batch["chosen_labels"] = torch.cat(
                [chosen_tokens["input_ids"][..., 1:].clone(),
                 torch.tensor([self.tokenizer.eos_token_id], device=chosen_tokens["input_ids"].device).unsqueeze(0)],
                dim=-1,
            )
            batch["prompt_input_ids"] = prompt_tokens["input_ids"]
            batch["prompt_attention_mask"] = prompt_tokens["attention_mask"]

            if model is not None and hasattr(model, "prepare_decoder_input_ids_from_labels"):
                batch["chosen_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                    labels=torch.tensor(batch["chosen_labels"])
                )

        return batch

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """
        Compute the standard cross-entropy SFT loss **only** on the 'chosen' tokens (prompt tokens are masked).
        """
        prefix = "eval_" if train_eval == "eval" else ""

        # Prepare chosen inputs
        chosen_input_ids = batch["chosen_input_ids"]
        chosen_attention_mask = batch["chosen_attention_mask"]
        chosen_labels = batch["chosen_labels"]

        model_kwargs = {}
        if self.is_encoder_decoder:
            model_kwargs["decoder_input_ids"] = batch.get("chosen_decoder_input_ids", None)
            model_kwargs["labels"] = chosen_labels
            outputs = model(
                batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                **model_kwargs,
            )
            loss = outputs.loss
        else:
            # Standard causal LM forward
            outputs = model(
                chosen_input_ids, attention_mask=chosen_attention_mask, use_cache=False, labels=chosen_labels
            )
            loss = outputs.loss

        final_loss = loss

        metrics = {}
        metrics[f"{prefix}loss"] = final_loss.detach().cpu()
        return final_loss, metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only guaranteed to work as intended with DPODataCollatorWithPadding. "
                "If you're using a different collator, please ensure it provides the same keys."
            )

        compute_loss_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with compute_loss_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        # Force log the metrics
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    # def get_batch_samples(self, model, batch: Dict[str, torch.LongTensor]) -> List[str]:
    #     """
    #     Generate samples from the model for the given batch of inputs, primarily for logging.
    #     """
    #     generate_context_manager = nullcontext if not self._peft_has_been_casted_to_bf16 else torch.cuda.amp.autocast

    #     with generate_context_manager():
    #         policy_output = model.generate(
    #             input_ids=batch["prompt_input_ids"],
    #             attention_mask=batch["prompt_attention_mask"],
    #             max_length=self.max_length,
    #             do_sample=True,
    #             pad_token_id=self.tokenizer.pad_token_id,
    #         )

    #     policy_output = pad_to_length(policy_output, self.max_length, self.tokenizer.pad_token_id)
    #     policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

    #     return policy_output_decoded

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if not self.use_dpo_data_collator:
            warnings.warn(
                "prediction_step is only guaranteed to work as intended with DPODataCollatorWithPadding."
            )
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        prediction_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with torch.no_grad(), prediction_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")

        # Force log the metrics
        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        # For classification-like logging, we simply put a dummy 0 label
        logits = torch.zeros((1,), device=self.accelerator.device)
        labels = torch.zeros_like(logits)
        return (loss.detach(), logits, labels)

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Overriding built-in evaluation loop to optionally generate samples for logging.
        """

        # Sample and save to game log if requested (for one batch to save time)
        if self.generate_during_eval and hasattr(dataloader.dataset, "select"):
            num_samples = len(dataloader.dataset)
            random_indices = random.sample(range(num_samples), k=self.args.eval_batch_size)
            random_batch_dataset = dataloader.dataset.select(random_indices)
            random_batch = self.data_collator(random_batch_dataset)
            random_batch = self._prepare_inputs(random_batch)

            # policy_output_decoded = self.get_batch_samples(self.model, random_batch)

            # # Log: each row is [prompt, generated_output_minus_prompt]
            # if is_wandb_available():
            #     self.log(
            #         {
            #             "game_log": wandb.Table(
            #                 columns=["Prompt", "Policy"],
            #                 rows=[
            #                     [pr, pol[len(pr) :]]
            #                     for pr, pol in zip(random_batch["prompt"], policy_output_decoded)
            #                 ],
            #             )
            #         }
            #     )
            #     self.state.log_history.pop()

        initial_output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )

        return initial_output

    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.
        """
        train_eval = "train" if "loss" in logs else "eval"
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)

    @wraps(Trainer.push_to_hub)
    def push_to_hub(self, commit_message: Optional[str] = "End of training", blocking: bool = True, **kwargs) -> str:
        """
        Overwrite the `push_to_hub` method in order to force-add the tag "sft" when pushing the
        model to the Hub.
        """
        kwargs = trl_sanitze_kwargs_for_tagging(model=self.model, tag_names=self._tag_names, kwargs=kwargs)
        return super().push_to_hub(commit_message=commit_message, blocking=blocking, **kwargs)