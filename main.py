#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
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
"""
Fine-tuning a 🤗 Transformers model on text translation.
"""
# You can also adapt this script on your own text translation task. Pointers for this are left as comments.

import argparse
import logging
import math
import os
from pathlib import Path

import datasets
import numpy as np
import torch
from datasets import load_metric
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from huggingface_hub import Repository
from transformers import (
  MODEL_MAPPING,
  AdamW,
  EncoderDecoderModel,
  AutoTokenizer,
  SchedulerType,
  get_scheduler,
  set_seed,
)
from transformers.file_utils import get_full_repo_name
from transformers.utils.versions import require_version
from data.translation_dataset import TranslationDataset

import numpy as np


logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/translation/requirements.txt")

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


# Parsing input arguments
def parse_args():

  parser = argparse.ArgumentParser(description="Finetune an encoder-decoder model from public pretrained weights")
  
  
  # dataset related arguments
  parser.add_argument(
    "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
  )
  parser.add_argument(
    "--train_idx_file", type=str, default=None, help="A csv or a json file containing metadata for each sentence of the training data."
  )
  parser.add_argument(
    "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
  )
  parser.add_argument(
    "--validation_idx_file", type=str, default=None, help="A csv or a json file containing metadata for each sentence of the validation data."
  )
  parser.add_argument(
    "--max_sequence_length",
    type=int,
    default=512,
    help="The maximum total input sequence length after "
    "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded."
  )
  parser.add_argument(
    "--max_batch_size",
    type=int,
    default=8,
    help="The maximum total sentence pairs in a batch per gpu"
  )
  parser.add_argument(
    "--max_tokens",
    type=int,
    default=1024,
    help="The maximum total tokens in a batch per gpu"
  )
  
  parser.add_argument("--source_lang", type=str, default=None, help="Source language id for translation.")
  parser.add_argument("--target_lang", type=str, default=None, help="Target language id for translation.")
  parser.add_argument(
    "--preprocessing_num_workers",
    type=int,
    default=4,
    help="The number of processes to use for the preprocessing.",
  )
  parser.add_argument(
    "--cache_dir", type=str, default=None, help="Huggingface cache directory to use for datasets"
  )
  
  
  
  parser.add_argument(
    "--predict_with_generate",
    type=bool,
    default=True,
    help="",
  )
  parser.add_argument(
    "--num_beams",
    type=int,
    default=None,
    help="Number of beams to use for evaluation. This argument will be "
    "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
  )

  parser.add_argument(
    "--encoder_model_name",
    type=str,
    help="Path to pretrained model or model identifier from huggingface.co/models.",
    required=True,
  )

  parser.add_argument(
    "--decoder_model_name",
    type=str,
    help="Path to pretrained model or model identifier from huggingface.co/models.",
    required=True,
  )

  parser.add_argument(
    "--input_tokenizer_name",
    type=str,
    default=None,
    help="Pretrained tokenizer name or path if not the same as encoder_model_name",
  )
  parser.add_argument(
    "--output_tokenizer_name",
    type=str,
    default=None,
    help="Pretrained tokenizer name or path if not the same as encoder_model_name",
  )
  parser.add_argument(
    "--use_slow_tokenizer",
    action="store_true",
    help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
  )
  
  parser.add_argument(
    "--learning_rate",
    type=float,
    default=5e-4,
    help="Initial learning rate (after the potential warmup period) to use.",
  )
  parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
  parser.add_argument("--num_train_epochs", type=int, default=5, help="Total number of training epochs to perform.")
  parser.add_argument(
    "--max_train_steps",
    type=int,
    default=None,
    help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
  )
  parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
  )
  parser.add_argument(
    "--lr_scheduler_type",
    type=SchedulerType,
    default="linear",
    help="The scheduler type to use.",
    choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
  )
  parser.add_argument(
    "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
  )
  parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
  parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
  
  parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
  parser.add_argument(
    "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
  )
  parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
  args = parser.parse_args()
  

  return args

def main():
  # Parse the arguments
  args = parse_args()

  # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
  accelerator = Accelerator()

  # Make one log on every process with the configuration for debugging.
  logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
  )
  logger.info(accelerator.state)

  # Setup logging, we only want one process per machine to log things on the screen.
  # accelerator.is_local_main_process is only True for one process per machine.
  logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
  if accelerator.is_local_main_process:
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()
  else:
    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()

  # If passed along, set the training seed now.
  if args.seed is not None:
    set_seed(args.seed)

  # Handle the repository creation
  if accelerator.is_main_process:
    if args.push_to_hub:
      if args.hub_model_id is None:
        repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
      else:
        repo_name = args.hub_model_id
      repo = Repository(args.output_dir, clone_from=repo_name)
    elif args.output_dir is not None:
      os.makedirs(args.output_dir, exist_ok=True)
  accelerator.wait_for_everyone()

  # load encoder-decoder model and their respective tokenizers
  model = EncoderDecoderModel.from_encoder_decoder_pretrained(
    args.encoder_model_name, 
    args.decoder_model_name, 
  )
  for param in model.encoder.parameters():
    param.requires_grad = False

  input_tokenizer = AutoTokenizer.from_pretrained(args.input_tokenizer_name, use_fast=not args.use_slow_tokenizer)
  output_tokenizer = AutoTokenizer.from_pretrained(args.output_tokenizer_name, use_fast=not args.use_slow_tokenizer)


  train_dataset = TranslationDataset(
    args.train_file, 
    args.train_idx_file, 
    input_tokenizer, 
    output_tokenizer, 
    cache_dir=args.cache_dir,
    max_length = args.max_sequence_length, 
    max_tokens = args.max_tokens,
    max_batch_size = args.max_batch_size,
    src_lang = args.source_lang, 
    tgt_lang = args.target_lang
  )
  eval_dataset = TranslationDataset(
    args.validation_file, 
    args.validation_idx_file, 
    input_tokenizer, 
    output_tokenizer, 
    cache_dir=args.cache_dir,
    max_length = args.max_sequence_length, 
    max_tokens = args.max_tokens,
    max_batch_size = args.max_batch_size,
    src_lang = args.source_lang, 
    tgt_lang = args.target_lang
  )

  train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset, 
    num_workers=args.preprocessing_num_workers, 
    batch_size=1, 
    collate_fn=train_dataset.collate_fn, 
    pin_memory=True
  )
  eval_dataloader = torch.utils.data.DataLoader(
    dataset=eval_dataset, 
    num_workers=args.preprocessing_num_workers, 
    batch_size=1, 
    collate_fn=eval_dataset.collate_fn, 
    pin_memory=True
  )

  # Optimizer
  # Split weights in two groups, one with weight decay and the other not.
  no_decay = ["bias", "LayerNorm.weight"]
  optimizer_grouped_parameters = [
      {
          "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
          "weight_decay": args.weight_decay,
      },
      {
          "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
          "weight_decay": 0.0,
      },
  ]
  optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

  # Prepare everything with our `accelerator`.
  model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
      model, optimizer, train_dataloader, eval_dataloader
  )

  # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
  # shorter in multiprocess)

  # Scheduler and math around the number of training steps.
  num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
  if args.max_train_steps is None:
      args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
  else:
      args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

  lr_scheduler = get_scheduler(
      name=args.lr_scheduler_type,
      optimizer=optimizer,
      num_warmup_steps=args.num_warmup_steps,
      num_training_steps=args.max_train_steps,
  )

  metric = load_metric("sacrebleu")

  def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

  # Train!
  effective_max_tokens = args.max_tokens * accelerator.num_processes * args.gradient_accumulation_steps

  logger.info("***** Running training *****")
  logger.info(f"  Num examples = {len(train_dataset)}")
  logger.info(f"  Num Epochs = {args.num_train_epochs}")
  logger.info(f"  Max tokens per device = {args.max_tokens}")
  logger.info(f"  Total train effective max tokens per batch (w. parallel, distributed & accumulation) = {effective_max_tokens}")
  logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
  logger.info(f"  Total optimization steps = {args.max_train_steps}")
  # Only show the progress bar once on each machine.
  progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
  completed_steps = 0

  accelerator.print(model)

  for epoch in range(args.num_train_epochs):
    model.train()
    for step, batch in enumerate(train_dataloader):
      
      source_tokens = batch['source']
      target_tokens = batch['target']
      source_masks = batch['source_mask']
      target_masks = batch['target_mask']
      lm_labels = target_masks.clone()
      outputs = model(
        input_ids=source_tokens, 
        attention_mask=source_masks,
        decoder_input_ids=target_tokens, 
        decoder_attention_mask=target_masks,
        labels=lm_labels
      )
      
      loss = outputs.loss
      loss = loss / args.gradient_accumulation_steps
      accelerator.backward(loss)
      if accelerator.is_main_process and completed_steps % 100 == 0:
        logger.info(f"Training steps {completed_steps}/{args.max_train_steps}, Training ppl: {np.exp(loss.item())}")
      
      if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        completed_steps += 1

      if completed_steps >= args.max_train_steps:
        break

    model.eval()


    gen_kwargs = {
      "max_length": args.max_sequence_length,
      "num_beams": args.num_beams,
    }
    for step, batch in enumerate(eval_dataloader):
      with torch.no_grad():
        generated_tokens = accelerator.unwrap_model(model).generate(
          batch["source"],
          attention_mask=batch["source_mask"],
          **gen_kwargs,
        )

        generated_tokens = accelerator.pad_across_processes(
          generated_tokens, dim=1, pad_index=output_tokenizer.pad_token_id
        )
        labels = batch['target'].clone()

        generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
        labels = accelerator.gather(labels).cpu().numpy()

        decoded_preds = output_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        decoded_labels = output_tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        metric.add_batch(predictions=decoded_preds, references=decoded_labels)
    eval_metric = metric.compute()
    logger.info({"bleu": eval_metric["score"]})

    #if args.push_to_hub and epoch < args.num_train_epochs - 1:
    #    accelerator.wait_for_everyone()
    #    unwrapped_model = accelerator.unwrap_model(model)
    #    unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
    #    if accelerator.is_main_process:
    #        tokenizer.save_pretrained(args.output_dir)
    #        repo.push_to_hub(
    #            commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
    #        )

  if args.output_dir is not None:
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
      input_tokenizer.save_pretrained(args.output_dir)
      output_tokenizer.save_pretrained(args.output_dir)
      if args.push_to_hub:
        repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)


if __name__ == "__main__":
  main()

