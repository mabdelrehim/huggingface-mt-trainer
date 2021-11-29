from logging import raiseExceptions
from torch.utils.data import IterableDataset
from datasets import load_dataset
from tqdm.auto import tqdm 
import random
import numpy as np
import torch
import math

class TranslationDataset(IterableDataset):
  def __init__(self, 
                tokenized_dataset_tsv_file,
                idx_lengths_file,
                input_tokenizer, 
                target_tokenizer,  
                src_lang, 
                tgt_lang,
                max_length=512,
                max_tokens=4096, 
                max_batch_size=64,
                cache_dir=None, 
                sort=True, 
                shuffle=True):
    
    """ 
    //// TODO: docstring
      
    """
    super(TranslationDataset).__init__()

    self.input_tokenizer = input_tokenizer
    self.target_tokenizer = target_tokenizer
    self.src_lang = src_lang
    self.tgt_lang = tgt_lang
    self.sort = sort
    self.max_sequence_length = max_length
    self.max_tokens = max_tokens
    self.max_batch_size = max_batch_size
    self.shuffle = shuffle
    

    print("Loading data ...")
    self.dataset =  load_dataset(
        'csv', 
        data_files = { 
          'train': tokenized_dataset_tsv_file
          }, 
        delimiter = '\t', 
        cache_dir=cache_dir
      )
    
    self.src_lengths = {}
    self.tgt_lengths = {}
    
    with open(idx_lengths_file, 'r') as idx_f:
      for l in idx_f:
        l = l.split("\t")
        self.src_lengths[int(l[0])] = int(l[1])
        self.tgt_lengths[int(l[0])] = int(l[2]) 
  
    self.idx = list(range(0, len(self.dataset['train'])))
    if self.sort:
      self.idx = sorted(self.src_lengths, key=self.src_lengths.get, reverse=True)
    self.batches = self._generate_batches(0, len(self.idx), self.max_tokens)

  def __len__(self):
    return len(self.batches)
  
  def __iter__(self):

    """
    //// TODO: docstring 
    """
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:
      if self.shuffle:
        random.shuffle(self.batches)
      return iter(self.batches)
    else:
      num_workers = worker_info.num_workers
      worker_id = worker_info.id
      per_worker = int(math.ceil(len(self.batches) / float(num_workers)))
      worker_start = worker_id * per_worker
      worker_end = min(worker_start + per_worker, len(self.batches))
      worker_batches = self.batches[worker_start:worker_end]
      if self.shuffle:
        random.shuffle(worker_batches)
      return iter(worker_batches)


  def collate_fn(self, batch_idx):
    batch_idx=batch_idx[0]
    source_sentences = []
    target_sentences = []
    max_toks = 0
    
    for i in range(len(batch_idx)):
      
      source_sentence = self.dataset["train"][batch_idx[i]][self.src_lang]
      source_sentence = [self.input_tokenizer.convert_tokens_to_ids(token) for token in source_sentence.split(" ")]
      
      source_sentence = source_sentence[:self.max_sequence_length - 2]
      if hasattr(self.input_tokenizer, 'bos_token_id') and self.input_tokenizer.bos_token_id is not None:
        source_sentence = [self.input_tokenizer.bos_token_id] + source_sentence
      if hasattr(self.input_tokenizer, 'eos_token_id') and self.input_tokenizer.eos_token_id is not None:
        source_sentence = source_sentence + [self.input_tokenizer.eos_token_id]
      
      source_sentences.append(source_sentence)
      max_toks += len(source_sentence)

      target_sentence = self.dataset["train"][batch_idx[i]][self.tgt_lang]   
      target_sentence = [self.target_tokenizer.convert_tokens_to_ids(token) for token in target_sentence.split(" ")]
      
      target_sentence = target_sentence[:self.max_sequence_length - 2]
      if self.target_tokenizer.bos_token_id is not None:
        target_sentence = [self.target_tokenizer.bos_token_id] + target_sentence
      if self.target_tokenizer.eos_token_id is not None:
        target_sentence = target_sentence + [self.target_tokenizer.eos_token_id] 
      
      target_sentences.append(target_sentence)
      
    source_sentences = self.input_tokenizer.pad(
         {'input_ids': source_sentences}, 
         return_tensors="pt", 
         max_length=self.max_sequence_length, 
         pad_to_multiple_of=8,
         padding='max_length'
       )
 
    target_sentences = self.target_tokenizer.pad(
         {'input_ids': target_sentences}, 
         return_tensors="pt", 
         max_length=self.max_sequence_length, 
         pad_to_multiple_of=8,
         padding='max_length'
       )
    return {
      'source': source_sentences.input_ids, 
      'source_mask': source_sentences.attention_mask, 
      'target': target_sentences.input_ids, 
      'target_mask': target_sentences.attention_mask,
    }  
    
  def _generate_batches(self, start, end, max_tokens):

    accumulated_toks = 0
    current_sentence_len = 0
    previous_sentence_len = 0
    sentences_in_batch = 0
    batch_num = 0
    num_padded_toks = 0
    batch_indices = []

    batches = {}
    i = start
    with tqdm(total=end-start, desc="Generating batches") as pbar:
      while i < end:
        
        src_sentence_toks = self.dataset["train"][self.idx[i]][self.src_lang].split(" ")
        current_sentence_len = len(src_sentence_toks) + 2 ## account for bos and eos tokens
        sentences_in_batch += 1
        if (current_sentence_len >= previous_sentence_len):
          if previous_sentence_len == 0 or current_sentence_len == previous_sentence_len:
            num_padded_toks = 0
          else:
            num_padded_toks = (current_sentence_len - previous_sentence_len)*sentences_in_batch
        else:
          num_padded_toks = previous_sentence_len - current_sentence_len
      
        if (accumulated_toks + current_sentence_len + num_padded_toks > max_tokens) or (sentences_in_batch >= self.max_batch_size):
          batches[batch_num] = batch_indices
          batch_num += 1
          batch_indices = []
        
          sentences_in_batch = 0
          num_padded_toks = 0
          current_sentence_len = 0
          previous_sentence_len = 0
          accumulated_toks = 0   
          i -= 1
          sentences_in_batch -= 1
      
        else:
        
          accumulated_toks += (current_sentence_len + num_padded_toks)
          previous_sentence_len = current_sentence_len
          batch_indices.append(self.idx[i])
          i += 1
        pbar.update(1)

    return list(batches.values())


