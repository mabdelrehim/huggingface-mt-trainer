import argparse
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
"""
A script to tokenize dataset and generate idx file for the dataset with lengths of sentences after tokenization
"""

def get_args_parser():
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument("--data_path", 
                          type=str, 
                          help="dataset file path (e.g. /path/to/train.tsv)")
  parser.add_argument("--cache_dir", 
                          type=str,
                          default=None, 
                          help="the directory used to cache the dataset (e.g. /path/to/cache_dir)")
  parser.add_argument("--src_lang", 
                            type=str, 
                            help='source language code (e.g. "ar")')
  parser.add_argument("--tgt_lang", 
                            type=str, 
                            help='target language code (e.g. "fr")')
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
  
  return parser

def tokenize_dataset(args, data_file, src_tokenizer, tgt_tokenizer, src_lang, tgt_lang, cache_dir=None):
  raw_data = load_dataset(
      'csv', 
      data_files = { 
        'train': data_file
        }, 
      delimiter = '\t', 
      cache_dir=cache_dir
    )
  tok_file_name = data_file.replace(".tsv", ".c.tok.tsv")
  idx_file_name = data_file.replace(".tsv", ".c.idx")  

  with open(idx_file_name, 'w') as idx_file, open(tok_file_name, 'w') as tok_file:
    tok_file.write("id" + "\t" + src_lang + "\t" + tgt_lang + "\n")
    for i in tqdm(range(len(raw_data["train"])), desc="Tokenizing"):
      src = raw_data["train"][i][src_lang]
      tgt = raw_data["train"][i][tgt_lang]
      pair_id = raw_data["train"][i]["id"] 
      src_tokenized = src_tokenizer.tokenize(src)
      src_length = len(src_tokenized)
      src_tokenized = " ".join(src_tokenized)
      tgt_tokenized = tgt_tokenizer.tokenize(tgt)
      tgt_length = len(tgt_tokenized)
      tgt_tokenized = " ".join(tgt_tokenized)
      tok_file.write(str(pair_id) + "\t" + src_tokenized.strip() + "\t" + tgt_tokenized.strip() + "\n")
      idx_file.write(str(pair_id) + "\t" + str(src_length) + "\t" + str(tgt_length) + "\n")
  return tok_file_name, idx_file_name

def main(args):
  input_tokenizer = AutoTokenizer.from_pretrained(args.input_tokenizer_name, use_fast=True)
  output_tokenizer = AutoTokenizer.from_pretrained(args.output_tokenizer_name, use_fast=True)

  tokenize_dataset(
    args,
    args.data_path, 
    input_tokenizer, 
    output_tokenizer,
    args.src_lang, 
    args.tgt_lang, 
    args.cache_dir
  )

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    'A script to tokenize dataset and generate idx file for the dataset with lengths of sentences after tokenization',
    parents=[get_args_parser()]
  )
  args = parser.parse_args()
    
  main(args)