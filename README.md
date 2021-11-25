# Huggingface Pretrained Encoder-Decoder Trainer

This is a script used to train a huggingface encoder-decoder model where both the encoder and the decoder are initialized from publicly pretrained BERT/RoBERTA weights.

For this specific experiment, the encoder is initialized from xlm-roberta-base and the decoder is initialized from camembert-base (French BERT). The encoder part is freezed and the decoder and crossattention part (initialized randomly) are finetuned on parallel Arabic-French data.

The command used to overfit the model to a small data can be found in submit.sh

```
$ python main.py -h

##usage: main.py [-h] [--train_file TRAIN_FILE] [--train_idx_file TRAIN_IDX_FILE] [--validation_file VALIDATION_FILE] [--validation_idx_file VALIDATION_IDX_FILE]
##               [--max_sequence_length MAX_SEQUENCE_LENGTH] [--max_batch_size MAX_BATCH_SIZE] [--max_tokens MAX_TOKENS] [--source_lang SOURCE_LANG] [--target_lang TARGET_LANG]
##               [--preprocessing_num_workers PREPROCESSING_NUM_WORKERS] [--cache_dir CACHE_DIR] [--predict_with_generate PREDICT_WITH_GENERATE] [--num_beams NUM_BEAMS]
##               --encoder_model_name ENCODER_MODEL_NAME --decoder_model_name DECODER_MODEL_NAME [--input_tokenizer_name INPUT_TOKENIZER_NAME]
##               [--output_tokenizer_name OUTPUT_TOKENIZER_NAME] [--use_slow_tokenizer] [--learning_rate LEARNING_RATE] [--weight_decay WEIGHT_DECAY]
##               [--num_train_epochs NUM_TRAIN_EPOCHS] [--max_train_steps MAX_TRAIN_STEPS] [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
##               [--lr_scheduler_type {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}] [--num_warmup_steps NUM_WARMUP_STEPS]
##               [--output_dir OUTPUT_DIR] [--seed SEED] [--push_to_hub] [--hub_model_id HUB_MODEL_ID] [--hub_token HUB_TOKEN]
##
##Finetune an encoder-decoder model from public pretrained weights
##
##optional arguments:
##  -h, --help            show this help message and exit
##  --train_file TRAIN_FILE
##                        A csv or a json file containing the training data.
##  --train_idx_file TRAIN_IDX_FILE
##                        A csv or a json file containing metadata for each sentence of the training data.
##  --validation_file VALIDATION_FILE
##                        A csv or a json file containing the validation data.
##  --validation_idx_file VALIDATION_IDX_FILE
##                        A csv or a json file containing metadata for each sentence of the validation data.
##  --max_sequence_length MAX_SEQUENCE_LENGTH
##                        The maximum total input sequence length after tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.
##  --max_batch_size MAX_BATCH_SIZE
##                        The maximum total sentence pairs in a batch per gpu
##  --max_tokens MAX_TOKENS
##                        The maximum total tokens in a batch per gpu
##  --source_lang SOURCE_LANG
##                        Source language id for translation.
##  --target_lang TARGET_LANG
##                        Target language id for translation.
##  --preprocessing_num_workers PREPROCESSING_NUM_WORKERS
##                        The number of processes to use for the preprocessing.
##  --cache_dir CACHE_DIR
##                        Huggingface cache directory to use for datasets
##  --predict_with_generate PREDICT_WITH_GENERATE
##  --num_beams NUM_BEAMS
##                        Number of beams to use for evaluation. This argument will be passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.
##  --encoder_model_name ENCODER_MODEL_NAME
##                        Path to pretrained model or model identifier from huggingface.co/models.
##  --decoder_model_name DECODER_MODEL_NAME
##                        Path to pretrained model or model identifier from huggingface.co/models.
##  --input_tokenizer_name INPUT_TOKENIZER_NAME
##                        Pretrained tokenizer name or path if not the same as encoder_model_name
##  --output_tokenizer_name OUTPUT_TOKENIZER_NAME
##                        Pretrained tokenizer name or path if not the same as encoder_model_name
##  --use_slow_tokenizer  If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).
##  --learning_rate LEARNING_RATE
##                        Initial learning rate (after the potential warmup period) to use.
##  --weight_decay WEIGHT_DECAY
##                        Weight decay to use.
##  --num_train_epochs NUM_TRAIN_EPOCHS
##                        Total number of training epochs to perform.
##  --max_train_steps MAX_TRAIN_STEPS
##                        Total number of training steps to perform. If provided, overrides num_train_epochs.
##  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
##                        Number of updates steps to accumulate before performing a backward/update pass.
##  --lr_scheduler_type {linear,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}
##                        The scheduler type to use.
##  --num_warmup_steps NUM_WARMUP_STEPS
##                        Number of steps for the warmup in the lr scheduler.
##  --output_dir OUTPUT_DIR
##                        Where to store the final model.
##  --seed SEED           A seed for reproducible training.
##  --push_to_hub         Whether or not to push the model to the Hub.
##  --hub_model_id HUB_MODEL_ID
##                        The name of the repository to keep in sync with the local `output_dir`.
##  --hub_token HUB_TOKEN
##                        The token to use to push to the Model Hub.
##
```

