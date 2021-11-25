srun -N1 --partition=gpu --gres=gpu:4 --export=ALL --time=1440 --job-name=xlm-camembert \
  accelerate launch --config_file $CHECKPOINTS_DIR/fr-ar/xlm-camembert-init/try/accelerate_config.yaml $CODE_DIR/huggingface-mt-trainer/main.py \
    --train_file $DATASETS_DIR/fr-ar/raw/news-commentary/data.tok.tsv \
    --train_idx_file $DATASETS_DIR/fr-ar/raw/news-commentary/data.idx  \
    --validation_file $DATASETS_DIR/fr-ar/raw/news-commentary/data.tok.tsv \
    --validation_idx_file $DATASETS_DIR/fr-ar/raw/news-commentary/data.idx \
    --cache_dir $CHECKPOINTS_DIR/fr-ar/xlm-camembert-init/hf-cache \
    --source_lang ar --target_lang fr \
    --cache_dir hf-cache \
    --max_sequence_length 512 --max_batch_size 16 --max_tokens 2048 \
    --preprocessing_num_workers 4 \
    --num_beams 5 \
    --encoder_model_name xlm-roberta-base --decoder_model_name camembert-base \
    --input_tokenizer_name xlm-roberta-base --output_tokenizer_name camembert-base \
    --learning_rate 5e-4 \
    --num_train_epochs 5 \
    --output_dir $CHECKPOINTS_DIR/fr-ar/xlm-camembert-init/try \
    --seed 8
