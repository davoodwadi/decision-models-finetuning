# From here down to "# Data recipes": earlier runs, kept as they were run. They used the dataset
# flags (--train, --min_n, --single_respondent, --train_slots, --calibration_frac), now replaced by --data <recipe>.

python train.py --kev --out runs/issp-0.8b-issp-subpop-v1 --epoch 10 --base jaredpalmer/kev-0.8b --train issp2020 subpop100


python train.py --kev --out runs/issp-0.8b-v1 --epoch 10 --base jaredpalmer/kev-0.8b 


python train.py --kev --out runs/issp-4b-v1 --epoch 10 --base jaredpalmer/kev-4b --checkpointing


kev/.venv/bin/python train.py --kev --base jaredpalmer/kev-0.8b --checkpointing --epochs 10 --out runs/issp-all-0.8b



kev/.venv/bin/python train.py --kev --base jaredpalmer/kev-0.8b --checkpointing --epochs 10 --out runs/issp-all-0.8b

kev/.venv/bin/python train.py --kev --base jaredpalmer/kev-0.8b --epochs 10 --out runs/issp-all-0.8b --checkpointing --lr 5e-5 --batch 128


# Single respondent for training
kev/.venv/bin/python train.py --kev --base jaredpalmer/kev-0.8b --epochs 10 --out runs/issp-all-0.8b --checkpointing --lr 5e-5 --batch 128 --single_respondent

# Train for longer with higher learning rate

kev/.venv/bin/python train.py --kev \
--base jaredpalmer/kev-0.8b \
--epochs 10 \
--out runs/issp-all-0.8b \
--checkpointing \
--lr 5e-4 \
--batch 128 

# Add min_n 10

kev/.venv/bin/python train.py --kev \
--base jaredpalmer/kev-0.8b \
--epochs 10 \
--out runs/issp-all-0.8b \
--checkpointing \
--lr 5e-4 \
--batch 128 \
--min_n 10

# lora_rank 64

kev/.venv/bin/python train.py --kev \
--base jaredpalmer/kev-0.8b \
--epochs 10 \
--out runs/issp-all-0.8b-r64 \
--checkpointing \
--lr 1e-4 \
--batch 128 \
--min_n 10 \
--lora_rank 64

## New data structure

kev/.venv/bin/python train.py --kev --data configs/data/issp-all-group-min10.json --base jaredpalmer/kev-0.8b --epochs 10 --out runs/issp-all-0.8b-r64 --checkpointing --lr 1e-4 --batch 128 --lora_rank 64

# lr_schedule: constant 
kev/.venv/bin/python train.py --kev \
--base jaredpalmer/kev-0.8b \
--epochs 10 \
--out runs/issp-all-0.8b-constlr \
--checkpointing \
--lr 5e-4 \
--batch 128 \
--min_n 10 \
--lr_schedule constant

# testing

kev/.venv/bin/python train.py --kev \
--base jaredpalmer/kev-0.8b \
--epochs 300 \
--out runs/issp-all-0.8b-test-r64-constant \
--checkpointing \
--lr 1e-4 \
--batch 128 \
--min_n 10 \
--train_slots 128 \
--no_eval \
--lora_rank 64 \
--lr_schedule constant


kev/.venv/bin/python train.py --kev \
--base jaredpalmer/kev-0.8b \
--epochs 300 \
--out runs/issp-all-0.8b-test-r64 \
--checkpointing \
--lr 1e-4 \
--batch 128 \
--min_n 10 \
--train_slots 128 \
--no_eval \
--lora_rank 64 


# Full finetune - kev pointer model 

kev/.venv/bin/python train.py --kev --data configs/data/issp-all-group-min10.json --base jaredpalmer/kev-0.8b --full_finetune --epochs 10 --out runs/issp-all-0.8b-full --checkpointing --lr 1e-4 --batch 128

# Full finetune - LLM with logits

kev/.venv/bin/python train.py --data configs/data/issp-all-group-min10.json --epochs 10 --lr 1e-4 --batch 128 --micro_batch 32 --checkpointing --out runs/issp-all-qwen0.8b-letters


# Data recipes (--data configs/data/*.json): the same runs as above with the current trainer

# lora_rank 64, group shares with min_n 10 (as issp-all-0.8b-r64)
kev/.venv/bin/python train.py --kev \
--data configs/data/issp-all-group-min10.json \
--base jaredpalmer/kev-0.8b \
--epochs 300 \
--out runs/issp-all-0.8b-r64 \
--checkpointing \
--lr 1e-4 \
--batch 128 \
--lora_rank 64

# single respondent
kev/.venv/bin/python train.py --kev \
--data configs/data/issp-all-single.json \
--base jaredpalmer/kev-0.8b \
--epochs 10 \
--out runs/issp-all-0.8b-single \
--checkpointing \
--lr 5e-5 \
--batch 128

# testing: 128 fixed records (as issp-all-0.8b-test-r64)
kev/.venv/bin/python train.py --kev \
--data configs/data/issp-all-group-min10-fixed128.json \
--base jaredpalmer/kev-0.8b \
--epochs 300 \
--out runs/issp-all-0.8b-test-r64 \
--checkpointing \
--lr 1e-4 \
--batch 128 \
--no_eval \
--lora_rank 64

# LLM through its own next-token head (option letters), every weight trained (models/llm_letters.py)
# default --base Qwen/Qwen3.5-0.8B (download it first), --prompt prompts/letters-v1.txt
kev/.venv/bin/python train.py \
--data configs/data/issp-all-group-min10.json \
--epochs 10 \
--out runs/issp-all-qwen0.8b-letters \
--checkpointing \
--lr 1e-4 \
--batch 128 \
--micro_batch 16

# US only

kev/.venv/bin/python train.py --kev --data configs/data/issp-us-group-min10-attr1.json --full_finetune --lr 1e-4 --batch 128 --checkpointing --out runs/issp-us-0.8b-full-attr1

# US only, exactly 2 attributes besides the country, groups of at least 100 respondents

kev/.venv/bin/python show_train_samples.py --kev --data configs/data/issp-us-group-min100-attr2.json

kev/.venv/bin/python train.py --kev --data configs/data/issp-us-group-min100-attr2.json --full_finetune --lr 1e-4 --batch 128 --checkpointing --epochs 100 --out runs/issp-us-0.8b-full-min100-attr2-noaug

## Testing different augs

kev/.venv/bin/python train.py --kev --data configs/data/issp-us-group-min100-attr2.json --full_finetune --lr 1e-4 --batch 128 --checkpointing --epochs 50 --validate_every 10 --out runs/issp-us-0.8b-full-min100-attr2-noaug

kev/.venv/bin/python train.py --kev --data configs/data/issp-us-group-min100-attr2.json --full_finetune --lr 1e-4 --batch 128 --checkpointing --epochs 50 --validate_every 10 --augment "layouts=lines,bullets,one line,json,prose" --out runs/issp-us-0.8b-full-min100-attr2-aug-layouts

kev/.venv/bin/python train.py --kev --data configs/data/issp-us-group-min100-attr2.json --full_finetune --lr 1e-4 --batch 128 --checkpointing --epochs 50 --validate_every 10 --augment key_names --out runs/issp-us-0.8b-full-min100-attr2-aug-keynames

kev/.venv/bin/python train.py --kev --data configs/data/issp-us-group-min100-attr2.json --full_finetune --lr 1e-4 --batch 128 --checkpointing --epochs 50 --validate_every 10 --augment value_wording --out runs/issp-us-0.8b-full-min100-attr2-aug-valuewording

kev/.venv/bin/python train.py --kev --data configs/data/issp-us-group-min100-attr2.json --full_finetune --lr 1e-4 --batch 128 --checkpointing --epochs 50 --validate_every 10 --augment punctuation --out runs/issp-us-0.8b-full-min100-attr2-aug-punctuation

kev/.venv/bin/python train.py --kev --data configs/data/issp-us-group-min100-attr2.json --full_finetune --lr 1e-4 --batch 128 --checkpointing --epochs 50 --validate_every 10 --augment shuffle_order --out runs/issp-us-0.8b-full-min100-attr2-aug-shuffleorder

kev/.venv/bin/python train.py --kev --data configs/data/issp-us-group-min100-attr2.json --full_finetune --lr 1e-4 --batch 128 --checkpointing --epochs 50 --validate_every 10 --augment p_reverse=0.5 --out runs/issp-us-0.8b-full-min100-attr2-aug-reverse

kev/.venv/bin/python train.py --kev --data configs/data/issp-us-group-min100-attr2.json --full_finetune --lr 1e-4 --batch 128 --checkpointing --epochs 50 --validate_every 10 --augment shuffle_options --out runs/issp-us-0.8b-full-min100-attr2-aug-shuffleoptions

kev/.venv/bin/python train.py --kev --data configs/data/issp-us-group-min100-attr2.json --full_finetune --lr 1e-4 --batch 128 --checkpointing --epochs 50 --validate_every 10 --augment full --out runs/issp-us-0.8b-full-min100-attr2-aug-full

kev/.venv/bin/python train.py --kev --data configs/data/issp-us-group-min100-attr2.json --full_finetune --lr 1e-4 --batch 128 --checkpointing --epochs 50 --validate_every 10 --augment shuffle_order p_reverse=0.5 --out runs/issp-us-0.8b-full-min100-attr2-aug-shuffleorder-reverse

# US only, 1-2 attributes besides the country, groups of at least 100 respondents

kev/.venv/bin/python show_train_samples.py --kev --data configs/data/issp-us-group-min100-attr1-2.json

kev/.venv/bin/python train.py --kev --data configs/data/issp-us-group-min100-attr1-2.json --full_finetune --lr 1e-4 --batch 128 --checkpointing --epochs 500 --out runs/issp-us-0.8b-full-min100-attr1-2-noaug

kev/.venv/bin/python train.py --kev --data configs/data/issp-us-group-min100-attr1-2.json --full_finetune --lr 1e-4 --batch 128 --checkpointing --epochs 50 --validate_every 10 --out runs/issp-us-0.8b-full-min100-attr1-2-noaug

# US only, 2-3 attributes besides the country, groups of at least 100 respondents

kev/.venv/bin/python train.py --kev --data configs/data/issp-us-group-min100-attr2-3.json --full_finetune --lr 1e-4 --batch 128 --checkpointing --epochs 50 --validate_every 10 --out runs/issp-us-0.8b-full-min100-attr2-3-noaug

# US only, 2-4 attributes besides the country, groups of at least 100 respondents

kev/.venv/bin/python train.py --kev --data configs/data/issp-us-group-min100-attr2-4.json --full_finetune --lr 1e-4 --batch 128 --checkpointing --epochs 50 --validate_every 10 --out runs/issp-us-0.8b-full-min100-attr2-4-noaug


# US only, 2-4 attributes besides the country, groups of at least 100 respondents - letters model

kev/.venv/bin/python train.py --data configs/data/issp-us-group-min100-attr2-4.json --lr 1e-4 --batch 128 --checkpointing --epochs 50 --validate_every 10 --out runs/issp-us-qwen0.8b-letters-min100-attr2-4-noaug

# Incremental unfreeze

kev/.venv/bin/python train.py --kev --data configs/data/issp-us-group-min100-attr2-4.json --full_finetune --train_last_layers 2 --lr 1e-4 --batch 128 --checkpointing --epochs 10 --validate_every 10 --out runs/unfreeze-l2

kev/.venv/bin/python train.py --kev --base runs/unfreeze-l2-v1/best --data configs/data/issp-us-group-min100-attr2-4.json --full_finetune --train_last_layers 3 --lr 1e-5 --batch 128 --checkpointing --epochs 10 --validate_every 10 --out runs/unfreeze-l3

kev/.venv/bin/python train.py --kev --base runs/unfreeze-l3-v1/best --data configs/data/issp-us-group-min100-attr2-4.json --full_finetune --train_last_layers 6 --lr 1e-6 --batch 128 --checkpointing --epochs 10 --validate_every 10 --out runs/unfreeze-l6

# Incremental unfreeze - 4B

kev/.venv/bin/python train.py --kev --base jaredpalmer/kev-4b --data configs/data/issp-us-group-min100-attr2-4.json --full_finetune --train_last_layers 2 --lr 1e-5 --batch 4 --checkpointing --epochs 100 --validate_every 50 --out runs/unfreeze-4b-l2

kev/.venv/bin/python train.py --kev --base runs/unfreeze-4b-l2-v1/best --data configs/data/issp-us-group-min100-attr2-4.json --full_finetune --train_last_layers 3 --lr 1e-5 --batch 4 --checkpointing --epochs 100 --validate_every 50 --out runs/unfreeze-4b-l3 --lr_decay 2.6

kev/.venv/bin/python train.py --kev --base runs/unfreeze-4b-l3-v1/best --data configs/data/issp-us-group-min100-attr2-4.json --full_finetune --train_last_layers 6 --lr 1e-5 --batch 4 --checkpointing --epochs 100 --validate_every 50 --out runs/unfreeze-4b-l6 --lr_decay 2.6

kev/.venv/bin/python train.py --kev --base runs/unfreeze-4b-l6-v1/best --data configs/data/issp-us-group-min100-attr2-4.json --full_finetune --train_last_layers 12 --lr 1e-5 --batch 4 --checkpointing --epochs 100 --validate_every 50 --out runs/unfreeze-4b-l12 --lr_decay 2.6

kev/.venv/bin/python train.py --kev --base runs/unfreeze-4b-l12-v1/best --data configs/data/issp-us-group-min100-attr2-4.json --full_finetune --train_last_layers 24 --lr 1e-5 --batch 4 --checkpointing --epochs 100 --validate_every 50 --out runs/unfreeze-4b-l24 --lr_decay 2.6

kev/.venv/bin/python train.py --kev --base runs/unfreeze-4b-l24-v1/best --data configs/data/issp-us-group-min100-attr2-4.json --full_finetune --lr 1e-5 --batch 4 --checkpointing --epochs 100 --validate_every 50 --out runs/unfreeze-4b-all --lr_decay 2.6

# 4B LoRA

kev/.venv/bin/python train.py --kev --base jaredpalmer/kev-4b --data configs/data/issp-us-group-min100-attr2-4.json --lr 1e-4 --batch 4 --checkpointing --epochs 100 --validate_every 50 --out runs/lora-4b-r64 --lora_rank 64

# 9B LoRA

kev/.venv/bin/python train.py --kev --base jaredpalmer/kev-9b --weights_dtype bf16 --data configs/data/issp-us-group-min100-attr2-4.json --lr 1e-4 --batch 4 --checkpointing --epochs 100 --validate_every 50 --out runs/lora-9b-r64 --lora_rank 64



# show what the LLM reads (tokenizer only)
kev/.venv/bin/python show_train_samples.py --year 2022 --question v45 --samples 2

# MLflow
mlflow ui --backend-store-uri sqlite:///runs/mlflow.db
