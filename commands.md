python train_issp.py --out runs/issp-0.8b-issp-subpop-v1 --epoch 10 --base jaredpalmer/kev-0.8b --train issp2020 subpop100


python train_issp.py --out runs/issp-0.8b-v1 --epoch 10 --base jaredpalmer/kev-0.8b 


python train_issp.py --out runs/issp-4b-v1 --epoch 10 --base jaredpalmer/kev-4b --checkpointing


kev/.venv/bin/python train_issp.py --base jaredpalmer/kev-0.8b --checkpointing --epochs 10 --out runs/issp-all-0.8b



kev/.venv/bin/python train_issp.py --base jaredpalmer/kev-0.8b --checkpointing --epochs 10 --out runs/issp-all-0.8b

kev/.venv/bin/python train_issp.py --base jaredpalmer/kev-0.8b --epochs 10 --out runs/issp-all-0.8b --checkpointing --lr 5e-5 --batch 128


# Single respondent for training
kev/.venv/bin/python train_issp.py --base jaredpalmer/kev-0.8b --epochs 10 --out runs/issp-all-0.8b --checkpointing --lr 5e-5 --batch 128 --single_respondent

# Train for longer with higher learning rate

kev/.venv/bin/python train_issp.py \
--base jaredpalmer/kev-0.8b \
--epochs 10 \
--out runs/issp-all-0.8b \
--checkpointing \
--lr 5e-4 \
--batch 128 

# Add min_n 10

kev/.venv/bin/python train_issp.py \
--base jaredpalmer/kev-0.8b \
--epochs 10 \
--out runs/issp-all-0.8b \
--checkpointing \
--lr 5e-4 \
--batch 128 \
--min_n 10

# lora_rank 64

kev/.venv/bin/python train_issp.py \
--base jaredpalmer/kev-0.8b \
--epochs 10 \
--out runs/issp-all-0.8b-r64 \
--checkpointing \
--lr 5e-4 \
--batch 128 \
--min_n 10 \
--lora_rank 64

# lr_schedule: constant 
kev/.venv/bin/python train_issp.py \
--base jaredpalmer/kev-0.8b \
--epochs 10 \
--out runs/issp-all-0.8b-constlr \
--checkpointing \
--lr 5e-4 \
--batch 128 \
--min_n 10 \
--lr_schedule constant


# MLflow
mlflow ui --backend-store-uri sqlite:///runs/mlflow.db
