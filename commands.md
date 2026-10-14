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



# MLflow
mlflow ui --backend-store-uri sqlite:///runs/mlflow.db
