#!/bin/bash

# ============================================================
# CAPA - ETTh1 Dataset Run Script
# Prediction Length: 96
# ============================================================

# 基础配置
TASK_NAME="long_term_forecast"
MODEL="CAPA"
MODEL_ID="CAPA_ETTh1"
DATA="ETTh1"
ROOT_PATH="./resources/all_datasets/ETT-small/"
DATA_PATH="ETTh1.csv"

# 数据配置
SEQ_LEN=720
LABEL_LEN=48
PRED_LEN=96
FEATURES="M"

# 训练配置
TRAIN_EPOCHS=20
BATCH_SIZE=32
PATIENCE=5
NUM_WORKERS=10
ITR=1
DES="capa_pl96"

# 优化器配置
LEARNING_RATE=0.001

# CAPA 特定配置
PERIOD_LEN=24
LATENT_DIM=8
PHASE_LAYERS=1
PHASE_ATTN_HEADS=4
PHASE_ATTN_DROPOUT=0.1

# RevIN 配置
USE_REVIN=1

# Huber Loss 配置
USE_HUBER=1
HUBER_DELTA=1.0

# GPU 配置 (根据实际情况修改)
USE_GPU=1
GPU=0

echo "============================================"
echo "CAPA - ETTh1 Dataset"
echo "Prediction Length: $PRED_LEN"
echo "Latent Dim: $LATENT_DIM"
echo "Phase Layers: $PHASE_LAYERS"
echo "Phase Attention Heads: $PHASE_ATTN_HEADS"
echo "============================================"

python run.py \
  --task_name $TASK_NAME \
  --is_training 1 \
  --model_id $MODEL_ID \
  --model $MODEL \
  --data $DATA \
  --root_path $ROOT_PATH \
  --data_path $DATA_PATH \
  --features $FEATURES \
  --seq_len $SEQ_LEN \
  --label_len $LABEL_LEN \
  --pred_len $PRED_LEN \
  --learning_rate $LEARNING_RATE \
  --period_len $PERIOD_LEN \
  --latent_dim $LATENT_DIM \
  --phase_layers $PHASE_LAYERS \
  --phase_attn_heads $PHASE_ATTN_HEADS \
  --phase_attn_dropout $PHASE_ATTN_DROPOUT \
  --use_revin $USE_REVIN \
  --use_huber $USE_HUBER \
  --huber_delta $HUBER_DELTA \
  --train_epochs $TRAIN_EPOCHS \
  --batch_size $BATCH_SIZE \
  --patience $PATIENCE \
  --itr $ITR \
  --num_workers $NUM_WORKERS \
  --use_gpu $USE_GPU \
  --gpu $GPU \
  --des $DES

echo "Training completed!"
