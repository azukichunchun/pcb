#!/bin/bash

# custom config
DATA=/home/yhiro/CoOp_/data

TRAINER=$1
DATASET=$2
ALMETHOD=$3 # Active learning method (random, entropy, coreset, badge)
MODE=$4
CFG=$5
LOADEP=$6
SEED=$7

SHOTS=-1
CSC=False  # class-specific context (False or True)
CTP="end"  # class token position (end or middle)
NCTX=4  # number of context tokens

COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}_${ALMETHOD}/${CFG}/seed${SEED}

DIR=output/base2new/train_base/${COMMON_DIR}
#DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS} \
        TRAINER.COOPAL.METHOD ${ALMETHOD} \
        DATASET.SUBSAMPLE_CLASSES base
fi

MODEL_DIR=output/base2new/train_base/${COMMON_DIR}
DIR=output/base2new/test_new/${COMMON_DIR}
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --model-dir ${MODEL_DIR} \
        --load-epoch ${LOADEP} \
        --eval-only \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_SHOTS ${SHOTS} \
        TRAINER.COOPAL.METHOD ${ALMETHOD} \
        DATASET.SUBSAMPLE_CLASSES new
fi