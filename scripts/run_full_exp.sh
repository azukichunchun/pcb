CUDA=$1
TRAINER=$2
DATASET=$3
ALMETHOD=$4
CFG=$5
LOADEP=$6

echo CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/alvlm/run_exp.sh ${TRAINER} ${DATASET} ${ALMETHOD} none ${CFG} ${LOADEP} 1 &&
echo CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/alvlm/run_exp.sh ${TRAINER} ${DATASET} ${ALMETHOD} none ${CFG} ${LOADEP} 2 &&
echo CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/alvlm/run_exp.sh ${TRAINER} ${DATASET} ${ALMETHOD} none ${CFG} ${LOADEP} 3

CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/alvlm/run_exp.sh ${TRAINER} ${DATASET} ${ALMETHOD} none ${CFG} ${LOADEP} 1 &&
CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/alvlm/run_exp.sh ${TRAINER} ${DATASET} ${ALMETHOD} none ${CFG} ${LOADEP} 2 &&
CUDA_VISIBLE_DEVICES=${CUDA} bash scripts/alvlm/run_exp.sh ${TRAINER} ${DATASET} ${ALMETHOD} none ${CFG} ${LOADEP} 3
