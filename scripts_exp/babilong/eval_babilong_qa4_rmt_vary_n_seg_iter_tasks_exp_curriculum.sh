#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./finetune_babilong_baseline.sh
set -e
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_TYPE=decoder
MEMORY_CELL=modeling_rmt.language_modeling:MemoryCell
RECURRENT_WRAPPER=modeling_rmt.language_modeling:RecurrentWrapper
BACKBONE_CLS=transformers:AutoModelForCausalLM

NOISE_DATASET=pg19
METRIC=exact_match

MODEL_NAME=gpt2  # backbone model

ITERS=5000
TBS=1
INIT_BS=1

for TASK_DATASET in qa4_two-arg-relations
do

for LR in 5e-05
do

SRC_N_SEGMENTSS=(8 16 32)

for (( j=1; j<${#SRC_N_SEGMENTSS[@]}; j++ ))
do

SRC_N_SEGMENTS=${SRC_N_SEGMENTSS[j]}

j1=$((j-1))
SRC_SRC_N_SEGMENTS=${SRC_N_SEGMENTSS[j1]}
echo SRC_N_SEGMENTS $SRC_N_SEGMENTS SRC_SRC_N_SEGMENTS $SRC_SRC_N_SEGMENTS


for SEGMENT_SIZE in 512
do # size of one segment in tokens

for MAX_N_SEGMENTS in 1 2 4 8 16 32 64 128 256
do


for MEMORY_SIZE in 16
do

SAMPLE_SIZE=$((MAX_N_SEGMENTS*SEGMENT_SIZE)) # length of task sample in tokens
BS=1

GRAD_ACC_STEPS=$(($TBS/($BS*$NP)))
echo TBS $TBS BS $BS GRAD_ACC_STEPS $GRAD_ACC_STEPS
SCHEDULER=linear

for N in 4
do

K2=-1   # BPTT unroll length

NP=$NP
ACCEL_CONFIG=/home/jovyan/rmt/babilong/accel_configs/accelerate/deepspeed_bf16_tbs${TBS}bs${BS}g${GRAD_ACC_STEPS}c1.0np${NP}.yaml
cd accel_configs/
python create_config.py \
        --bf16 \
        --train_batch_size $TBS\
        --train_micro_batch_size_per_gpu $BS\
        --gradient_accumulation_steps $GRAD_ACC_STEPS\
        --np $NP\
        --gradient_clipping 1.0
cd ..

echo RUNNING: TASK_DATASET $TASK_DATASET MEMORY_SIZE $MEMORY_SIZE SEGMENT_SIZE $SEGMENT_SIZE MAX_N_SEGMENTS $MAX_N_SEGMENTS
echo SAMPLE_SIZE $SAMPLE_SIZE MODEL_NAME $MODEL_NAME  LR $LR N $N
echo gradient accumulation steps $GRAD_ACC_STEPS

accelerate launch --config_file $ACCEL_CONFIG --main_process_port 29031 run_finetuning_babilong_rmt.py \
        --task_dataset $TASK_DATASET \
        --noise_dataset $NOISE_DATASET \
        --babi_path /home/jovyan/rmt/babilong/data/tasks_1-20_v1-2/en-10k \
        --model_path /home/jovyan/rmt/runs/babilong/${TASK_DATASET}/$MODEL_NAME/${SCHEDULER}_adamw_wd1e-03_${MAX_N_SEGMENTS}x${SEGMENT_SIZE}_mem${MEMORY_SIZE}_bs64_bptt-${K2}_eval_from_${SRC_N_SEGMENTS}x${SEGMENT_SIZE}_bs1/run_$N \
        --model_cpt /home/jovyan/rmt/runs/babilong/${TASK_DATASET}/$MODEL_NAME/${SCHEDULER}_adamw_wd1e-03_${SRC_N_SEGMENTS}x${SEGMENT_SIZE}_mem${MEMORY_SIZE}_bs64_bptt-${K2}_from_cpt_${SRC_SRC_N_SEGMENTS}-${SRC_N_SEGMENTS}/run_$N/model_best \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --memory_cell_cls $MEMORY_CELL \
        --recurrent_wrapper_cls $RECURRENT_WRAPPER \
        --model_cls $BACKBONE_CLS \
        --segment_size $SEGMENT_SIZE \
        --sample_size $SAMPLE_SIZE \
        --max_n_facts $((SAMPLE_SIZE/10)) \
        --num_mem_tokens $MEMORY_SIZE \
        --max_n_segments $MAX_N_SEGMENTS\
        --vary_n_segments \
        --batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
        --num_training_steps $((ITERS*2)) \
        --iters $ITERS \
        --use_generate_on_valid \
        --validate_only \
        --k2 $K2 \
        --optimizer AdamW  --weight_decay 0.01 \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps $(($ITERS/10)) \
        --data_n_workers 2 \
        --log_interval $(($ITERS/100)) --valid_interval $(($ITERS/20)) \
        --optimize_metric $METRIC --optimize_mode max \
        --show_valid_examples 5 \
        --early_stopping_patience 15 \
        --seed $(($N+42)) \
        --clip_grad_norm 1.0
        
done
done
done
done
done
done
done
done
echo "done"