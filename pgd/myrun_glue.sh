#!/usr/bin/env bash

function runexp {

export GLUE_DIR=/data/cheng/GLUE
export TASK_NAME=${1}

gpu=${2}      # The GPU you want to use
mname=${3}    # Model name
alr=${4}      # Step size of gradient ascent
amag=${5}     # Magnitude of initial (adversarial?) perturbation
anorm=${6}    # Maximum norm of adversarial perturbation
asteps=${7}   # Number of gradient ascent steps for the adversary
lr=${8}       # Learning rate for model parameters
bsize=${9}    # Batch size
gas=${10}     # Gradient accumulation. bsize * gas = effective batch size
seqlen=512    # Maximum sequence length
hdp=${11}     # Hidden layer dropouts for ALBERT
adp=${12}     # Attention dropouts for ALBERT
ts=${13}      # Number of training steps (counted as parameter updates)
ws=${14}      # Learning rate warm-up steps
seed=${15}    # Seed for randomness
wd=${16}      # Weight decay

expname=PGD-${mname}-${TASK_NAME}-alr${alr}-amag${amag}-anm${anorm}-as${asteps}-sl${seqlen}-lr${lr}-bs${bsize}-gas${gas}-hdp${hdp}-adp${adp}-ts${ts}-ws${ws}-wd${wd}-seed${seed}

nohup python run_glue_pgd.py \
  --model_type albert \
  --model_name_or_path ${mname} \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length ${seqlen} \
  --per_gpu_train_batch_size ${bsize} --gradient_accumulation_steps ${gas} \
  --learning_rate ${lr} --weight_decay ${wd} \
  --gpu ${gpu} \
  --output_dir /data/cheng/pgd/checkpoints/${expname}/ \
  --hidden_dropout_prob ${hdp} --attention_probs_dropout_prob ${adp} \
  --adv-lr ${alr} --adv-init-mag ${amag} --adv-max-norm ${anorm} --adv-steps ${asteps} \
  --expname ${expname} --evaluate_during_training \
  --max_steps ${ts} --warmup_steps ${ws} --seed ${seed} \
  --logging_steps 100 --save_steps 100 \
  --fp16 \
  --comet \
  --overwrite_output_dir > /data/cheng/pgd/tmp/sst2_alr06_l2_anm06.log 2>&1
}


# runexp TASK_NAME  gpu      model_name      adv_lr  adv_mag  anorm  asteps  lr     bsize  grad_accu  hdp  adp      ts     ws     seed      wd
runexp  SST-2       0       albert-xxlarge-v2  0.6       6e-1   0.6      3    1e-5     8       1        0.1   0    20935   1256     42     1e-2



