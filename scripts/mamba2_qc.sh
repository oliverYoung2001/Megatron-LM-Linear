#!/bin/bash

# Use: ./train.sh <data-path> <tokenizer-path>
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Experiment Configurations
MODEL_SCALE="small_7B" # 
# MODEL_SCALE="800M" #
# MODEL_SCALE="7B" #
HYBRID_ATTENTION_RATIO=0
VOCAB_SIZE=32768
SEQ_LEN=4096
# SEQ_LEN=$((128*1024))  # OOM
SEQ_LEN_PER_GPU=$((64*1024))  # OOM
CP_TYPE='cp'
CPP_STAGES=4
CP_TYPE='hp'
# End

case "${MODEL_SCALE}" in
    "small_7B")
        WORLD_SIZE=1
        WORLD_SIZE=2
        # WORLD_SIZE=8
        TP=1
        CP=$WORLD_SIZE
        SEQ_LEN=$(($SEQ_LEN_PER_GPU * $WORLD_SIZE))
        NUM_LAYERS=4
        HIDDEN_SIZE=4096
        NUM_ATTENTION_HEADS=128 # Hqk=8, Nv=128; dk=128, dv=64
        GLOBAL_BATCH_SIZE=1
        MBS=1
        ;;
    "800M")
        WORLD_SIZE=1
        TENSOR_MODEL_PARALLEL_SIZE=1
        NUM_LAYERS=48
        HIDDEN_SIZE=1024
        NUM_ATTENTION_HEADS=16
        GLOBAL_BATCH_SIZE=32
        MBS=4
        ;;
    "8B")
        TENSOR_MODEL_PARALLEL_SIZE=4
        NUM_LAYERS=56
        HIDDEN_SIZE=4096
        NUM_ATTENTION_HEADS=32
        GLOBAL_BATCH_SIZE=8
        MBS=4
        ;;
    "7B")
        WORLD_SIZE=8
        TENSOR_MODEL_PARALLEL_SIZE=4
        NUM_LAYERS=64
        HIDDEN_SIZE=4096
        NUM_ATTENTION_HEADS=128 # 8, 128
        GLOBAL_BATCH_SIZE=8
        MBS=4
        ;;
    *)
        echo "Invalid version specified"
        exit 1
        ;;
esac

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_IB_TIMEOUT=19
export NCCL_IB_QPS_PER_CONNECTION=4

export TRITON_CACHE_DIR="./triton-cache/"
export TRITON_CACHE_MANAGER="megatron.core.ssm.triton_cache_manager:ParallelFileCacheManager"

TRAIN_ITERS=10

# Slurm Args
CLUSTER_NAME="qc"
export WORLD_SIZE
PARTITION=debug
# NODES="bjdb-h20-node-020"
if [[ $WORLD_SIZE -le 8 ]]; then
    NNODES=1
    NPROC_PER_NODE=${WORLD_SIZE}
else
    NNODES=$((WORLD_SIZE/8))
    NPROC_PER_NODE=8
fi
NGPUS_PER_NODE=$NPROC_PER_NODE
export MASTER_ADDR="localhost"
export MASTER_PORT=12582
SRUN_SCRIPT="srun \
    --partition=${PARTITION} \
    --nodes=${NNODES} \
    --ntasks-per-node=${NPROC_PER_NODE} \
    --gpus-per-node=${NGPUS_PER_NODE} \
"
if [ ! -z $NODES ]; then
    SRUN_SCRIPT+=" -w ${NODES} "
fi
# Log Args
EXP_NAME=mamba2_${MODEL_SCALE}
mkdir -p logs/${EXP_NAME}
# Profile Args
PROFILE_ARGS=" \
    --profile \
    --profile-step-start $(($TRAIN_ITERS-2)) \
    --profile-step-end $TRAIN_ITERS \
    --use-pytorch-profiler \
    --profile-ranks 0 \
"
TENSORBOARD_DIR="./logs/tb"
mkdir -p ${TENSORBOARD_DIR}
export TRACE_NAME=${CLUSTER_NAME}_${EXP_NAME}_S${SEQ_LEN}_CP${CP}_TP${TP}

PROFILE_ARGS=""
# End
options=" \
       --context-parallel-type ${CP_TYPE} \
       --cpp-stages ${CPP_STAGES} \
       ${PROFILE_ARGS} \
       --train-iters $TRAIN_ITERS \
       --mock-data \
       --no-gradient-accumulation-fusion \
       --context-parallel-size ${CP} \
       --tensor-model-parallel-size ${TP} \
       --sequence-parallel \
       --pipeline-model-parallel-size 1 \
       --use-distributed-optimizer \
       --overlap-param-gather \
       --overlap-grad-reduce \
       --untie-embeddings-and-output-weights \
       --init-method-std 0.02 \
       --position-embedding-type none \
       --num-layers ${NUM_LAYERS} \
       --hidden-size ${HIDDEN_SIZE} \
       --num-attention-heads ${NUM_ATTENTION_HEADS} \
       --group-query-attention \
       --num-query-groups 8 \
       --hybrid-attention-ratio $HYBRID_ATTENTION_RATIO \
       --hybrid-mlp-ratio 0.5 \
       --seq-length ${SEQ_LEN} \
       --max-position-embeddings ${SEQ_LEN} \
       --split 99,1,0 \
       --tokenizer-type NullTokenizer \
       --vocab-size ${VOCAB_SIZE} \
       --distributed-backend nccl \
       --micro-batch-size $MBS \
       --global-batch-size ${GLOBAL_BATCH_SIZE} \
       --lr 2.5e-4 \
       --min-lr 2.5e-5 \
       --lr-decay-style cosine \
       --weight-decay 0.1 \
       --clip-grad 1.0 \
       --attention-dropout 0.0 \
       --hidden-dropout 0.0 \
       --disable-bias-linear \
       --normalization RMSNorm \
       --adam-beta1 0.9 \
       --adam-beta2 0.95 \
       --log-interval 10000 \
       --save-interval 2000 \
       --eval-interval 2000 \
       --eval-iters 0 \
       --bf16 \
       --use-mcore-models \
       --spec megatron.core.models.mamba.mamba_layer_specs mamba_stack_spec \
       --no-create-attention-mask-in-dataloader \
       --tensorboard-dir ${TENSORBOARD_DIR}"


    #    --tokenizer-type GPTSentencePieceTokenizer \
    #    --tokenizer-model ${TOKENIZER_MODEL} \
    #    --train-samples ${TRAIN_SAMPLES} \
    #    --lr-warmup-samples ${LR_WARMUP_SAMPLES} \
    #    --lr-decay-samples ${LR_DECAY_SAMPLES} \
    #    --save ${CHECKPOINT_DIR} \
    #    --load ${CHECKPOINT_DIR} \
    #    --data-path ${DATA_PATH} \
    #    --data-cache-path ${DATACACHE_DIR} \
    
# torchrun --nproc_per_node 8 ../../pretrain_mamba.py ${options}

$SRUN_SCRIPT python pretrain_mamba.py ${options} \
2>&1 | tee logs/${EXP_NAME}/output_${TIMESTAMP}.log
