DS_CONFIG=./examples_deepspeed/finetune_hf_llama/ds_config_default.json
DATASET_PATH=./alpaca_data.json
# dataset link: https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json
NODES="1"
HF_LLAMA_PATH=/data/llama-7b/
if [ ! -d $HF_LLAMA_PATH ]; then
  echo "[INFO] Cloning model weight into $HF_LLAMA_PATH ..."
  git lfs install
  dir_name=$(basename $HF_LLAMA_PATH)
  clone_path="/data/"$dir_name 
  git clone https://huggingface.co/huggyllama/llama-7b $clone_path
fi

HF_LLAMA_REPO="huggyllama/llama-7b"
# weights link: https://huggingface.co/huggyllama/llama-7b

MICRO_BATCH_SIZE=16
GLOBAL_BATCH_SIZE=256
TP=2
PP=2
# require to align with weight dimensions
HIDDEN_SIZE=4096
FFN_HIDDEN_SIZE=11008
NUM_LAYERS=32
NUM_HEADS=32
SEQ_LENGTH=512
######################################

MEGA_DS_LLAMA_PATH=/output/"llama-7b-mega-ds-T${TP}P${PP}"

# Below configuration required for llama model as per llama paper
# --no-query-key-layer-scaling \
# --attention-dropout 0 \
# --hidden-dropout 0 \
# --use-rotary-position-embeddings \
# --untie-embeddings-and-output-weights \
# --swiglu \
# --normalization rmsnorm \
# --disable-bias-linear \
######################################
cat <<EOT > $DS_CONFIG
{
  "train_batch_size" : $GLOBAL_BATCH_SIZE,
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "steps_per_print": 100,
  "zero_optimization": {
    "stage": 0
  },
  "bf16": {
    "enabled": true
  }
}
EOT

#--origin-hf-ckpt-dir $HF_LLAMA_REPO \
covert_args="deepspeed --num_nodes $NODES \ tools/hf2megads_weight_converter.py \
--hf-ckpt-num-shards 2 \
--origin-hf-ckpt-dir $HF_LLAMA_PATH \
--save $MEGA_DS_LLAMA_PATH"

finetune_args="deepspeed --num_nodes $NODES \ /home/deepspeed/Megatron-DeepSpeed/finetune_llama.py \
--load $MEGA_DS_LLAMA_PATH"

comm_args="--tensor-model-parallel-size $TP \
--pipeline-model-parallel-size $PP \
--lr-warmup-iters 2000 \
--weight-decay 0.1 \
--clip-grad 1 \
--num-layers $NUM_LAYERS \
--hidden-size $HIDDEN_SIZE \
--num-attention-heads $NUM_HEADS \
--ffn-hidden-size $FFN_HIDDEN_SIZE \
--attention-dropout 0 \
--hidden-dropout 0 \
--no-query-key-layer-scaling \
--disable-bias-linear \
--normalization rmsnorm \
--use-rotary-position-embeddings \
--untie-embeddings-and-output-weights \
--swiglu \
--seq-length $SEQ_LENGTH \
--max-position-embeddings $SEQ_LENGTH \
--micro-batch-size $MICRO_BATCH_SIZE \
--global-batch-size $GLOBAL_BATCH_SIZE \
--train-iters 3500 \
--lr 2e-5 \
--tensorboard-dir tensorboard_output \
--lr-decay-iters 320000 \
--lr-decay-style cosine \
--log-interval 1 \
--eval-iters 100 \
--eval-interval 100 \
--data-path $DATASET_PATH \
--save-interval 1500 \
--split 100,0,0 \
--bf16 \
--zero-stage 0 \
--tokenizer-type HFTokenizer \
--tokenizer-model $HF_LLAMA_PATH \
--deepspeed_config ./examples_deepspeed/finetune_hf_llama/ds_config_default.json \
--deepspeed \
--distributed-backend nccl \
--num-workers 0 \
--no-masked-softmax-fusion \
--no-bias-gelu-fusion \
--no-bias-dropout-fusion \
--no-gradient-accumulation-fusion \
--repeated-dataloader"

if [ "$1" = "convert" ]; then
    task_args="$covert_args"
else
    task_args="$finetune_args"
fi

full_cmd="$task_args $comm_args"

eval "$full_cmd"
