export TRANSFORMERS_CACHE=/root/autodl-tmp/fqy/cache
export HF_HOME=/root/autodl-tmp/fqy/cache

# bridge generation
CUDA_VISIBLE_DEVICES=0 python main.py \
    --model pretrained_models/roberta-base-squad2 \
    --img_type detr \
    --epoch 10 --bs 32 \
    --prompt_format PQCAP-B \
    --bridge_eval --fp16

# answer inference
CUDA_VISIBLE_DEVICES=0 python main.py \
    --model pretrained_models/bert-large-uncased \
    --epoch 10 --bs 8 \
    --lr 1e-7 \
    --load_best_model \
    --prompt_format RQC-A \
    --eval_generated_bridge_le bridge_generation_result/box0.4_text0.5_short/dev_prediction.json \
    --test_generated_bridge_le bridge_generation_result/box0.4_text0.5_short/test_prediction.json \
    --eval_q_bridge_le bridge_generation_result/box0.4_text0.5_short/dev_q_bridges.json \
    --test_q_bridge_le bridge_generation_result/box0.4_text0.5_short/test_q_bridges.json \
    --fp16 \
    --img_type detr

# one-hop
CUDA_VISIBLE_DEVICES=0 python main.py \
    --model pretrained_models/roberta-base-squad2 \
    --epoch 10 --bs 8 \
    --lr 1e-7 \
    --load_best_model \
    --prompt_format QC-A \
    --fp16 \
    --img_type detr
