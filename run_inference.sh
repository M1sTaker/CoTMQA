export TRANSFORMERS_CACHE=/root/autodl-tmp/fqy/cache
export HF_HOME=/root/autodl-tmp/fqy/cache

# bridge generation
CUDA_VISIBLE_DEVICES=0,1 python main.py \
    --model pretrained_models/roberta-base-squad2 \
    --user_msg answer --img_type detr \
    --bs 8 --eval_bs 4 --eval_acc 10 \
    --prompt_format ENTCAP-B \
    --evaluate_dir experiments/pretrained_models-roberta-base-squad2_detr_QESCAP-B_lr5e-05_bs24_ep20_maxtextlen512_stride128_fp16 \
    --bridge_eval --skip-no-answer

# answer inference
CUDA_VISIBLE_DEVICES=0 python main.py \
    --model pretrained_models/roberta-base-squad2 \
    --bs 16 \
    --prompt_format QPBC-A \
    --evaluate_dir experiments/39.5roberta-base-squad2_None_QPBC-A_lr1e-07_bs8_ep10_maxtextlen512_stride128_fp16/checkpoint-48377 \
    --eval_generated_bridge_le bridge_generation_result/dev_correct_bridge.json \
    --test_generated_bridge_le bridge_generation_result/test_correct_bridge.json \
    --eval_q_bridge_le bridge_generation_result/box0.4_text0.5_short/dev_q_bridges.json \
    --test_q_bridge_le bridge_generation_result/box0.4_text0.5_short/test_q_bridges.json \
    --img_type detr