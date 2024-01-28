import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np
import os
import argparse
import json
from tqdm import tqdm
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='experiments/pretrained_models-roberta-base-squad2_detr_PQCAP-B_lr5e-05_bs32_ep10_maxtextlen512_stride128_fp16')
    parser.add_argument('--output_dir', type=str, default='experiments/pretrained_models-roberta-base-squad2_detr_PQCAP-B_lr5e-05_bs32_ep10_maxtextlen512_stride128_fp16')
    parser.add_argument('--split', type=str, default='test')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print("args",args)
    
    if args.split == 'train':
        
        with open(os.path.join(args.data_root, 'new_train.json'), 'r', encoding='utf-8') as file:
            data = json.load(file)

        #turn to predictions format
        train_predictions = {entry['id']: entry['bridge'] for entry in data}

        with open(os.path.join(args.output_dir, 'train_predictions.json'), 'w', encoding='utf-8') as output_file:
            json.dump(train_predictions, output_file, ensure_ascii=False, indent=4)
    
    with open(os.path.join(args.data_root, args.split + '_predictions.json'), 'r', encoding='utf-8') as file:
        predictions = json.load(file)
    
    if args.split == 'train':
        image_pixels = np.load(os.path.join(args.data_root, args.split + '.npy'))
    elif args.split == 'eval':
        image_pixels = np.load('MuMuQA/eval/dev.npy')
    elif args.split == 'test':
        image_pixels = np.load('MuMuQA/eval/eval.npy')
    
    ckpt = "pretrained_models/instructblip-flan-t5-xl"
    model = InstructBlipForConditionalGeneration.from_pretrained(ckpt)
    processor = InstructBlipProcessor.from_pretrained(ckpt)
    device = "cuda:0"
    model.to(device)
    
    extended_bridges = {}
    for index, (id, bridge) in tqdm(enumerate(predictions.items()), desc="Processing", ncols=100, position=0, leave=True):
        image = Image.fromarray(image_pixels[index])
        text = f"Analyze the following picture which contains an entity named '{bridge}'. Describe the physical features, what is the entity '{bridge}' wearing, where it is located, and any other relevant information."
        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        outputs = model.generate(
                **inputs,
                do_sample=False,
                num_beams=5,
                max_length=128,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.5,
                length_penalty=1.0,
                temperature=1,
        )
        
        extended_bridge = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()  
        extended_bridges[id] = extended_bridge
    
    with open(os.path.join(args.output_dir, args.split + '_extended_bridges.json'), "w") as writer:
        writer.write(json.dumps(extended_bridges, indent=4) + "\n")