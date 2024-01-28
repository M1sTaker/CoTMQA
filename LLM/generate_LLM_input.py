import json
import os
from tqdm import tqdm
import argparse

def generate_prompt_bridge(args):
    prompt_bridge = {}
    
    with open(os.path.join(args.bridge_root, f'{args.split}_prediction.json'), 'r', encoding='utf-8') as file:
        prediction = json.load(file)
        
    with open(os.path.join(args.bridge_root, f'{args.split}_q_bridges.json'), 'r', encoding='utf-8') as file:
        q_bridges = json.load(file)
    
    for key, q_bridge in q_bridges.items():
        text = q_bridge + ' is ' + prediction[key] + '.'
        prompt_bridge[key] = text
    
    return prompt_bridge

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/root/autodl-tmp/fqy/Code/MQA/MuMuQA/eval')
    parser.add_argument('--bridge_root', type=str, default='/root/autodl-tmp/fqy/Code/MQA/bridge_generation_result/box0.4_text0.5_short')
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--output_dir', type=str, default='/root/autodl-tmp/fqy/Code/MQA/LLM/LLM_input')
    parser.add_argument('--input_format', type=str, default='QPBC', help='prompt format template',choices=['QC', 'QBC', 'PBQC', 'QPBC', 'RQC'])
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()
    print("args",args)
    
    with open(os.path.join(args.data_root, f'{args.split}.json'), 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    
    if 'PB' in args.input_format:
        prompt_bridge = generate_prompt_bridge(args)
    
    LLM_input = {}
    
    # 在dev中有一条样本有问题
    for idx, example in enumerate(tqdm(json_data)):
        id = example['id']
        question = example['question']
        context = example['context']
        
        if args.input_format == 'QC':
            text = f"""you are an extract machine, your task it to perform the following action:
            extract a short answer of the question from the given the context directly without any explanation.\
            Use the following format: Answer:<answer>. 
            Here is the Question and Context.
            Question:{question}.
            Context:{context}
            """
        
        elif args.input_format == 'QPBC':
            text = f"""you are an extract machine, your task it to perform the following action:
            extract a short answer of the question from the given the context directly without any explanation.\
            Use the following format: Answer:<answer>. 
            Here is the Question and Context.
            Question:{question}.Already known that {prompt_bridge[id]}.
            Context:{context}
            """
        
        LLM_input[id] = text
    
    with open(os.path.join(args.output_dir, f'{args.split}_{args.input_format}.json'), 'w') as outfile:
        json.dump(LLM_input, outfile, indent=4)  
            
