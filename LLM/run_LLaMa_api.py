import json
import os
from tqdm import tqdm
import argparse
from llamaapi import LlamaAPI
import requests

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='/root/autodl-tmp/fqy/Code/MQA/LLM/LLM_input')
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--output_dir', type=str, default='/root/autodl-tmp/fqy/Code/MQA/LLM/LLaMa_output')
    parser.add_argument('--input_format', type=str, default='QPBC', help='prompt format template',choices=['QC', 'QBC', 'PBQC', 'QPBC', 'RQC'])
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    
    args = parse_args()
    print("args",args)
    
    with open(os.path.join(args.data_root, f'{args.split}_{args.input_format}.json'), 'r', encoding='utf-8') as file:
        inputs = json.load(file)
    
    # Initialize the llamaapi with your api_token
    llama = LlamaAPI("LL-Afw22QqBIyALT1N7FrPT7XtX84o6xTLnNZMADOttQxHS6hgayte7sTMQygk05ryp")
    
    LLM_output = {}
    
    for id, text in tqdm(inputs.items()):
        
        # Define your API request
        api_request_json = {
            "model": "llama-70b-chat",
            "messages": [
                {"role": "user", "content": f"{text}"},
            ],
            "max_tokens": 500
        }
        
        # Make your request and handle the response
        try:
            response = llama.run(api_request_json)
            response_json = response.json()
            
            content = response_json["choices"][0]["message"]["content"]
        except(json.decoder.JSONDecodeError, requests.exceptions.JSONDecodeError):
            content = "Error"
        
        LLM_output[id] = content
        
    with open(os.path.join(args.output_dir, f'{args.split}_{args.input_format}_prediction.json'), 'w') as outfile:
        json.dump(LLM_output, outfile, indent=4)  
            
