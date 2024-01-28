from datasets import Dataset
import evaluate
from transformers import RobertaTokenizer, RobertaTokenizerFast, EvalPrediction, TrainingArguments, RobertaForQuestionAnswering
import os
import numpy as np
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
import torch
from PIL import Image
import requests
import spacy
import stanza
import sys
sys.path.append('/root/autodl-tmp/fqy/Code/MQA/pretrained_models/GroundingDINO/groundingdino/util')

ckpt = "pretrained_models/instructblip-flan-t5-xl"

model = InstructBlipForConditionalGeneration.from_pretrained(ckpt)
processor = InstructBlipProcessor.from_pretrained(ckpt)

device = "cuda:0"
model.to(device)

input_image = "Obama.bmp"

pixels = np.load('/root/autodl-tmp/fqy/Code/MQA/MuMuQA/eval/dev.npy')    
image = Image.fromarray(pixels[27])

image = Image.open(input_image).convert("RGB")

text = "Caption:In Afghanistan, the Taliban released to the media this picture, which it said shows the suicide bombers who attacked the army base in Mazar-i-Sharif, April 21, 2017.Base on the caption, briefly list the entity in the image."

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

generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
print(generated_text)