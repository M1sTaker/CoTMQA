from datasets import Dataset
import evaluate
from transformers import RobertaTokenizer, RobertaTokenizerFast, EvalPrediction, TrainingArguments, RobertaForQuestionAnswering
import os
import numpy as np
from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image
import requests
import spacy
import stanza

def get_phrases(tree, label):
    if tree.is_leaf():
        return []
    results = [] 
    for child in tree.children:
        results += get_phrases(child, label)
    
    
    if tree.label == label:
        return [' '.join(tree.leaf_labels())] + results
    else:
        return results

pixels = np.load('/root/autodl-tmp/fqy/Code/MQA/MuMuQA/eval/dev.npy')    
# image = Image.fromarray(pixels[27])
image = Image.fromarray(pixels[0])


# text = "In Afghanistan, the Taliban released to the media this picture, which it said shows the suicide bombers who attacked the army base in Mazar-i-Sharif, April 21, 2017."
# question = "How many of the people in the image had bombs?"

text = "Palestinian President Mahmoud Abbas, right, is greeted by Hanan Ashrawi, left, legislator and activist, as he arrives at his hotel in New York, Monday, Sept. 19, 2011, to attend the 66th General Assembly session of United Nations."
question = "Who did the person on the left of the image berate?"
# text = question

nlp = spacy.load('en_core_web_lg')
stanza_nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency', dir='/root/autodl-tmp/fqy/cache/stanza_resources', download_method=None)

doc = nlp(text)
stanza_doc = stanza_nlp(text)

ents = [ent.text for sent in doc.sents for ent in sent.noun_chunks] 
ents += [ent.text  for sent in doc.sents for ent in sent.ents]
ents += [phrase for sent in stanza_doc.sentences for phrase in get_phrases(sent.constituency, 'NP')]
# ents += [phrase for sent in stanza_doc.sentences for phrase in get_phrases(sent.constituency, 'VP')]
# ents += [word.text for sent in stanza_doc.sentences for word in sent.words if word.upos in ['VERB','ADV','ADJ','NOUN']]

ents = list(set(ents))

print("all ents:", ents)
print("nums of ents:", len(ents))

ckpt = "/root/autodl-tmp/fqy/Code/MQA/pretrained_models/clip-vit-large-patch14"

model = CLIPModel.from_pretrained(ckpt)
processor = CLIPProcessor.from_pretrained(ckpt)

inputs = processor(text=ents, images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
# print(logits_per_image)
probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

topk_values, topk_indices = torch.topk(probs, k=5, dim=1)

# 将topk_indices转换为Python列表
topk_indices = topk_indices.squeeze().tolist()

# 使用topk_indices来索引ents
selected_ents = [ents[i] for i in topk_indices]

print("selected_ents:", selected_ents)

# print(probs)



