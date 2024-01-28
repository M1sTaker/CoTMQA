import argparse
from PIL import Image
import os
import json
import numpy as np
from tqdm import tqdm
import torch
from torchvision.ops import box_convert
from typing import Tuple, List
from sklearn.cluster import KMeans

from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
from transformers import CLIPModel, CLIPProcessor, CLIPConfig, CLIPTextConfig, CLIPVisionConfig

from mmqa_eval import read_answers, read_predictions, mmqa_evaluate
import sys

def intersection_over_union(box_a: torch.Tensor, box_b: torch.Tensor) -> float:
    # 计算两个边界框的坐标
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    
    # 计算交集的面积
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    
    # 计算并集的面积
    box_a_area = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    box_b_area = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)
    union_area = box_a_area + box_b_area - intersection_area
    
    # 计算交并比
    iou = intersection_area / union_area
    
    return iou

def find_same_entity_boxes(boxes, threshold=0.5):
    num_boxes = len(boxes)
    
    # 计算交并比矩阵
    iou_matrix = torch.zeros((num_boxes, num_boxes))
    for i in range(num_boxes):
        for j in range(num_boxes):
            iou_matrix[i, j] = intersection_over_union(boxes[i], boxes[j])
    
    # 使用K均值聚类
    kmeans = KMeans(n_clusters=num_boxes)
    kmeans.fit(iou_matrix)
    
    # 获取聚类结果
    labels = kmeans.labels_
    
    # 根据聚类结果分组边界框
    entity_boxes = {}
    for i in range(num_boxes):
        label = labels[i]
        if label not in entity_boxes:
            entity_boxes[label] = []
        entity_boxes[label].append(boxes[i])
    
    return entity_boxes

def crop_and_save(image_source: np.ndarray, boxes: torch.Tensor, logits: torch.Tensor, phrases: List[str]) -> List[Tuple[np.ndarray, str, float]]:
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

    cropped_images = []
    for bbox, phrase, logit in zip(xyxy, phrases, logits):
        x1, y1, x2, y2 = bbox.astype(int)
        cropped_image = image_source[y1:y2, x1:x2]
        cropped_images.append((cropped_image, phrase, logit.item()))
    
    if len(cropped_images) == 0:
        cropped_images.append((image_source, "None", 1.0))

    return cropped_images

# 使用CLIP进行图文匹配
def get_matched_caption_ents(ents, image, k=1):
    inputs = CLIP_processor(text=ents, images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = CLIP_model(**inputs)
        
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score

    probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

    topk_values, topk_indices = torch.topk(probs, k=k, dim=1)

    # 将topk_indices转换为Python列表
    topk_indices = topk_indices.squeeze().tolist()

    # 使用topk_indices来索引ents
    if isinstance(topk_indices, list):
        selected_ents = [ents[i] for i in topk_indices]
    else:
        selected_ents = ents[topk_indices]
    
    return selected_ents

# 从得到bridge对应的问题中的短语/单词
def get_q_bridge_ent(question_ents, ent_from_DINO):
    
    if ent_from_DINO == 'None' or ent_from_DINO == '':
        # 如果返回的是None或者空串，则返回question_ents中最长的那个短语
        longest_ent = max(question_ents, key=lambda x: len(x.split()))
    else:
        # 寻找包含ent_from_DINO的最长的字符串
        ent_words = ent_from_DINO.split()  # 将ent_from_DINO中的字符串拆分成单词
        filtered_ents = [ent for ent in question_ents if all(word in ent.lower() for word in ent_words)]
        longest_ent = max(filtered_ents, key=len)
    
    return longest_ent


def bridge_evaluate(all_predictions, q_bridge_ents):
    data_path = os.path.join(args.data_root, f'new_{args.split}_{args.bridge_len}.json')
    answers = read_answers(data_path, bridge_eval=True)
    metrics = mmqa_evaluate(answers, all_predictions)
    
    print("metric: ", json.dumps(metrics))
    
    if args.save_result:
        
        output_dir = os.path.join(args.output_dir, f'box{args.box_threshold}_text{args.text_threshold}_{args.bridge_len}')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
        metrics_file = os.path.join(output_dir, f'{args.split}_metrics.json')
        prediction_file = os.path.join(output_dir, f'{args.split}_prediction.json')
        q_bridge_ents_file = os.path.join(output_dir, f'{args.split}_q_bridges.json')
        
        with open(metrics_file, "w") as writer:
            writer.write(json.dumps(metrics, indent=4))
            
        with open(prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4)) 
        
        with open(q_bridge_ents_file, "w") as writer:
            writer.write(json.dumps(q_bridge_ents, indent=4)) 

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='MuMuQA/eval/preprocessed_data')
    parser.add_argument('--output_dir', type=str, default='bridge_generation_result')
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--bridge_len', type=str, default='long')
    parser.add_argument('--box_threshold', type=float, default=0.4)
    parser.add_argument('--text_threshold', type=float, default=0.5)
    parser.add_argument('--save_result', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print("args",args)
    
    with open(os.path.join(args.data_root, f'new_{args.split}_{args.bridge_len}.json'), 'r', encoding='utf-8') as file:
        json_data = json.load(file)
        
    #load image 
    if args.split == 'train':
        image_pixels = np.load('/root/autodl-tmp/fqy/Code/MQA/MuMuQA/train/train.npy')
    elif args.split == 'dev':
        image_pixels = np.load('/root/autodl-tmp/fqy/Code/MQA/MuMuQA/eval/dev.npy')
    elif args.split == 'test':
        image_pixels = np.load('/root/autodl-tmp/fqy/Code/MQA/MuMuQA/eval/eval.npy')          
    
    DINO_model = load_model("/root/autodl-tmp/fqy/Code/MQA/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py", "/root/autodl-tmp/fqy/Code/MQA/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth")
    
    clip_ckpt = "/root/autodl-tmp/fqy/Code/MQA/pretrained_models/clip-vit-large-patch14"
    
    # text_config = CLIPTextConfig.from_pretrained(clip_ckpt)
    # text_config.max_position_embeddings = 128
    
    # vision_config = CLIPVisionConfig.from_pretrained(clip_ckpt)
    
    # config = CLIPConfig(text_config=text_config.to_dict(), vision_config=vision_config.to_dict())
    
    # CLIP_model = CLIPModel(config=config)
    
    CLIP_model = CLIPModel.from_pretrained(clip_ckpt)
    CLIP_processor = CLIPProcessor.from_pretrained(clip_ckpt)
    
    all_predictions = {}
    q_bridge_ents = {}
    BOX_TRESHOLD = args.box_threshold #0.35
    TEXT_TRESHOLD = args.text_threshold #0.25
    for idx, example in enumerate(tqdm(json_data)):

        image_pixel = image_pixels[idx]
        id = example['id']
        question_ents = example['selected_question_ents']
        caption_ents = example['selected_caption_ents']
        text = ".".join(question_ents)

        image_source, image = load_image(image_pixel)
        boxes, logits, phrases = predict(
            model=DINO_model,
            image=image,
            caption=text,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD,
            remove_combined = True,
        )
        
        cropped_images = crop_and_save(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        
        selected_image = cropped_images[0]
        
        # 通过CLIP图文匹配从caption_ents中选择概率最高的作为bridge
        bridge_ent = get_matched_caption_ents(caption_ents, Image.fromarray(selected_image[0]))
        # 选择bridge对应问题中的ent短语
        q_bridge_ent = get_q_bridge_ent(question_ents, selected_image[1])
        
        all_predictions[id] = bridge_ent
        q_bridge_ents[id] = q_bridge_ent
    
    bridge_evaluate(all_predictions, q_bridge_ents)
        
        