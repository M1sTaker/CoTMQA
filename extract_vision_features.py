import torch
from PIL import Image
import torchvision.transforms as T
from transformers import AutoImageProcessor, ViTModel
import numpy as np
import os
import argparse
import json
from tqdm import tqdm
# from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration, Blip2Model, Blip2Processor, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='MuMuQA/train/train_images')
    parser.add_argument('--output_dir', type=str, default='MuMuQA/train/vision_features')
    parser.add_argument('--img_type', type=str, default="vit", choices=['detr', 'vit', 'blip2', 'instblip'], help='type of image features')
    args = parser.parse_args()
    return args

def extract_features(img_type, input_image, idx, problem=None):
    if img_type == "vit":
        try:
            img = Image.open(input_image).convert("RGB")
        except(OSError):
            img = train_pixels[idx]
            img = Image.fromarray(img)
        inputs = vit_image_processor(img, return_tensors="pt")
        
        with torch.no_grad():
            outputs = vit_model(**inputs)
        return outputs.last_hidden_state
    
    elif img_type == "detr":
        transform = T.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        with torch.no_grad():
            try:
                img = Image.open(input_image).convert("RGB")
            except(OSError):
                img = train_pixels[idx]
                img = Image.fromarray(img)
            input = transform(img).unsqueeze(0)
            feature = detr_model(input)[-1]
        return feature
    
    elif img_type == "blip2":
        with torch.no_grad():
            img = Image.open(input_image).convert("RGB")
            inputs = processor(images=img, return_tensors='pt').to(device)
            qformer_outputs = blip2_model.get_qformer_features(**inputs)
        return qformer_outputs.last_hidden_state
    
    elif img_type == "instblip":
        with torch.no_grad():
            img = Image.open(input_image).convert("RGB")
            
            text = problem['question']
            for choice in problem['choices']:
                text = text + choice + ','
            
            inputs = processor(images=img, text=text, return_tensors='pt')
            outputs = instblip_model(**inputs, decoder_input_ids=inputs["input_ids"])
            feature = outputs.qformer_outputs.last_hidden_state
            feature = feature.squeeze()
        return feature

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '7'
    args = parse_args()
    print("args",args)
    all_images = os.listdir(args.data_root)
    tmp = []
    name_map = {}
    # all_images.sort(key=lambda x:int(x))
    print(len(all_images))
    
    train_pixels = np.load('/home/fengqiyuan/Dataset/MuMuQA/train/concatenated_pixel_data_train.npy')    

    if args.img_type == "vit":
        vit_image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        vit_model.eval()
    elif args.img_type == "detr":
        # detr_model = torch.hub.load('cooelf/detr', 'detr_resnet101_dc5', pretrained=True)
        detr_model = torch.hub.load('/home/fengqiyuan/.cache/torch/hub/cooelf_detr_main', 'detr_resnet101_dc5', pretrained=True, source='local')
        detr_model.eval()
    elif args.img_type == "blip2":
        blip2_model = Blip2Model.from_pretrained("Salesforce/blip2-flan-t5-xl")
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        blip2_model.to(device)
        blip2_model.eval()
    elif args.img_type == "instblip":
        instblip_model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl")
        processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
        instblip_model.to(device)
        instblip_model.eval()
    for idx, image in enumerate(tqdm(all_images)):
        if idx == 3: break
        if os.path.exists(os.path.join(args.data_root, image)):
            curr_dir = os.path.join(args.data_root, image)
        else:
            curr_dir = os.path.join(args.data_root, image, "choice_0.png")
        # problem = problems[image]
        feature = extract_features(args.img_type, curr_dir, idx)
        tmp.append(feature.detach().cpu())
        name_map[str(image)] = idx
    
    if args.img_type == "instblip":
        torch.save(tmp, os.path.join(args.output_dir, args.img_type +'.pth'))
        # tmp_np = [x.numpy() for x in tmp]
        # np.save(os.path.join(args.output_dir, args.img_type +'.npy'), tmp_np)
    else:
        res = torch.cat(tmp).cpu()
        print(res.shape)
        torch.save(res, os.path.join(args.output_dir, args.img_type +'.pth'))
        
    # with open(os.path.join(args.output_dir, 'name_map.json'), 'w') as outfile:
    #     json.dump(name_map, outfile)