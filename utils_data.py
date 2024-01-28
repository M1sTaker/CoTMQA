import os
from torch.utils.data import Dataset
import os
import json
import numpy as np
import torch
from utils_prompt import *
import datasets

img_shape = {
    "resnet": (512, 2048),
    "clip": (49, 2048),
    "detr": (100, 256),
    "vit": (577, 768),
    "instblip": (52, 768),
    "blip2": (32, 768),
}

def add_imgfeature(dataset, image_features):
    image_features_list = [image_features.numpy()[i] for i in range(image_features.shape[0])]
    
    dataset_imgfeature = datasets.Dataset.from_dict({"image_ids": image_features_list})
    dset_concat = datasets.concatenate_datasets([dataset, dataset_imgfeature], axis=1)    
    
    return dset_concat

# In final answer stage, add generated_bridge for eval and test
def map_generated_bridge(example, generated_bridges):
    example['generated_bridge'] = generated_bridges.get(example["id"], None)
    return example

# In final answer stage, add extended_bridge for train eval and test
def map_extended_bridge(example, extended_bridges):
    example['extended_bridge'] = extended_bridges.get(example["id"], None)
    return example

# In final answer stage, add q_bridge for train eval and test
def map_q_bridge(example, q_bridges):
    example['q_bridge'] = q_bridges.get(example["id"], None)
    return example

def add_generated_bridge(dataset, bridge_dir):
    with open(bridge_dir, 'r') as json_file:
        generated_bridge = json.load(json_file)
    # add generated_bridge    
    dataset = dataset.map(lambda example: map_generated_bridge(example, generated_bridge))
    
    return dataset

def add_extended_bridge(dataset, bridge_dir):
    with open(bridge_dir, 'r') as json_file:
        extended_bridge = json.load(json_file)
    # add extended_bridge    
    dataset = dataset.map(lambda example: map_extended_bridge(example, extended_bridge))
    
    return dataset

def add_q_bridge(dataset, bridge_dir):
    with open(bridge_dir, 'r') as json_file:
        q_bridge = json.load(json_file)
    # add q_bridge    
    dataset = dataset.map(lambda example: map_q_bridge(example, q_bridge))
    
    return dataset


def load_data_std(args):
    train_examples = datasets.Dataset.from_json(os.path.join(args.data_root, 'train/preprocessed_data', 'new_train.json'))
    eval_examples = datasets.Dataset.from_json(os.path.join(args.data_root, 'eval', 'dev.json'))
    test_examples = datasets.Dataset.from_json(os.path.join(args.data_root, 'eval', 'test.json'))
    
    train_examples.rename_column("id", "example_id")
    eval_examples.rename_column("id", "example_id")
    test_examples.rename_column("id", "example_id")
    
    train_image_features, eval_image_features, test_image_features = None, None, None
    
    return train_examples, eval_examples, test_examples, train_image_features, eval_image_features, test_image_features

def load_data_img(args):
    train_examples = datasets.Dataset.from_json(os.path.join(args.data_root, 'train/preprocessed_data', 'new_train.json'))
    eval_examples = datasets.Dataset.from_json(os.path.join(args.data_root, 'eval', 'dev.json'))
    test_examples = datasets.Dataset.from_json(os.path.join(args.data_root, 'eval', 'test.json'))
    
    train_examples.rename_column("id", "example_id")
    eval_examples.rename_column("id", "example_id")
    test_examples.rename_column("id", "example_id")

    # check
    if args.img_type == "resnet":
        image_features = np.load('vision_features/resnet.npy')
        image_features = np.expand_dims(image_features, axis=1)
        image_features = image_features.repeat(512, axis=1)
    elif args.img_type == "clip":
        image_features = np.load('vision_features/clip.npy')
    elif args.img_type == "detr":
        # image_features = np.load('vision_features/detr.npy')
        train_image_features = torch.load(os.path.join(args.data_root, 'train/vision_features', 'detr_train.pth'))
        eval_image_features = torch.load(os.path.join(args.data_root, 'eval/vision_features', 'detr_dev.pth'))
        test_image_features = torch.load(os.path.join(args.data_root, 'eval/vision_features', 'detr_test.pth'))
    elif args.img_type == "vit":
        image_features = np.load('vision_features/vit.npy')
    elif args.img_type == "blip2":
        image_features = torch.load('vision_features/blip2.pth')
    elif args.img_type == "instblip":
        image_features = torch.load('vision_features/instblip.pth')
    else:
        image_features = np.load('vision_features/detr.npy')
    print("img_features size: ", train_image_features.shape)

    # region
    # for qid in problems:
    #     problems[qid]['caption'] = captions[qid] if qid in captions else ""
    # train_qids = pid_splits['%s' % (args.train_split)]
    # val_qids = pid_splits['%s' % (args.val_split)]
    # test_qids = pid_splits['%s' % (args.test_split)]
    # print(f"number of train problems: {len(train_qids)}\n")
    # print(f"number of val problems: {len(val_qids)}\n")
    # print(f"number of test problems: {len(test_qids)}\n")
    # qids = {'train': train_qids, 'val':val_qids,'test':test_qids}
    # endregion
    return train_examples, eval_examples, test_examples, train_image_features, eval_image_features, test_image_features
        
# class MuMuQADatasetImg(Dataset):
#     """
#     Creating a custom dataset for reading the dataset and
#     loading it into the dataloader to pass it to the
#     neural network for finetuning the model

#     """

#     def __init__(
#         self, texts, tokenizer, args, image_features, test_le=None
#     ):
#         """
#         Initializes a Dataset class

#         Args:
#             dataframe (pandas.DataFrame): Input dataframe
#             tokenizer (transformers.tokenizer): Transformers tokenizer
#             source_text (str): column name of source text
#             target_text (str): column name of target text
#         """
#         self.tokenizer = tokenizer
#         self.texts = texts
#         self.image_features = image_features
#         self.args = args
#         self.max_len = args.max_len
        
#         self.source_texts = []
#         self.target_positions = {'start':[], 'end':[]}
        
#         curr_le_data = None
        
#         for idx, text in enumerate(texts):
#             source_text, target_position = build_train_pair(text, args, curr_le_data)
#             self.source_texts.append(source_text)
#             self.target_positions['start'].append(target_position['start'])
#             self.target_positions['end'].append(target_position['end'])
        
#         #region
#         # self.target_text = []
#         # self.source_text = []
        
#         # if test_le is not None:
#         #     test_le_data =json.load(open(test_le))["preds"]
#         # else:
#         #     test_le_data = None
#         # idx = 0
#         # for qid in self.data:
#         #     if test_le_data is not None:
#         #         curr_le_data = test_le_data[idx]
#         #         idx += 1
#         #     else:
#         #         curr_le_data = None
#         #     prompt, target = build_train_pair(problems, qid, args, curr_le_data)
#         #     self.target_text.append(target)
#         #     self.source_text.append(prompt)
#         #     if str(qid) in name_maps:
#         #         i_vectors = image_features[int(name_maps[str(qid)])]
#         #         self.image_ids.append(i_vectors)
#         #     else:
#         #         shape = img_shape[args.img_type]
#         #         self.image_ids.append(np.zeros(shape))
#         #endregion
        
                
    
#     def __len__(self):
#         """returns the length of dataframe"""

#         return len(self.text_data)

#     def __getitem__(self, index):
#         """return the input ids, attention masks and target ids"""

#         source_text = str(self.source_texts[index])
#         start_position = self.target_positions['start'][index]
#         end_position = self.target_positions['end'][index]
        
#         image_ids = self.image_features[index]

#         # cleaning data so as to ensure data is in string type
#         source_text = " ".join(source_text.split())
#         # target_text = " ".join(target_text.split())

#         source = self.tokenizer.batch_encode_plus(
#             [source_text],
#             max_length=self.source_len,
#             pad_to_max_length=True,
#             truncation=True,
#             padding="max_length",
#             return_tensors="pt",
#         )
#         target = self.tokenizer.batch_encode_plus(
#             [target_text],
#             max_length=self.summ_len,
#             pad_to_max_length=True,
#             truncation=True,
#             padding="max_length",
#             return_tensors="pt",
#         )
#         source_ids = source["input_ids"].squeeze()
#         source_mask = source["attention_mask"].squeeze()
#         target_ids = target["input_ids"].squeeze().tolist()

#         image_ids = torch.tensor(image_ids).squeeze()
        
        
        
#         return {
#             "input_ids": source_ids,
#             "attention_mask": source_mask,
#             "image_ids": image_ids,
#             "labels": target_ids,
    # }        

#evaluate in batches
class ScienceQADatasetIterator:
    def __init__(self, dataset, eval_iter_batch_size):
        self._dataset = dataset
        self.batch_size =  eval_iter_batch_size
        self.num_batches = int(len(self._dataset) / self.batch_size)
        if len(self._dataset) % self.batch_size:
            self.num_batches += 1
    
    def __iter__(self):
        self._index = 0
        return self
    
    def __next__(self):
        if self._index < self.num_batches:
            items = []
            for i in range(self.batch_size):
                try:
                    index = (self._index * self.batch_size) + i
                    items.append(self._dataset.__getitem__(index))
                except IndexError:
                    break
            self._index += 1
            return items
        else:
            raise StopIteration

def evaluate_in_batches(eval_iter_batch_size, dataset, trainer, args):
    test_set_iterator = ScienceQADatasetIterator(dataset=dataset, eval_iter_batch_size=eval_iter_batch_size)
    
    eval_metrics = {}
    for batch in test_set_iterator:
        batch_metrics = trainer.evaluate(eval_dataset = batch)
        for key, value in batch_metrics.items():
            if key not in eval_metrics:
                eval_metrics[key] = []
            eval_metrics[key].append(value)
    for key, value in eval_metrics.items():
        eval_metrics[key] = sum(value) / len(value)
    
    return eval_metrics

def predict_in_batches(predict_iter_batch_size, dataset, trainer, args):
    test_set_iterator = ScienceQADatasetIterator(dataset=dataset, eval_iter_batch_size=predict_iter_batch_size)
    
    # predicts_ls = []
    for i, batch in enumerate(test_set_iterator):
        #output the prediction of i batch
        if i == 0:
            batch_predictions = trainer.predict(test_dataset=batch, max_length=args.output_len)
            batch_index = i
        
        # if i == 0:
        #     predict_results = batch_predictions
        # else:
        #     #concat the prediction
        #     pred1 = list(predict_results.predictions)
        #     pred2 = list(batch_predictions.predictions)
        #     pred1[0] = np.append(pred1[0], pred2[0], axis=0)
        #     pred1[1] = np.append(pred1[1], pred2[1], axis=0)
            
        #     #concat the labels
        #     label1 = predict_results.label_ids
        #     label2 = batch_predictions.label_ids 
        #     label1 = np.append(label1, label2, axis=0)
            
        #     predict_results = predict_results._replace(predictions=tuple(pred1))
        #     predict_results = predict_results._replace(label_ids=label1)
    
    # print(len(dataset))
    
    return batch_predictions, batch_index

def split_test_qids_in_batches(test_qids, args):
    num_batches = int(len(test_qids) / args.predict_iter_batch_size)
    if len(test_qids) % args.predict_iter_batch_size:
        num_batches += 1
    
    part_lengths = []    
    for i in range(num_batches-1):
        part_lengths.append(args.predict_iter_batch_size)
    
    if len(test_qids) % args.predict_iter_batch_size:
        part_lengths.append(len(test_qids) % args.predict_iter_batch_size)
    else:
        part_lengths.append(args.predict_iter_batch_size)
        
    split_test_qids = []
    start_index = 0
    for length in part_lengths:
        split_test_qids.append(test_qids[start_index:start_index+length])
        start_index += length
    
    return split_test_qids
    
    