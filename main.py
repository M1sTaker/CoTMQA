import os
import numpy as np
import torch
import os
import re
import json
import argparse
import random
import datasets
import collections
from tqdm.auto import tqdm

from utils.trainer_qa import QuestionAnsweringTrainer
from transformers import (
    RobertaTokenizerFast, RobertaForQuestionAnswering,
    DebertaV2TokenizerFast, DebertaV2ForQuestionAnswering,
    BertTokenizerFast, BertForQuestionAnswering,
    AutoTokenizer, AutoModelForQuestionAnswering,
    EvalPrediction, TrainingArguments,
)
from model import RobertaForMultimodalQuestionAnswering, BertForMultimodalQuestionAnswering
from datasets import Dataset

from utils_data import img_shape, load_data_std, load_data_img, add_imgfeature, add_generated_bridge, add_extended_bridge, add_q_bridge
from utils_prompt import create_train_examples, create_validation_examples
from utils_qa import postprocess_qa_predictions
from mmqa_eval import read_answers, read_predictions, mmqa_evaluate

from rich.table import Column, Table
from rich import box
from rich.console import Console
console = Console(record=True)
# import nltk
import evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='MuMuQA')
    parser.add_argument('--output_dir', type=str, default='experiments')
    parser.add_argument('--model', type=str, default="pretrained_models/roberta-base-squad2")
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--max_text_len', type=int, default=512)
    parser.add_argument('--max_answer_len', type=int, default=30)
    parser.add_argument('--stride', type=int, default=128)
    parser.add_argument('--num_best_ans', type=int, default=20, help='number of generated top n answers')
    parser.add_argument('--fp16', action='store_true', help='use mix precision')
    parser.add_argument('--load_best_model', action='store_true', help='load best model at the end of training')
    
    parser.add_argument('--eval_bs', type=int, default=16)
    parser.add_argument('--eval_acc', type=int, default=None, help='evaluate accumulation step')
    # parser.add_argument('--eval_in_batch', action='store_true', help='split evalutation into several iterations due to high RAM occupation')
    # parser.add_argument('--predict_in_batch', action='store_true', help='split evalutation into several iterations due to high RAM occupation')
    # parser.add_argument('--eval_iter_batch_size', type=int, default=1500, help='split evalutation into several iterations due to high RAM occupation')
    # parser.add_argument('--predict_iter_batch_size', type=int, default=1500, help='split prediction into several iterations due to high RAM occupation')
    # parser.add_argument('--user_msg', type=str, default="baseline", help='experiment type in the save_dir')
    parser.add_argument('--small_dataset', action='store_true')
    parser.add_argument('--bridge_eval', action='store_true', help='evaluate bridge generation')
    parser.add_argument('--use_correct_bridge', action='store_true', help='use f1=100 bridge')
    parser.add_argument('--final_eval', action='store_true', help='only evaluate the model at the final epoch')
    parser.add_argument('--skip-no-answer', action='store_true')
    
    parser.add_argument('--img_type', type=str, default=None, choices=['detr', 'clip', 'resnet', 'vit'], help='type of image features')
    parser.add_argument('--eval_generated_bridge_le', type=str, default=None, help='generated bridge answer for the dev set')
    parser.add_argument('--test_generated_bridge_le', type=str, default=None, help='generated bridge answer for the test set')
    
    parser.add_argument('--eval_q_bridge_le', type=str, default=None, help='bridge phrase in question for the dev set')
    parser.add_argument('--test_q_bridge_le', type=str, default=None, help='bridge phrase in question for the test set')
    
    parser.add_argument('--train_extended_bridge_le', type=str, default=None, help='extended bridge answer for the train set')
    parser.add_argument('--eval_extended_bridge_le', type=str, default=None, help='extended bridge answer for the dev set')
    parser.add_argument('--test_extended_bridge_le', type=str, default=None, help='extended bridge answer for the test set')
    
    parser.add_argument('--evaluate_dir', type=str, default=None, help='the directory of model for evaluation')
    parser.add_argument('--prompt_format', type=str, default='ENTCAP-B', help='prompt format template',
                        choices=['ENTCAP-B', 'QCAP-B', 'PQCAP-B', 'QC-A', 'QBC-A', 'EBQC-A', 'PBQC-A', 'QPBC-A', 'RQC-A'])
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    args = parser.parse_args()
    return args
        
def MainTrainer(
    args,
    train_examples, eval_examples, test_examples,
    train_image_features, eval_image_features, test_image_features
):
    torch.manual_seed(args.seed)  # pytorch random seed
    np.random.seed(args.seed)  # numpy random seed
    torch.backends.cudnn.deterministic = True
    
    if args.evaluate_dir is not None:
        args.model = args.evaluate_dir

    #prepare tokenizer
    if "roberta" in args.model:
        tokenizer = RobertaTokenizerFast.from_pretrained(args.model)
    elif "bert-large" in args.model:
        tokenizer = BertTokenizerFast.from_pretrained(args.model)
    elif "deberta" in args.model:
        tokenizer = DebertaV2TokenizerFast.from_pretrained(args.model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)

    # console.log(f"""[Model]: Loading {args.model}...\n""")
    # console.log(f"[Data]: Reading data...\n")
    
    if args.evaluate_dir is not None:
        save_dir = args.evaluate_dir
    else:
        model_name = args.model.replace("/","-")
        gpu_count = torch.cuda.device_count()
        save_dir = f"{args.output_dir}/{model_name}_{args.img_type}_{args.prompt_format}_lr{args.lr}_bs{args.bs * gpu_count}_ep{args.epoch}_maxtextlen{args.max_text_len}_stride{args.stride}"
        if args.fp16:
            save_dir += "_fp16"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    
    def preprocess_training_examples(examples):
        context, answers_start, answers_end, questions = create_train_examples(input_format, output_format, examples)     
            
        inputs = tokenizer(
            questions,
            context,
            max_length=args.max_text_len,
            truncation="only_second",
            stride=args.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        # answers = examples["answers"]
        start_positions = []
        end_positions = []
        image_ids = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            # answer = answers[sample_idx]
            start_char = answers_start[sample_idx]
            end_char = answers_end[sample_idx]
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label is (0, 0)
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)
            
            if args.img_type is not None:     
                image_ids.append(examples["image_ids"][sample_idx])

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        if args.img_type is not None:
            inputs["image_ids"] = image_ids
        return inputs

    def preprocess_validation_examples(examples):
        context, questions = create_validation_examples(input_format, output_format, examples, args.use_correct_bridge)
            
        inputs = tokenizer(
            questions,
            context,
            max_length=args.max_text_len,
            truncation="only_second",
            stride=args.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids = []
        image_ids = []

        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(examples["id"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]
            
            if args.img_type is not None:
                image_ids.append(examples["image_ids"][sample_idx])

        inputs["example_id"] = example_ids
        if args.img_type is not None:
            inputs["image_ids"] = image_ids
        return inputs
    
    if args.img_type is not None:
        patch_size = img_shape[args.img_type]
        
        #prepare model
        if "roberta" in args.model:
            model = RobertaForMultimodalQuestionAnswering.from_pretrained(args.model, patch_size=patch_size)
        elif "bert-large" in args.model:
            model = BertForMultimodalQuestionAnswering.from_pretrained(args.model, patch_size=patch_size)
            
        # model = RobertaForMultimodalQuestionAnswering.from_pretrained(args.model, patch_size=patch_size)
        
        print("model parameters: ", model.num_parameters())
        
        #use small dataset for test
        if args.small_dataset:
            train_examples = train_examples.select(range(16))
            train_image_features = train_image_features[:16, :, :]
            
            eval_examples = eval_examples.select(range(16))
            eval_image_features = eval_image_features[:16, :, :]
            
            test_examples = test_examples.select(range(16))
            test_image_features = test_image_features[:16, :, :]
        
        #use extended bridge    
        if args.train_extended_bridge_le is not None:
            train_examples = add_extended_bridge(train_examples, args.train_extended_bridge_le)
        
        #preprocess train examples
        train_examples = add_imgfeature(train_examples, train_image_features)
        train_dataset = train_examples.map(
            preprocess_training_examples,
            batched=True,
            remove_columns=train_examples.column_names,
        )    
        train_dataset.set_format('torch')
        
        # if eval for final answer
        if output_format == 'A':
            if args.eval_extended_bridge_le is not None:
                eval_examples = add_extended_bridge(eval_examples, args.eval_extended_bridge_le)
            if args.eval_generated_bridge_le is not None:
                eval_examples = add_generated_bridge(eval_examples, args.eval_generated_bridge_le)
            if args.eval_q_bridge_le is not None:
                eval_examples = add_q_bridge(eval_examples, args.eval_q_bridge_le)
                
        #preprocess eval examples    
        eval_examples = add_imgfeature(eval_examples, eval_image_features)     
        eval_dataset = eval_examples.map(
            preprocess_validation_examples,
            batched=True,
            remove_columns=eval_examples.column_names,
        )  
        eval_dataset.set_format('torch')
        
        # if test for final answer
        if output_format == 'A':
            if args.test_extended_bridge_le is not None:
                test_examples = add_extended_bridge(test_examples, args.test_extended_bridge_le)
            if args.test_generated_bridge_le is not None:
                test_examples = add_generated_bridge(test_examples, args.test_generated_bridge_le)
            if args.test_generated_bridge_le is not None:
                test_examples = add_q_bridge(test_examples, args.test_q_bridge_le)
            
        # preprocess test examples    
        test_examples = add_imgfeature(test_examples, test_image_features)      
        test_dataset = test_examples.map(
            preprocess_validation_examples,
            batched=True,
            remove_columns=test_examples.column_names,
        )
        test_dataset.set_format('torch')       
        
    elif args.img_type is None:
        #prepare model
        if "roberta" in args.model:
            model = RobertaForQuestionAnswering.from_pretrained(args.model)
        elif "bert-large" in args.model:
            model = BertForQuestionAnswering.from_pretrained(args.model)
        elif "deberta" in args.model:
            model = DebertaV2ForQuestionAnswering.from_pretrained(args.model)
        else:
            model = AutoModelForQuestionAnswering.from_pretrained(args.model)
            
        # model = RobertaForQuestionAnswering.from_pretrained(args.model)
        
        print("model parameters: ", model.num_parameters())
        
        #use small dataset for test
        if args.small_dataset:
            train_examples = train_examples.select(range(16))
            
        train_dataset = train_examples.map(
            preprocess_training_examples,
            batched=True,
            remove_columns=train_examples.column_names,
        )    
        train_dataset.set_format('torch')
        
        # eval for final answer
        if output_format == 'A':
            if args.eval_extended_bridge_le is not None:
                eval_examples = add_extended_bridge(eval_examples, args.eval_extended_bridge_le)
            if args.eval_generated_bridge_le is not None:
                eval_examples = add_generated_bridge(eval_examples, args.eval_generated_bridge_le)
            if args.eval_q_bridge_le is not None:
                eval_examples = add_q_bridge(eval_examples, args.eval_q_bridge_le)
                       
        eval_dataset = eval_examples.map(
            preprocess_validation_examples,
            batched=True,
            remove_columns=eval_examples.column_names,
        )  
        eval_dataset.set_format('torch')
        
        # test for final answer
        if output_format == 'A':
            if args.test_extended_bridge_le is not None:
                test_examples = add_extended_bridge(test_examples, args.test_extended_bridge_le)
            if args.test_generated_bridge_le is not None:
                test_examples = add_generated_bridge(test_examples, args.test_generated_bridge_le)
            if args.test_generated_bridge_le is not None:
                test_examples = add_q_bridge(test_examples, args.test_q_bridge_le)
                 
        test_dataset = test_examples.map(
            preprocess_validation_examples,
            batched=True,
            remove_columns=test_examples.column_names,
        )
        test_dataset.set_format('torch') 

    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=False,
            n_best_size=args.num_best_ans,
            max_answer_length=args.max_answer_len,
            output_dir=save_dir,
            prefix=stage,
            args=args,
        )
        # Format the result to the format the metric expects.
        version_2_with_negative=False
        if version_2_with_negative:
            formatted_predictions = [
                {"id": str(k), "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [{"id": str(k), "prediction_text": v} for k, v in predictions.items()]

        if output_format == 'B':
            answer_column_name = "bridge"
        elif output_format == 'A':
            answer_column_name = "answer"
            
        references = [{"id": str(ex["id"]), "answers": ex[answer_column_name]} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)
    
    metric = evaluate.load('mmqa_metric')
    
    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    training_args = TrainingArguments(
            save_dir,
            do_train=True if args.evaluate_dir is None else False,
            do_eval=True,
            evaluation_strategy="epoch",
            logging_strategy="steps",
            save_strategy="epoch",
            save_total_limit = 1,
            load_best_model_at_end = args.load_best_model,
            learning_rate= args.lr,
            eval_accumulation_steps=args.eval_acc,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.eval_bs,
            weight_decay=0.01,
            num_train_epochs=args.epoch,
            report_to="none",
            metric_for_best_model="f1",
            fp16=args.fp16,
        )

    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        eval_examples=eval_examples,
        tokenizer=tokenizer,
        post_process_function=post_processing_function,
        compute_metrics = compute_metrics,
    )

    if args.evaluate_dir is None:
        trainer.train()
        # trainer.save_model(save_dir)
    
    # catch ValueError
    try:
        predictions = trainer.predict(predict_dataset=eval_dataset, predict_examples=eval_examples, file_prefix="eval")
    except(ValueError):
        pass
    
    try:
        predictions = trainer.predict(predict_dataset=test_dataset, predict_examples=test_examples, file_prefix="test")
    except(ValueError):
        pass
    
    if trainer.is_world_process_zero():
        eval_data_path = os.path.join(args.data_root, 'eval/dev.json')
        test_data_path = os.path.join(args.data_root, 'eval/test.json')
        
        eval_answers = read_answers(eval_data_path, args.bridge_eval)
        test_answers = read_answers(test_data_path, args.bridge_eval)
        
        eval_predictions = read_predictions(os.path.join(save_dir, 'eval_predictions.json'))
        test_predictions = read_predictions(os.path.join(save_dir, 'test_predictions.json'))
        
        eval_metrics = mmqa_evaluate(eval_answers, eval_predictions, args.skip_no_answer)
        test_metrics = mmqa_evaluate(test_answers, test_predictions, args.skip_no_answer)
        
        metrics = {'eval_metric':eval_metrics, 'test_metric': test_metrics}
        
        print("metric: ", json.dumps(metrics))
        
        metrics_file = os.path.join(save_dir, 'metrics.json')
        with open(metrics_file, "w") as writer:
            writer.write(json.dumps(metrics, indent=4))            
    

if __name__ == '__main__':
    # training logger to log training progress
    # training_logger = Table(
    #     Column("Epoch", justify="center"),
    #     Column("Steps", justify="center"),
    #     Column("Loss", justify="center"),
    #     title="Training Status",
    #     pad_edge=False,
    #     box=box.ASCII,
    # )
    
    args = parse_args()
    print("args",args)
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    random.seed(args.seed)
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    if args.img_type is not None:
        train_examples, eval_examples, test_examples, train_image_features, eval_image_features, test_image_features = load_data_img(args)
    else:
        train_examples, eval_examples, test_examples, train_image_features, eval_image_features, test_image_features = load_data_std(args)
    
    input_format, output_format = args.prompt_format.split("-")   

    MainTrainer(
        args = args,
        train_examples=train_examples, eval_examples=eval_examples, test_examples=test_examples,
        train_image_features=train_image_features, eval_image_features=eval_image_features, test_image_features=test_image_features
    )
