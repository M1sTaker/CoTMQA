from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pathlib import Path
from urllib.parse import urlparse
import argparse
import string
import re
import json
from collections import Counter

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def read_predictions(prediction_file):
    with open(prediction_file) as f:
        predictions = json.load(f)
    return predictions

def read_answers(gold_file, bridge_eval):
    answers = {}
    data = json.load(open(gold_file, "r"))
    for i, item in enumerate(data):
        if bridge_eval:
            answers[str(item["id"])] = item["bridge"]
        else:
            answers[str(item["id"])] = item["answer"]
    return answers

def mmqa_evaluate(answers, predictions, skip_no_answer=False):
    f1 = exact_match = total = 0
    for qid, ground_truths in answers.items():
        if qid not in predictions:
            if not skip_no_answer:
                message = 'Unanswered question %s will receive score 0.' % qid
                print(message)
                total += 1
            continue
        total += 1
        prediction = predictions[qid]
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(
            f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}

def find_fully_wrong(answers, predictions):
    
    fully_wrong_id = []
    fully_wrong_num = 0
    
    correct_id = []
    correct_num = 0
    
    for qid, ground_truths in answers.items():
        prediction = predictions[qid]
        f1 = metric_max_over_ground_truths(f1_score, prediction, ground_truths)
        
        if f1 == 0:
            fully_wrong_num += 1
            fully_wrong_id.append(qid)
        else:
            correct_num += 1
            correct_id.append(qid)
            
    return fully_wrong_num, fully_wrong_id, correct_num, correct_id

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation for MRQA Workshop Shared Task')
    parser.add_argument('--dataset_file', default='MuMuQA/eval/test.json' , type=str, help='Dataset File')
    parser.add_argument('--bridge_prediction_file', default='bridge_generation_result/box0.4_text0.5_short/test_prediction.json',  type=str, help='Prediction File')
    parser.add_argument('--prediction_file', default='experiments/roberta-base-squad2_None_QC-A_lr1e-07_bs8_ep10_maxtextlen512_stride128_fp16/test_predictions.json',  type=str, help='Prediction File')
    parser.add_argument('--skip-no-answer', action='store_true')
    args = parser.parse_args()

    # 读取桥接实体答案
    bridge_answers = read_answers(args.dataset_file, bridge_eval=True)
    bridge_predictions = read_predictions(args.bridge_prediction_file)
    
    # 找出完全错误
    bridge_fully_wrong_num, bridge_fully_wrong_id, bridge_correct_num, bridge_correct_id = find_fully_wrong(bridge_answers, bridge_predictions)
    
    print(args.prediction_file)
    print("=================================================================================================")
    print("桥接答案中完全错误的比例：", bridge_fully_wrong_num/len(bridge_answers))
    print("桥接答案中正确（包含部分正确）的比例：", bridge_correct_num/len(bridge_answers))
    print("=================================================================================================")
    # 读取最终答案
    answers = read_answers(args.dataset_file, bridge_eval=False)
    predictions = read_predictions(args.prediction_file)
    
    # 只保留第一步错误的情况
    for key in bridge_correct_id:
        if key in answers:
            del answers[key]
            del predictions[key]
    
    fully_wrong_num, fully_wrong_id, correct_num, correct_id = find_fully_wrong(answers, predictions)
    
    print("在第一步错误的情况下，第二步也错误的比例：", fully_wrong_num)
    print("在第一步错误的情况下，第二步正确的比例：", correct_num)
    print("=================================================================================================")
    
    # 读取最终答案
    answers = read_answers(args.dataset_file, bridge_eval=False)
    predictions = read_predictions(args.prediction_file)
    
    # 只保留第一步正确的情况
    for key in bridge_fully_wrong_id:
        if key in answers:
            del answers[key]
            del predictions[key]
        
    fully_wrong_num, fully_wrong_id, correct_num, correct_id = find_fully_wrong(answers, predictions)
    
    print("在第一步正确的情况下，第二步也正确的比例：", correct_num)
    print("在第一步正确的情况下，第二步错误的比例：", fully_wrong_num)
    print("=================================================================================================")
    
    # print(json.dumps(metrics))