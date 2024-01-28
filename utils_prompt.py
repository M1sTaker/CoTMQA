'''
Adapted from https://github.com/lupantech/ScienceQA
'''

from dataclasses import dataclass
from typing import List, Optional

def get_ents_in_question(text):
    ents = text['np_in_question']
    return ents
    
def get_question_text(text):
    question = text["question"]
    return question\
        
def get_bridge_start(text):
    start = text["bridge_start"]
    return start

def get_bridge_end(text):
    end = text["bridge_end"]
    return end

def get_caption(text):
    caption = text["caption"]
    return caption

def get_context(text):
    context = text['context']
    return context

def create_train_examples(input_format, output_format, examples):
    # generate bridge answer
    if output_format == 'B':
        context = examples['caption']
        answers_start = examples["bridge_start"]
        answers_end = examples["bridge_end"]
        
        if input_format == 'ENTCAP':
            questions = examples["select_np"]    
        elif input_format == 'QCAP':
            questions = examples['question']
        elif input_format == 'PQCAP':
            questions = [prefix_prompt + Q + prompt for prefix_prompt, Q, prompt in zip(["Question:"] * len(examples['question']), examples['question'], [".what is the entity being asked about in the question?"] * len(examples['question']))]
    
    #generate final answer
    elif output_format == 'A':
        context = examples['context']
        answers_start = examples['answer_start']
        answers_end = examples['answer_end']
        
        if input_format == 'QBC':
            prompt_list = ["Known the entity in question is "] * len(examples['question'])
            dot_list = ['.'] * len(examples['question'])
            Q_marks = ['?'] * len(examples['question'])
            questions = [prompt + B + dot + Q + Q_mark for prompt, B, dot, Q, Q_mark in zip(prompt_list, examples['bridge'], dot_list, examples['question'], Q_marks)]
        
        # extended bridge + question + context
        elif input_format == 'EBQC':
            Q_marks = ['?'] * len(examples['question'])
            questions = [extended_bridge + Q + Q_mark for extended_bridge, Q, Q_mark in zip(examples['extended_bridge'], examples['question'], Q_marks)]
        
        # prompt bridge + question + context    
        elif input_format == 'PBQC':
            dot_list = ['.'] * len(examples['question'])
            Q_marks = ['?'] * len(examples['question'])
            is_list = [" is "] * len(examples['question'])
            
            questions = [q_bridge + _is + bridge + dot + Q + Q_mark for q_bridge, _is, bridge, dot, Q, Q_mark in zip(examples['q_bridge'], is_list, examples['bridge'], dot_list, examples['question'], Q_marks)]
        
        #  question + prompt bridge + context    
        elif input_format == 'QPBC':
            dot_list = ['.'] * len(examples['question'])
            Q_marks = ['?'] * len(examples['question'])
            is_list = [" is "] * len(examples['question'])
            
            questions = [Q + Q_mark + q_bridge + _is + bridge + dot for Q, Q_mark, q_bridge, _is, bridge, dot  in zip(examples['question'], Q_marks, examples['q_bridge'], is_list, examples['bridge'], dot_list)]
        
        # replaced question + context    
        elif input_format == 'RQC':
            Q_marks = ['?'] * len(examples['question'])
            questions = [replaced_question + Q_mark for replaced_question, Q_mark in zip(examples['generated_question'], Q_marks)]
        
        # one-hop reasoning
        elif input_format == 'QC':
            questions = examples['question']
        
        
            
    return context, answers_start, answers_end, questions

def create_validation_examples(input_format, output_format, examples, use_correct_bridge):
    if output_format == 'B':
        context = examples['caption']
        
        if input_format == 'ENTCAP':
            questions = examples["select_np"]    
        elif input_format == 'QESCAP':
            questions = examples['question']
        elif input_format == 'PQCAP':
            prefix_prompt_list = ["Question:"] * len(examples['question'])
            questions = [prefix_prompt + Q + prompt for prefix_prompt, Q, prompt in zip(prefix_prompt_list, examples['question'], [".what is the entity being asked about in the question?"] * len(examples['question']))]
            
    elif output_format == 'A':
        context = examples['context']
        
        if input_format == 'QBC':
            prompt_list = ["Known the entity in question is "] * len(examples['question'])
            dot_list = ['.'] * len(examples['question'])
            questions = [prompt + B + dot + Q for prompt, B, dot, Q in zip(prompt_list, examples['generated_bridge'], dot_list, examples['question'])]
        
        # extended bridge + question + context
        elif input_format == 'EBQC':
            questions = [extended_bridge + Q for extended_bridge, Q in zip(examples['extended_bridge'], examples['question'])]
        
        # prompt bridge + question + context  
        elif input_format == 'PBQC':
            dot_list = ['.'] * len(examples['question'])
            is_list = [" is "] * len(examples['question'])
            
            if use_correct_bridge:
                questions = [q_bridge + _is + bridge + dot + Q for q_bridge, _is, bridge, dot, Q in zip(examples['q_bridge'], is_list, examples['bridge'], dot_list, examples['question'])]
            else:
                questions = [q_bridge + _is + bridge + dot + Q for q_bridge, _is, bridge, dot, Q in zip(examples['q_bridge'], is_list, examples['generated_bridge'], dot_list, examples['question'])]
        
        # prompt bridge + question + context  
        elif input_format == 'QPBC':
            dot_list = ['.'] * len(examples['question'])
            is_list = [" is "] * len(examples['question'])
             
            questions = [ Q + q_bridge + _is + bridge + dot for Q, q_bridge, _is, bridge, dot in zip(examples['question'], examples['q_bridge'], is_list, examples['generated_bridge'], dot_list)]
        
        # replaced question + context    
        elif input_format == 'RQC':
            
            replaced_question = []
            for question, q_bridge, generated_bridge in zip(examples['question'], examples['q_bridge'], examples['generated_bridge']):
                replaced_question.append(question.replace(q_bridge, generated_bridge))
                
            questions = replaced_question
        
        # one-hop reasoning    
        elif input_format == 'QC':
            questions = examples['question']
            
    return context, questions

#region
# def get_context_text(problem, use_caption):
#     txt_context = problem['hint']
#     img_context = problem['caption'] if use_caption else ""
#     context = " ".join([txt_context, img_context]).strip()
#     if context == "":
#         context = "N/A"
#     return context


# def get_choice_text(probelm, options):
#     choices = probelm['choices']
#     choice_list = []
#     for i, c in enumerate(choices):
#         choice_list.append("({}) {}".format(options[i], c))
#     choice_txt = " ".join(choice_list)
#     #print(choice_txt)
#     return choice_txt

# def get_origin_answer(problem, options):
#     return problem['choices'][problem['answer']]

# def get_answer(problem, options):
#     return options[problem['answer']]


# def get_lecture_text(problem):
#     # \\n: GPT-3 can generate the lecture with more tokens.
#     lecture = problem['lecture'].replace("\n", "\\n")
#     return lecture


# def get_solution_text(problem):
#     # \\n: GPT-3 can generate the solution with more tokens
#     solution = problem['solution'].replace("\n", "\\n")
#     return solution
#endregion


def create_one_example(format, question, context, choice, answer, lecture, solution, test_example=True, WithOutput = False, curr_le_data=None):

    input_format, output_format = format.split("-")

    ## Inputs
    if input_format == "CQM":
        input = f"Context: {context}\nQuestion: {question}\nOptions: {choice}\n"
    elif input_format == "QCM":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\n"
    elif input_format == "QM":
        input = f"Question: {question}\nOptions: {choice}\n"
    elif input_format == "QC":
        input = f"Question: {question}\nContext: {context}\n"
    elif input_format == "QCMG":
        if curr_le_data is not None:
            input = f"Question: {question}\nContext: {context}\nOptions: {choice}\n{curr_le_data}\n"
        else:
            input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nSolution: {lecture} {solution}\n"
    elif input_format == "CQMG":
        if curr_le_data is not None:
            input = f"Context: {context}\nQuestion: {question}\nOptions: {choice}\n{curr_le_data}\n"
        else:
            input = f"Context: {context}\nQuestion: {question}\nOptions: {choice}\nSolution: {lecture} {solution}\n"
    # upper bound experiment
    elif input_format == "QCML":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture}\n"
    elif input_format == "QCME":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {solution}\n"
    elif input_format == "QCMLE":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture} {solution}\n"

    elif input_format == "QCLM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture}\nOptions: {choice}\n"
    elif input_format == "QCEM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {solution}\nOptions: {choice}\n"
    elif input_format == "QCLEM":
        input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture} {solution}\nOptions: {choice}\n"
    elif input_format == "QCMA":
        input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nAnswer: The answer is {answer}.\n"
    elif input_format == "QCA":
        input = f"Question: {question}\nContext: {context}\nAnswer: The answer is {answer}. \nBECAUSE:"

    # Outputs
    if test_example:
        if output_format == 'A':
            output = "Answer:"
        elif output_format == 'E':
            output = "Solution:"
        else:
            output = "Solution:"
    elif output_format == 'A':
        output = f"Answer: The answer is {answer}."

    elif output_format == 'AL':
        output = f"Answer: The answer is {answer}. BECAUSE: {solution}"
    elif output_format == 'AE':
        output = f"Answer: The answer is {answer}. BECAUSE: {lecture}"
    elif output_format == 'ALE':
        output = f"Answer: The answer is {answer}. BECAUSE: {lecture} {solution}"
    elif output_format == 'AEL':
        output = f"Answer: The answer is {answer}. BECAUSE: {solution} {lecture}"

    elif output_format == 'LA':
        output = f"Answer: {lecture} The answer is {answer}."
    elif output_format == 'EA':
        output = f"Answer: {solution} The answer is {answer}."
    elif output_format == 'LEA':
        output = f"Answer: {lecture} {solution} The answer is {answer}."
    elif output_format == 'ELA':
        output = f"Answer: {solution} {lecture} The answer is {answer}."

    elif output_format == 'LE':
        output = f"Solution: {lecture} {solution}."

    elif output_format == 'E':
        output = f"Solution: {solution}"
        
    
    if WithOutput:
        if output.endswith("BECAUSE:"):
            output = output.replace("BECAUSE:", "").strip()
        if output_format == 'E':
            text = input + f'Solution:'
        elif output_format == 'A':
            text = input + f'Answer:'
        else: 
            text = input + f'Solution:'
        text = text.replace("  ", " ").strip()
        output = output.replace("  ", " ").strip()
        return text, output
        
        
    text = input + output
    text = text.replace("  ", " ").strip()
    if text.endswith("BECAUSE:"):
        text = text.replace("BECAUSE:", "").strip()
    return text

def create_promt_input(format, question, ent, caption, context, test_example=True, WithOutput = False, curr_le_data=None):

    input_format, output_format = format.split("-")

    ## Inputs
    if input_format == "ENTCAP":
        input = f"Find Entity: {ent}. Caption: {caption}\n"
            
    # # upper bound experiment
    # elif input_format == "QCML":
    #     input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture}\n"
    # elif input_format == "QCME":
    #     input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {solution}\n"
    # elif input_format == "QCMLE":
    #     input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nBECAUSE: {lecture} {solution}\n"

    # elif input_format == "QCLM":
    #     input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture}\nOptions: {choice}\n"
    # elif input_format == "QCEM":
    #     input = f"Question: {question}\nContext: {context}\nBECAUSE: {solution}\nOptions: {choice}\n"
    # elif input_format == "QCLEM":
    #     input = f"Question: {question}\nContext: {context}\nBECAUSE: {lecture} {solution}\nOptions: {choice}\n"
    # elif input_format == "QCMA":
    #     input = f"Question: {question}\nContext: {context}\nOptions: {choice}\nAnswer: The answer is {answer}.\n"
    # elif input_format == "QCA":
    #     input = f"Question: {question}\nContext: {context}\nAnswer: The answer is {answer}. \nBECAUSE:"

    # Outputs
    # if test_example:
    #     if output_format == 'A':
    #         output = "Answer:"
    #     elif output_format == 'E':
    #         output = "Solution:"
    #     else:
    #         output = "Solution:"
    # elif output_format == 'A':
    #     output = f"Answer: The answer is {answer}."

    # elif output_format == 'AL':
    #     output = f"Answer: The answer is {answer}. BECAUSE: {solution}"
    # elif output_format == 'AE':
    #     output = f"Answer: The answer is {answer}. BECAUSE: {lecture}"
    # elif output_format == 'ALE':
    #     output = f"Answer: The answer is {answer}. BECAUSE: {lecture} {solution}"
    # elif output_format == 'AEL':
    #     output = f"Answer: The answer is {answer}. BECAUSE: {solution} {lecture}"

    # elif output_format == 'LA':
    #     output = f"Answer: {lecture} The answer is {answer}."
    # elif output_format == 'EA':
    #     output = f"Answer: {solution} The answer is {answer}."
    # elif output_format == 'LEA':
    #     output = f"Answer: {lecture} {solution} The answer is {answer}."
    # elif output_format == 'ELA':
    #     output = f"Answer: {solution} {lecture} The answer is {answer}."

    # elif output_format == 'LE':
    #     output = f"Solution: {lecture} {solution}."

    # elif output_format == 'E':
    #     output = f"Solution: {solution}"
        
    
    # if WithOutput:
    #     if output.endswith("BECAUSE:"):
    #         output = output.replace("BECAUSE:", "").strip()
    #     if output_format == 'E':
    #         text = input + f'Solution:'
    #     elif output_format == 'A':
    #         text = input + f'Answer:'
    #     else: 
    #         text = input + f'Solution:'
    #     text = text.replace("  ", " ").strip()
    #     output = output.replace("  ", " ").strip()
    #     return text, output
        
        
    # text = input + output
    # text = text.replace("  ", " ").strip()
    # if text.endswith("BECAUSE:"):
    #     text = text.replace("BECAUSE:", "").strip()
    # return text
    
    return input


def build_prompt(problems, shot_qids, test_qid, args):

    examples = []

    # n-shot training examples
    for qid in shot_qids:
        question = get_question_text(problems[qid])
        context = get_context_text(problems[qid], args.use_caption)
        choice = get_choice_text(problems[qid], args.options)
        answer = get_answer(problems[qid], args.options)
        lecture = get_lecture_text(problems[qid])
        solution = get_solution_text(problems[qid])

        train_example = create_one_example(args.prompt_format,
                                           question,
                                           context,
                                           choice,
                                           answer,
                                           lecture,
                                           solution,
                                           test_example=False)
        examples.append(train_example)

    # test example
    question = get_question_text(problems[test_qid])
    context = get_context_text(problems[test_qid], args.use_caption)
    choice = get_choice_text(problems[test_qid], args.options)
    answer = get_answer(problems[test_qid], args.options)
    lecture = get_lecture_text(problems[test_qid])
    solution = get_solution_text(problems[test_qid])

    test_example = create_one_example(args.prompt_format,
                                      question,
                                      context,
                                      choice,
                                      answer,
                                      lecture,
                                      solution,
                                      test_example=True)
    examples.append(test_example)

    # create the prompt input
    prompt_input = '\n\n'.join(examples)

    return prompt_input

def build_train_pair(text, args, curr_le_data=None):

    #region
    # examples = []

    # # test example
    # question = get_question_text(problems[test_qid])
    # context = get_context_text(problems[test_qid], args.use_caption)
    # choice = get_choice_text(problems[test_qid], args.options)
    
    # lecture = get_lecture_text(problems[test_qid])
    # solution = get_solution_text(problems[test_qid])

    # # answer_text = get_origin_answer(problems[test_qid], args.options)
    # answer_option = get_answer(problems[test_qid], args.options)
    # answer = "(" + answer_option + ")"
    
    
    
    # test_example, target = create_one_example(args.prompt_format,
    #                                   question,
    #                                   context,
    #                                   choice,
    #                                   answer,
    #                                   lecture,
    #                                   solution,
    #                                   test_example=False,WithOutput = True, curr_le_data=curr_le_data)
    # examples.append(test_example)
    #endregion
    
    #create example
    question = get_question_text(text)
    caption = get_caption(text)
    context = get_context(text)
    
    ents_in_question = get_ents_in_question(text)
    selected_ent = ents_in_question[0]
    
    bridge_start = get_bridge_start(text)
    bridge_end = get_bridge_end(text)
    
    prompt_input = create_promt_input(args.prompt_format, question, selected_ent, caption, context)
    target_position = {'start': bridge_start, 'end': bridge_end}
    
    
    # target = target.replace("Answer:", "").strip()
    # # create the prompt input
    # prompt_input = '\n\n'.join(examples)

    # return prompt_input, target
    return prompt_input, target_position

@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    le_input_ids: List[List[int]]
    le_attention_mask: Optional[List[List[int]]]
    le_token_type_ids: Optional[List[List[int]]]
    label: Optional[int]