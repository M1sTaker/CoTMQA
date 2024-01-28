import os
import json

def merge_json_files(input_folder, output_file):
    def merge_dicts(dict1, dict2):
        for key, value in dict2.items():
            if key == "num_fail":
                dict1[key] = dict1.get(key, 0) + value
            if key in dict1:
                if isinstance(dict1[key], list) and isinstance(value, list):
                    dict1[key].extend(value)
                elif isinstance(dict1[key], dict) and isinstance(value, dict):
                    merge_dicts(dict1[key], value)
                elif isinstance(dict1[key], float) and isinstance(value, float):
                    dict1[key] += value
            else:
                dict1[key] = value
                
    def calculate_average(dict_data):
        for key, value in dict_data.items():
            if isinstance(value, dict):
                calculate_average(value)
            elif isinstance(value, float):
                if key != "num_fail":
                    dict_data[key] /= num_files

    merged_data = {}
    num_files=0
    for filename in os.listdir(input_folder):
        if filename.endswith('.json'):
            file_path = os.path.join(input_folder, filename)
            with open(file_path, 'r') as file:
                data = json.load(file)
                num_files += 1
                merge_dicts(merged_data, data)
    
    calculate_average(merged_data)

    with open(output_file, 'w') as output:
        json.dump(merged_data, output, indent=4)

# def merge_test(input_folder, output_file):
#     merged_data = {}
#     for filename in os.listdir(input_folder):
#         if filename.endswith('.json'):
#             file_path = os.path.join(input_folder, filename)
#             with open(file_path, 'r') as file:
#                 data = json.load(file)
#                 for key, value in data.items():
#                     if key in merged_data:
#                         merged_data[key].extend(value)
#                     else:
#                         merged_data[key] = value

#     with open(output_file, 'w') as output:
#         json.dump(merged_data, output)

if __name__ == '__main__':
    model_path = 'experiments/rationale_allenai-unifiedqa-t5-base_blip2_QCM-LE_lr5e-05_bs16_op512_ep20'
    
    input_eval = os.path.join(model_path, 'predictions_ans_eval')
    input_test = os.path.join(model_path, 'predictions_ans_test')
    
    output_eval = os.path.join(model_path, 'predictions_ans_eval.json')
    output_test = os.path.join(model_path, 'predictions_ans_test.json')
    
    merge_json_files(input_eval, output_eval)
    merge_json_files(input_test, output_test)