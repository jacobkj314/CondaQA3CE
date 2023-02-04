import json
import jsonlines
# import tensorflow as tf
import re
import glob

def write_data(data, filename):
    with jsonlines.open(filename, mode='w') as writer:
        writer.write_all(data)
        writer.close()


train_splits_folder="./few-shot/train_splits/sampling_c_shots_9_cot/"
test_splits_folder="./few-shot/test_splits/"

for data_split in [train_splits_folder, test_splits_folder]: 
    for filename in glob.glob(data_split+"*"):
        data = []
        with open(filename) as f:
            for line in f:
                data.append(json.loads(line))

        samples = []
        for sample in data:
            p = sample["sentence1"].replace("\n", "")
            passage = f'Passage: {p}' + '\\n' 
            q = sample["sentence2"].replace("\n", "")
            question = f'Question: {q}' + '\\n' 
            #task_description = "Answer the following yes/no/don't know question by reasoning step-by-step.\\n"
            prompt = "Give the rationale before answering."
            if 'test' in test_splits_folder:
                answer = sample["label"]
            else:
                answer = f'{sample["explanation"]} So the answer is {sample["label"]}.'
            sample_new = {"input": passage + question + prompt, 
                          "answer": answer
                         }
            samples.append(sample_new)
        

        OUTPUT_DIR = "./few-shot/flan_formatted_data/"
        out_filename = OUTPUT_DIR + data_split.split("few-shot/")[-1] + filename.split("/")[-1]
        print(out_filename)

        write_data(samples, out_filename)
