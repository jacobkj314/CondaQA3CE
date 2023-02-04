import jsonlines 

train_splits_dir = "few-shot/train_splits/sampling_c_shots_9_cot/"
our_dir = "few-shot/train_splits/sampling_c_shots_9_cot/spreadsheet/"

for split in list(range(1,6)):
    out_file = []
    train_split_file = f'{train_splits_dir}conda_fewshot_train_{split}.json'

    with jsonlines.open(train_split_file, 'r') as reader: 
        for item in reader: 
            out_str = f"{item['original cue']}\t{item['sentence1']}\t{item['sentence2']}\t{item['label']}\n"
            out_file.append(out_str)
        with open(f"{our_dir}conda_fewshot_train_{split}", "w") as fout:
            for item in out_file:
                fout.write(item)

