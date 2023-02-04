import json
import jsonlines
import argparse
import numpy
import random
import os


def make_few_shot_data(output_dir, train_file, test_file, n_splits, n_question_groups, sampling_strategy, n_shots, gpt3_version):
    filenames = {'train': train_file, 'test': test_file}

    for split in ['train', 'test']:
        data = []
        with open(filenames[split]) as f:
            for line in f:
                data.append(json.loads(line))

        all_passage_ids = [x["PassageID"] for x in data]

        if split == "train":  # groups with 3 questions about 4 (minimally different) passages
            split_passage_ids = [x["PassageID"] for x in data if all_passage_ids.count(x["PassageID"]) == 12]
        else:  # all instances
            split_passage_ids = [x["PassageID"] for x in data]
        split_passage_ids = list(set(split_passage_ids))

        passage_id_to_samples = {}
        for each_sample in data:
            if each_sample["PassageID"] in split_passage_ids:
                passage_id = each_sample["PassageID"]
                edit_id = each_sample['PassageEditID']
                question_id = each_sample["QuestionID"]
                if passage_id not in passage_id_to_samples:
                    passage_id_to_samples[passage_id] = {}

                if not (edit_id == 3 and sampling_strategy in ['B', 'D'] and split == "train"):
                    if edit_id not in passage_id_to_samples[passage_id]:
                        passage_id_to_samples[passage_id][edit_id] = {}

                    passage_id_to_samples[passage_id][edit_id][question_id] = each_sample


        random.shuffle(split_passage_ids)
        print (f"============> Number of {split} passages = {len(split_passage_ids)}")
        passage_ids_to_splits = numpy.array_split(numpy.array(split_passage_ids), n_splits)
        passage_ids_to_splits_list = [list(l) for l in passage_ids_to_splits]

        for split_id in range(n_splits):
            fewshot_samples = []
            if split == "train":
                if sampling_strategy in ['C', 'D']:
                    sampled_passage_ids = passage_ids_to_splits_list[split_id][:n_shots]
                    for passage_id in sampled_passage_ids:
                        # Randomly choose one question
                        qset = random.choice(list(passage_id_to_samples[passage_id][0].keys()))

                        # Randomly choose one edit type (with an option of choosing from only non-affirmative edit types)
                        edit_id = random.sample(range(4), 1)[0] if sampling_strategy == 'C' else random.sample(range(3), 1)[0]
                        fewshot_samples.append(passage_id_to_samples[passage_id][edit_id][qset])
                else:
                    sampled_passage_ids = passage_ids_to_splits_list[split_id][:n_question_groups]
                    for passage_id in sampled_passage_ids:
                        # Randomly choose one question
                        qset = random.choice(list(passage_id_to_samples[passage_id][0].keys()))

                        # Iterate through types of passages
                        for edit_id in passage_id_to_samples[passage_id].keys():
                            fewshot_samples.append(passage_id_to_samples[passage_id][edit_id][qset])
            else:
                sampled_passage_ids = passage_ids_to_splits_list[split_id]
                for passage_id in sampled_passage_ids:
                    for edit_id in passage_id_to_samples[passage_id].keys():
                        for q_set_idx in passage_id_to_samples[passage_id][edit_id]:
                            fewshot_samples.append(passage_id_to_samples[passage_id][edit_id][q_set_idx])

            print(f"============> Num examples for {split}_split_#{split_id + 1} = {len(fewshot_samples)}")
            if not os.path.exists(f"{output_dir}{gpt3_version}/{sampling_strategy}/"):
                os.makedirs(f"{output_dir}{gpt3_version}/{sampling_strategy}/")
                        
            out_filename = f"{output_dir}{gpt3_version}/{sampling_strategy}/conda_fewshot_{split}_{split_id + 1}.json"
            with jsonlines.open(out_filename, mode='w') as writer:
                writer.write_all(fewshot_samples)
                writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, default="./condaqa_train.json")  # TODO: change this
    parser.add_argument("--test_file", type=str, default="./condaqa_test.json")
    parser.add_argument("--output_dir", type=str, default="./few-shot/")
    parser.add_argument("--n_splits", type=int, default=5, help="the number of train-dev splits to use for eval")
    parser.add_argument("--n_question_groups", type=int, default=3,
                        help="a question group is 1 question x 4 minimally diff passages; we include args.n_question_groups such groups in the train context")
    parser.add_argument("--n_shots", type=int, default=9, help="needed for certain values of sampling_strategy")                      
    parser.add_argument("--sampling_strategy", type=str, default="A", help="A: args.n_question_groups x 1 question x 4 paragraphs (original, affirmative, paraphrase, scope)\
                                                                            B: args.n_question_groups x 1 question x 3 paragraphs (original, paraphrase, scope) aka no affirmative\
                                                                            C: args.n_shots random question about args.n_shots different passages\
                                                                            D: args.n_shots random question about args.n_shots different passages with negation (no affirmative)")
    parser.add_argument("--gpt3_version", type=str, help="davinci or text-davinci-002")
    args = parser.parse_args()

    make_few_shot_data(args.output_dir, args.train_file, args.test_file, args.n_splits, args.n_question_groups, args.sampling_strategy, args.n_shots, args.gpt3_version)
