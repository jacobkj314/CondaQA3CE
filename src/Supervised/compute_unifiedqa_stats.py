import json
import glob
import os


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../../data/unifiedqa_formatted_data/", help="dir with test files to evaluate unifiedqa") #TODO: change this
    parser.add_argument("--results_dir", type=str, default="./results/", help="dir to store results")
    parser.add_argument("--predictions_dir", type=str, default="./predictions/", help="dir with all the checkpoints")
    parser.add_argument("--model_name", type=str, default="unifiedqa-v2-t5-base-1251000")
    parser.add_argument("--validation_filename", type=str, default="../../data/condaqa_dev.json")
    parser.add_argument("--test_filename", type=str, default="../../data/condaqa_test.json")
    parser.add_argument("--seed", type=str, default="70")
    parser.add_argument("--output_dir", type=str, default="./")
    args = parser.parse_args()
    return args


def read_data(filename):
    data = []
    with open(filename) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def write_results(filename, accuracy, consistency, pp_c, scope_c, aff_c):
    print(filename)
    print(f"============> Accuracy = {accuracy}")
    print(f"============> Consistency = {consistency}")
    print(f"============> Paraphrase-Original Consistency = {pp_c}")
    print(f"============> Scope-Original Consistency = {scope_c}")
    print(f"============> Affirmative-Original Consistency = {aff_c}")
    f = open(filename, "w")
    f.write(f"Accuracy = {accuracy}\n")
    f.write(f"Consistency = {consistency}\n")
    f.write(f"Paraphrase-Original Consistency = {pp_c}\n")
    f.write(f"Scope-Original Consistency = {scope_c}\n")
    f.write(f"Affirmative-Original Consistency = {aff_c}\n")
    f.close()


def compute_accuracy(pred_file, data_file, label_key="label"):
    gold_data = read_data(data_file)
    predictions = open(pred_file).readlines()
    assert len(predictions) == len(gold_data)

    met = [gold_l[label_key].strip().lower() == pred_l.strip().lower() for gold_l, pred_l in
           zip(gold_data, predictions)]
    accuracy = sum(met) * 1.0 / len(met) * 100
    print (accuracy)
    return accuracy


def get_groups(gold_data):
    groups = {}
    all_questions = [str(x["PassageID"]) + "_" + str(x["QuestionID"]) for x in gold_data]

    # To compute consistency, we need the question to have had answers that were agreed upon by all
    # crowdworkers, for all edits that were made to the passage
    consistency_subset = [ind for ind, x in enumerate(gold_data) if
                          all_questions.count(str(x["PassageID"]) + "_" + str(x["QuestionID"])) == 4]

    # Forms a group of all samples corresponding to one question, and its answers
    # for 4 different types of passages
    for ind in consistency_subset:
        x = gold_data[ind]
        passage_id = x["PassageID"]
        if passage_id not in groups:
            groups[passage_id] = {}

        passage_edit = x["PassageEditID"]
        question_id = x["QuestionID"]
        if question_id not in groups[passage_id]:
            groups[passage_id][question_id] = {}
        groups[passage_id][question_id][passage_edit] = {"index": ind, "sample": x}

    # Sanity check
    for passage_id in groups:
        for question_id in groups[passage_id]:
            assert len(groups[passage_id][question_id].keys()) == 4

    return groups, consistency_subset


def compute_group_score(pred_answers, gold_answers):
    assert len(pred_answers) == len(gold_answers)
    incorrect = 0 # # #
    for ind in range(len(gold_answers)):
        if pred_answers[ind].lower().strip() != gold_answers[ind].lower().strip():
            incorrect += 1 # # #
            '''return 0
    return 1'''
    return (len(gold_answers) - incorrect)/len(gold_answers) # # # Changing the score formulation 


def compute_consistency(pred_file, data_file, label_key="label"):
    gold_data = read_data(data_file)
    predictions = open(pred_file).readlines()
    groups, consistency_subset = get_groups(gold_data)

    consistency_dict = {x: {"correct": 0, "total": 0, "consistency": 0} for x in ["all", "0-1", "0-2", "0-3"]}

    for passage_id in groups:
        for question in groups[passage_id]:
            group = groups[passage_id][question]

            # Compute overall consistency
            all_gold_answers = [group[edit_id]["sample"][label_key] for edit_id in range(4)]
            all_predictions = [predictions[group[edit_id]["index"]] for edit_id in range(4)]

            consistency_dict["all"]["correct"] += compute_group_score(all_predictions, all_gold_answers)
            consistency_dict["all"]["total"] += 1

            # Compute consistency for each edit type
            og_passage_key = 0
            for contrast_edit in range(1, 4):
                all_gold_answers = [group[og_passage_key]["sample"][label_key],
                                    group[contrast_edit]["sample"][label_key]]
                all_predictions = [predictions[group[og_passage_key]["index"]],
                                   predictions[group[contrast_edit]["index"]]]
                consistency_dict["0-" + str(contrast_edit)]["correct"] += compute_group_score(all_predictions,
                                                                                              all_gold_answers)
                consistency_dict["0-" + str(contrast_edit)]["total"] += 1

    for key in consistency_dict:
        consistency_dict[key]["consistency"] = consistency_dict[key]["correct"] * 100.0 / consistency_dict[key]["total"]

    return consistency_dict["all"]["consistency"], consistency_dict["0-1"]["consistency"], consistency_dict["0-2"][
        "consistency"], consistency_dict["0-3"]["consistency"]


def evaluate_checkpoints(MODEL_NAME, SEED, validation_filename, checkpoint_dir="./predictions/"):
    best_checkpoints = {}

    SETTING = "unifiedqa"
    TEST_FILE = "unifiedqa"
    filepath = checkpoint_dir + MODEL_NAME + "_negation_all_" + SEED + "_train_" + SETTING + "_test_" + TEST_FILE

    best_checkpoints[(MODEL_NAME, SEED)] = []
    for checkpoint in glob.glob(filepath + "/checkpoint*"):
        filepath = checkpoint + "/val_predictions/"
        pred_file = filepath + "generated_predictions.txt"
        accuracy = compute_accuracy(pred_file, validation_filename, "label")
        print(checkpoint)
        print(accuracy)
        best_checkpoints[(MODEL_NAME, SEED)].append((checkpoint, accuracy))
    return best_checkpoints




def main(args):
    DATA_DIR = args.data_dir #"../../data/unifiedqa_formatted_data/"
    RESULTS_DIR= args.results_dir #"./results/"
    validation_filename = args.validation_filename#"../../data/condaqa_dev.json"
    test_filename = args.test_filename #"../../data/condaqa_test.json"
    PREDICTIONS_DIR = args.predictions_dir

    MODEL_NAME = args.model_name#["unifiedqa-v2-t5-base-1251000", "unifiedqa-v2-t5-large-1251000", "unifiedqa-v2-t5-3b-1251000"]
    SEED = args.seed #["70", "69", "68", "67", "66"]


    # Evaluate all dev checkpoints

    best_checkpoints=evaluate_checkpoints(MODEL_NAME, SEED, validation_filename, PREDICTIONS_DIR)
    
    # Pick best model

    best_checkpoints[(MODEL_NAME, SEED)].sort(key=lambda x: x[1])
    best_checkpoint = best_checkpoints[(MODEL_NAME, SEED)][-1][0]  # Gets name of best checkpoint

    os.system("mkdir -p " + best_checkpoint + "/test_predictions")

    OUTPUT_DIR = best_checkpoint

    # Evaluate on test set

    test_command = "python run_negatedqa_t5.py \
    --model_name_or_path {OUTPUT_DIR} \
    --train_file {DATA_DIR}condaqa_train_unifiedqa.json \
    --validation_file {DATA_DIR}condaqa_dev_unifiedqa.json \
    --test_file {DATA_DIR}condaqa_test_unifiedqa.json \
    --do_eval \
    --do_predict \
    --predict_with_generate \
    --per_device_train_batch_size 8 \
    --learning_rate 1e-5 \
    --num_train_epochs 5 \
    --output_dir {OUTPUT_DIR}/test_predictions \
    --logging_strategy epoch\
    --evaluation_strategy epoch\
    --report_to wandb\
    --save_strategy epoch\
    --overwrite_cache\
    --seed {SEED}\
    --summary_column answer \
    --text_column input \
    --source_prefix ''\
    --max_source_length 1024\
    --max_target_length 100\
    --overwrite_output_dir > {OUTPUT_DIR}/test_predictions/{MODEL_NAME}_results_all_{SEED}_train_{SETTING}_test_{TEST_FILE}_{checkpoint}.txt".format(
        OUTPUT_DIR=OUTPUT_DIR, DATA_DIR=DATA_DIR, MODEL_NAME=MODEL_NAME, SEED=SEED, SETTING="unifiedqa",
        TEST_FILE="unifiedqa", checkpoint=best_checkpoint.split("/")[-1])


    # Run predictions on test
    print(test_command)
    os.system(test_command)

    # Make directory for results

    os.system("mkdir -p {RESULTS_DIR}".format(RESULTS_DIR=RESULTS_DIR))
    filepath = best_checkpoint + "/test_predictions/"
    pred_file = filepath + "generated_predictions.txt"

    # Compute results

    accuracy = compute_accuracy(pred_file, test_filename, "label")
    consistency, pp_c, scope_c, aff_c = compute_consistency(pred_file, test_filename, "label")

    write_results(RESULTS_DIR+MODEL_NAME+"_"+SEED+".txt", accuracy, consistency, pp_c, scope_c, aff_c)




if __name__ == "__main__":
    args = get_args()
    main(args)
# os.system("")

# Calculate stats

