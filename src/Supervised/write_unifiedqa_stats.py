import json
import glob
import os


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="./results/", help="dir to store results")
    parser.add_argument("--model_name", type=str, default="unifiedqa-v2-t5-base-1251000", help="dir to store results")
    args = parser.parse_args()
    return args


def parse_lines(lines):
    print(lines)
    accuracy=lines[0].split("Accuracy = ")[1]
    consistency=lines[1].split("Consistency = ")[1]
    p_o_consistency=lines[2].split("Paraphrase-Original Consistency = ")[1]
    sc_o_consistency=lines[3].split("Scope-Original Consistency = ")[1]
    aff_o_consistency=lines[4].split("Affirmative-Original Consistency = ")[1]

    return accuracy, consistency, p_o_consistency, sc_o_consistency, aff_o_consistency

def read_data(filename):
    f= open(filename, "r")
    all_lines = f.read().split("\n")
    print(filename)

    accuracy, consistency, p_o_consistency, sc_o_consistency, aff_o_consistency= parse_lines(all_lines)
    return accuracy, consistency, p_o_consistency, sc_o_consistency, aff_o_consistency


def write_results(filename, model, metrics, metric_list):
    f= open(filename,"w")
    f.write("MODEL")
    for metric in metric_list:
        f.write(","+metric+"_avg")
        f.write(","+metric+"_std")
    f.write("\n")
    f.write(model)
    for metric in metric_list:
        f.write(","+str(round(metrics[metric+"_avg"],2))+",")
        f.write(str(round(metrics[metric + "_std"],2)))
    f.close()




def compute_stats(metrics, metric_list):
    for metric in metric_list:
        print(metric)
        print(metric_list)
        scores=metrics[metric]

        assert len(scores)==5
        mean = sum(scores) / len(scores)
        variance = sum([((x - mean) ** 2) for x in scores]) / len(scores)
        std = variance ** 0.5

        import pdb
        pdb.set_trace()

        metrics[metric+"_avg"]=mean
        metrics[metric + "_std"] = std
    return metrics


def main(args):
    RESULTS_DIR= args.results_dir #"./results/"
    model=args.model_name
    #MODELS = ["unifiedqa-v2-t5-base-1251000", "unifiedqa-v2-t5-large-1251000", "unifiedqa-v2-t5-3b-1251000"]
    SEEDS = ["70", "69", "68", "67", "66"]


    # Evaluate all dev checkpoints

    # for model in MODELS:
    metrics={"accuracy":[],"consistency":[], "p_o_consistency":[], "sc_o_consistency":[], "aff_o_consistency":[]}
    metric_list = list(metrics.keys())
    for seed in SEEDS:
        accuracy, consistency, p_o_consistency, sc_o_consistency, aff_o_consistency = read_data(RESULTS_DIR+model+"_"+seed+".txt")
        metrics["accuracy"].append(float(accuracy))
        metrics["consistency"].append(float(consistency))
        metrics["p_o_consistency"].append(float(p_o_consistency))
        metrics["sc_o_consistency"].append(float(sc_o_consistency))
        metrics["aff_o_consistency"].append(float(aff_o_consistency))

    metrics=compute_stats(metrics, metric_list)
    write_results(RESULTS_DIR+model+"_aggregatestats.txt",model, metrics, metric_list)





if __name__ == "__main__":
    args = get_args()
    main(args)
# os.system("")

# Calculate stats

