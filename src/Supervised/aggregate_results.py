import numpy 
seeds = [70] #[66, 67, 68, 69, 70]
model_sizes = ['large'] #['base', 'large', '3b']

for model_size in model_sizes:
    results_all_seeds = []
    for seed in seeds:
        fname = f'results/unifiedqa-v2-t5-{model_size}-1251000_{seed}.txt'
        results = open(fname, "r").readlines()
        results_clean = [float(line.split('= ')[1].split('\n')[0]) for line in results]
        results_all_seeds.append(results_clean)

    fout_name = f'results/unifiedqa-v2-t5-{model_size}-1251000_aggregate'
    with open(fout_name, "w") as fout: 
        header_str = '\t'.join(['Accuracy Mean', 'Accuracy Std', 'Consistency Mean', 'Consistency Std', 
                                'Paraphrase-Original Consistency Mean', 'Paraphrase-Original Consistency Std',
                                'Scope-Original Consistency Mean', 'Scope-Original Consistency Std',
                                'Affirmative-Original Consistency Mean', 'Affirmative-Original Consistency Std']) +'\n'
        accuracy_list = [item[0] for item in results_all_seeds]
        consistency_list = [item[1] for item in results_all_seeds]
        pp_c_list = [item[2] for item in results_all_seeds]
        scope_c_list = [item[3] for item in results_all_seeds]
        aff_c_list = [item[4] for item in results_all_seeds]

        results_str = '\t'.join([str(x) for x in [numpy.mean(accuracy_list), numpy.std(accuracy_list), numpy.mean(consistency_list), numpy.std(consistency_list), 
                                                    numpy.mean(pp_c_list), numpy.std(pp_c_list), numpy.mean(scope_c_list), numpy.std(scope_c_list), 
                                                    numpy.mean(aff_c_list), numpy.std(aff_c_list)]])
        fout.write(header_str)
        fout.write(results_str)