import streamlit as st
import nltk
import datetime
st.set_page_config(layout="wide")
import json
import glob
import numpy as np
from collections import Counter
import numpy as np
import math


# ===========================================#
#        Loads Model and word_to_id         #
# ===========================================#

bgcolor = "#f0f7fa"
edit_mapping = {0:"original", 1:"paraphrase", 2:"scope", 3:"affirmative"}

def get_groups_consistent(gold_data):
    groups = {}

    all_questions = [x["sentence2"] for x in gold_data]
    consistency_subset = [ind for ind, x in enumerate(gold_data) if all_questions.count(x["sentence2"]) == 4]


    for ind in consistency_subset:
        x = gold_data[ind]
        passage_id = x["PassageID"]
        if passage_id not in groups:
            groups[passage_id] = {}

        passage_edit = x["PassageEditID"]
        question = x["sentence2"]
        if passage_edit not in groups[passage_id]:
            groups[passage_id][passage_edit]=[]

        groups[passage_id][passage_edit].append(x)


    return groups, consistency_subset


def get_groups_all(gold_data):
    groups = {}

    for x in gold_data:
        passage_id = x["PassageID"]
        if passage_id not in groups:
            groups[passage_id] = {}

        passage_edit = x["PassageEditID"]
        if passage_edit not in groups[passage_id]:
            groups[passage_id][passage_edit] = []

        groups[passage_id][passage_edit].append(x)

    return groups

def read_data(filename):
    f=open(filename,"r")
    data = []
    with open(filename) as f:
        for line in f:
            data.append(json.loads(line))
    f.close()
    return data

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def get_highlighted_passage(edit_state, annotation):
    # sample[edit_state] = {"Passage": passage, "original sentence": original_sentence,
    #                       "original cue": original_cue,
    #                       "pre_sentence": pre_sentence, "post_sentence": post_sentence,
    #                       "pre_cue": pre_cue,
    #                       "post_cue": post_cue,
    #                       "QA Pairs": []}

    sample = annotation[edit_state]
    original_sentence = sample[0]["original sentence"]
    original_cue = sample[0]["original cue"]
    sentences = nltk.sent_tokenize(sample[0]["sentence1"])
    html_string = "<b>Cue: "+original_cue+"</b></br>"

    most_sim_sentence = 0
    sentence_sims = []
    for sentence_ind, sentence in enumerate(sentences):
        sentence_sims.append(levenshteinDistance(sentence, original_sentence))

    min_value = min(sentence_sims)
    min_index = sentence_sims.index(min_value)

    for sentence_ind, sentence in enumerate(sentences):
        if sentence_ind == min_index:
            if original_cue in sentence:
                parts = sentence.split(original_cue)
                html_string += " <span style='background-color:yellow'>" + parts[0] + "<b>" + original_cue + "</b>" + \
                               parts[1] + "</span>"
            else:
                html_string += " <span style='background-color:yellow'>" + sentence + "</span>"
        else:
            html_string += " <span style='background-color:" + bgcolor + ";'>" + sentence + "</span>"

    # pdb.set_trace()

    return html_string + "\n"


# ===========================================#
#              Streamlit Code               #
# ===========================================#
desc = "Analyze Data for Reading Comprehension over negated statements"

st.title('CONDA QA')
st.write(desc)
# col1, col2 = st.columns([4, 1])


all_annotations = read_data("./condaqa_test.json")
groups, consistent_subset=get_groups_consistent(all_annotations)

passage_ids=list(groups.keys())
import random
random.shuffle(passage_ids)
passage_ids=passage_ids[:10]

N_HITS = 0
QA_pairs = 0
N_Passages=0



worker_feedback={}
st.subheader('Annotations')
bgcolor = "#f0f7fa"


recent_crowdworkers=[]


# Annotations correspond to batches
# Only show the last 1000 annotations to avoid slowing down the interface
#all_annotations[:10]
# for batch_num, batch_info in enumerate(all_annotations):
#     batch = batch_info[0]
#     for worker_id in batch["Workers"]:
#         recent_crowdworkers.append(worker_id)
#
#         HITID=batch["Workers"][worker_id][0]["HITID"]
#         feedback="\n".join([x["Feedback"] for x in batch["Workers"][worker_id]])
#
#
#         for each_hit in batch["Workers"][worker_id]:

for passage_id in passage_ids:


        annotation = groups[passage_id]
        markdown = ""
        markdown = "<div style='background-color:" + bgcolor + ";'>"
        for edit_state in range(4):
            markdown += "<h4><div style='background-color:" + bgcolor + ";'> " + edit_mapping[edit_state].upper() + "</div> </h4>"
            markdown += "\n\n"
            markdown += get_highlighted_passage(edit_state, annotation)
            # # markdown += " \n " + each_hit[edit_state]["Passage"].replace("'''", '"')


            for each_q in annotation[edit_state]:
                print(each_q)
                markdown += "\n <b>Question </b>: " + each_q["sentence2"]  + " <b>Answer</b>: " + each_q["label"] + "\n\n"

            markdown += "\n"


        markdown += "</div>"
        st.markdown(markdown, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

