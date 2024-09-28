import os
import csv
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from pprint import pprint
import json
from collections import defaultdict
from glob import glob
from copy import deepcopy

def add_synthetics_to_dataset(synthetics, dataset):
    new_dataset = []
    for example in dataset:
        question = example["question"]
        synth_questions = synthetics[question]
        new_dataset.append(example)
        for synth_question in synth_questions:
            synth_example = deepcopy(example)
            synth_example["question"] = synth_question
            new_dataset.append(synth_example)

    return new_dataset

if __name__ == "__main__":
    num_folds = 5
    src_csv_file = "data/all_data.csv"
    dst_file = "data/train_val_test.json"
    
    synth_dir = "data/generated_similar_questions"
        
    augs = defaultdict(list)
    filenames = glob(os.path.join(synth_dir, "*.json"))
    for filename in filenames:
        with open(filename, "r") as f:
            gen_data = json.load(f)
            for question in gen_data["questions"]["user_queries"]:
                augs[gen_data["question"]].append(question["query"])
    
    questions = {}
    with open(src_csv_file, "r") as src_f:
        reader = csv.DictReader(src_f, delimiter=";")
        
        for row in reader:   
            print(row)
            questions[row["question"]] = {
                "question": row["question"],
                "response": row["answer"], 
                "classifier_1": row["classifier_1"],
                "classifier_2": row["classifier_2"],
                "is_knowledge_base": row['is_knowledge_base'] 
            }
            
    examples = list(questions.values())
    stratified_labels = []
    for data in examples:
        stratified_labels.append(data["classifier_2"])
    

    examples = np.array(examples, dtype=object)                
    examples, test_data, examples_stratification_labels, _ = train_test_split(examples, stratified_labels, test_size=0.2, random_state=0, shuffle=True, stratify=stratified_labels)

    dataset = {} 
    dataset["folds"] = []
    
    splitter = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=0)
    for i, (train_index, test_index) in enumerate(splitter.split(examples, examples_stratification_labels)):        
        train_data, val_data = examples[train_index], examples[test_index]            
        dataset["folds"].append({
            "train": train_data.tolist(),
            "val": val_data.tolist()
        })

        
    dataset["test"] = test_data.tolist()
    dataset["train"] = examples.tolist()

    dataset["train"] = add_synthetics_to_dataset(augs, dataset["train"])
    for i in range(num_folds): 
        dataset["folds"][i]["train"] = add_synthetics_to_dataset(augs, dataset["folds"][i]["train"])
    
    with open(dst_file, "w") as dst_f:
        json.dump(dataset, dst_f, ensure_ascii=False, indent=4)