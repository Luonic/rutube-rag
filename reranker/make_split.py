import os
import csv
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from pprint import pprint
import json


if __name__ == "__main__":
    num_folds = 5
    src_csv_file = "data/all_data.csv"
    dst_file = "data/train_val_test.json"
    
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
    
    with open(dst_file, "w") as dst_f:
        json.dump(dataset, dst_f, ensure_ascii=False)