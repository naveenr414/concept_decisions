# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: food
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

# +
import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["GRB_LICENSE_FILE"] = "/usr0/home/naveenr/gurobi.lic"
# -

from concept_abstraction.selection import greedy_selection_supervised, lp_selection_supervised, lp_selection_supervised_imperfect, multiple_selection_supervised, greedy_selection_supervised,imperfect_lp_selection_supervised
from concept_abstraction.env_utils import *
from concept_abstraction.utils import *
import sys 
import argparse
import secrets
import numpy as np 
import random 
import time 
from collections import Counter
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import torch.nn as nn
from torchvision import models
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import pickle 

is_jupyter = 'ipykernel' in sys.modules

# +
if is_jupyter: 
    seed        = 42
    num_concepts_selected = 21
    out_folder = "cub"
else:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', help='Random Seed', type=int, default=42)
    parser.add_argument('--num_concepts_selected', help='Number of concepts selected by greedy or random',type=int, default=0)
    parser.add_argument('--out_folder', help='Which folder', type=str, default="exploration")

    args = parser.parse_args()

    seed = args.seed
    num_concepts_selected = args.num_concepts_selected
    out_folder = args.out_folder

save_name = secrets.token_hex(4)  
# -

results = {}
results['parameters'] = {'seed'      : seed,
        'num_concepts_selected': num_concepts_selected,
}
print("Parameters {}".format(results['parameters']))

np.random.seed(seed)
random.seed(seed)


def get_performance(selected_concepts,accuracy_by_concept):
    mlp = MLPClassifier(
        hidden_layer_sizes=(128),
        activation='relu',
        solver='adam',
    )

    # Train the model
    mlp.fit(train_X[:,selected_concepts], train_Y)

    # Predict on the test set
    y_pred = mlp.predict(test_X[:,selected_concepts])

    # Compute accuracy
    acc = accuracy_score(test_Y.reshape(-1,1), y_pred)
    return acc


# ## Perfrect Concepts

results['perfect'] = {}

train = pickle.load(open("../../data/cub/train.pkl","rb"))
test = pickle.load(open("../../data/cub/test.pkl","rb"))

train_X = np.array([i['attribute_label'] for i in train])
train_Y = np.array([i['class_label'] for i in train])
test_X = np.array([i['attribute_label'] for i in test])
test_Y = np.array([i['class_label'] for i in test])

results['perfect']['lp'] = {}
for c in range(20,num_concepts_selected,20):
    lp_concept_list = lp_selection_supervised(train_X,train_Y,c)
    results['perfect']['lp'][c] = {
        'reward': get_performance(lp_concept_list,np.ones(312)),
        'concepts': lp_concept_list
    }
print("Finished LP")

manually_selected_concepts = open("../../data/cub/manual_concepts.txt").read().strip().split("\n")
manually_selected_concepts = [int(i) for i in manually_selected_concepts]
results['perfect']['manual'] = {
    'reward': get_performance(manually_selected_concepts,np.ones(312)),
    'concepts': manually_selected_concepts
}
print("Manual Performance {}".format(results['perfect']['manual']['reward']))


# #### Imperfect Concepts

# +
def sigmoid(z):
    return 1/(1 + np.exp(-z))

train = pickle.load(open("../../data/cub/train_error.pkl","rb"))
test = pickle.load(open("../../data/cub/test_error.pkl","rb"))

pred_train_X = sigmoid(np.array([i['attribute_label'] for i in train])).round()
train_Y = np.array([i['class_label'] for i in train])
pred_test_X = sigmoid(np.array([i['attribute_label'] for i in test])).round()
test_Y = np.array([i['class_label'] for i in test])

# +
from sklearn.neural_network import MLPClassifier

def get_performance_real(selected_concepts):
    mlp = MLPClassifier(
        hidden_layer_sizes=(128),
        activation='relu',
        solver='adam',
    )

    # Train the model
    mlp.fit(pred_train_X[:,selected_concepts], train_Y)

    # Predict on the test set
    y_pred = mlp.predict(pred_test_X[:,selected_concepts])

    # Compute accuracy
    acc = accuracy_score(test_Y.reshape(-1,1), y_pred)
    return acc


# -

results['imperfect'] = {}

manually_selected_concepts = open("../../data/cub/manual_concepts.txt").read().strip().split("\n")
manually_selected_concepts = [int(i) for i in manually_selected_concepts]


results['imperfect']['manual'] = {'reward': get_performance_real(manually_selected_concepts), 'concepts': manually_selected_concepts}

results['imperfect']['lp'] = {}
for c in results['perfect']['lp']:
    results['imperfect']['lp'][c] = {
        'reward': get_performance_real(results['perfect']['lp'][c]['concepts']),
        'concepts': c
    }

results['imperfect']['multiple'] = {}
for c in results['perfect']['lp']:
    imperfect_concepts = multiple_selection_supervised(train_X,train_Y,c)
    results['imperfect']['multiple'][c] = {
        'reward': get_performance_real(imperfect_concepts), 
        'concepts': imperfect_concepts
    }
results['imperfect']['multiple']

# +
results['imperfect']['random'] = {}

for c in results['perfect']['lp']:
    random_concepts = random.sample(list(range(312)),c)
    results['imperfect']['random'][c] = {
        'reward': get_performance_real(random_concepts), 
        'concepts': random_concepts
    }
results['imperfect']['random']

# +
results['imperfect']['greedy'] = {}

for c in results['perfect']['lp']:
    greedy_concepts = greedy_selection_supervised(train_X,train_Y,c)
    results['imperfect']['greedy'][c] = {
        'reward': get_performance_real(greedy_concepts), 
        'concepts': greedy_concepts
    }
results['imperfect']['greedy']
# -

# ## Intervention

test = pickle.load(open("../../data/cub/test_error.pkl","rb"))


num_concepts = len(manually_selected_concepts)
lp_selection  = lp_selection_supervised(train_X,train_Y,num_concepts)
multiple_selection  = multiple_selection_supervised(train_X,train_Y,num_concepts)
greedy_selection = greedy_selection_supervised(train_X,train_Y,num_concepts)
manual_selection = manually_selected_concepts
random_selection = random.sample(list(range(312)),num_concepts)
accuracies = np.mean(test_X == pred_test_X,axis=0)
imperfect_selection = imperfect_lp_selection_supervised(train_X,train_Y,num_concepts,accuracies)


def get_performance_real(selected_concepts):
    mlp = MLPClassifier(
        hidden_layer_sizes=(128),
        activation='relu',
        solver='adam',
    )

    # Train the model
    mlp.fit(pred_train_X[:,selected_concepts], train_Y)

    # Predict on the test set
    y_pred = mlp.predict(intervention_test_X[:,selected_concepts])

    # Compute accuracy
    acc = accuracy_score(test_Y.reshape(-1,1), y_pred)
    return acc


results['intervention'] = {}

for intervention_percent in [0.2,0.4,0.6,0.8,1.0]:
    num_cols = test_X.shape[1]
    num_cols_to_intervene = int((1-intervention_percent) * num_cols)

    # Randomly pick columns
    cols = np.random.choice(
        num_cols, num_cols_to_intervene, replace=False
    )

    # Start from original
    intervention_test_X = test_X.copy()

    # Replace selected columns entirely
    intervention_test_X[:, cols] = pred_test_X[:, cols]

    for arr,description in zip([manually_selected_concepts,
                                lp_selection,
                                multiple_selection,
                                greedy_selection,
                                random_selection,
                                imperfect_selection],[
                                    "manual","lp","multiple",
                                    'greedy','random',
                                    'imperfect'
                                ]):
        if description not in results['intervention']:
            results['intervention'][description] = {}
        results['intervention'][description][intervention_percent] = {
            'reward': get_performance_real(arr)
        }
        print(description,intervention_percent,results['intervention'][description][intervention_percent]['reward'])


# ## Save Data

save_path = get_save_path(out_folder,save_name)

delete_duplicate_results(out_folder,"",results)

json.dump(results,open('../../results/'+save_path,'w'))


