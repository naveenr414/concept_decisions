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

from concept_abstraction.selection import *
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
    num_concepts_selected = 121
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
results['imperfect'] = {}
results['intervention'] = {}

train = pickle.load(open("../../data/cub/train.pkl","rb"))
test = pickle.load(open("../../data/cub/test.pkl","rb"))

train_X = np.array([i['attribute_label'] for i in train])
train_Y = np.array([i['class_label'] for i in train])
test_X = np.array([i['attribute_label'] for i in test])
test_Y = np.array([i['class_label'] for i in test])


def get_performance(selected_concepts):
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



# #### Imperfect Concepts

def get_performance_real(selected_concepts):
    mlp = MLPClassifier(
        hidden_layer_sizes=(256),
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


# +
def sigmoid(z):
    return 1/(1 + np.exp(-z))

train = pickle.load(open("../../data/cub/train_error.pkl","rb"))
test = pickle.load(open("../../data/cub/test_error.pkl","rb"))

pred_train_X = sigmoid(np.array([i['attribute_label'] for i in train])).round()
train_Y = np.array([i['class_label'] for i in train])
pred_test_X = sigmoid(np.array([i['attribute_label'] for i in test])).round()
test_Y = np.array([i['class_label'] for i in test])
# -

num_concepts_range = range(20,num_concepts_selected,20)

train_concept_accuracy = np.mean(pred_train_X.round() == train_X,axis=0)
test_concept_accuracy = np.mean(pred_test_X.round() == test_X,axis=0)

manually_selected_concepts = open("../../data/cub/manual_concepts.txt").read().strip().split("\n")
manually_selected_concepts = [int(i) for i in manually_selected_concepts]


results['imperfect']['manual'] = {'reward': get_performance_real(manually_selected_concepts), 'concepts': manually_selected_concepts}

# +
results['imperfect']['random'] = {}

for c in num_concepts_range:
    random_concepts = random.sample(list(range(312)),c)
    results['imperfect']['random'][c] = {
        'reward': get_performance_real(random_concepts), 
        'concepts': random_concepts
    }
results['imperfect']['random']

# +
results['imperfect']['entropy'] = {}

for c in num_concepts_range:
    entropy_concepts = basic_greedy_selection_supervised(train_X,train_Y,c)
    results['imperfect']['entropy'][c] = {
        'reward': get_performance_real(entropy_concepts), 
        'concepts': entropy_concepts
    }
results['imperfect']['entropy']

# +
results['imperfect']['greedy'] = {}

for c in num_concepts_range:
    greedy_concepts = greedy_selection_supervised(train_X,train_Y,c)
    results['imperfect']['greedy'][c] = {
        'reward': get_performance_real(greedy_concepts), 
        'concepts': greedy_concepts
    }
results['imperfect']['greedy']

# +
results['imperfect']['lp_hybrid'] = {}

for c in num_concepts_range:
    greedy_concepts = lp_selection_supervised(train_X,train_Y,c)
    results['imperfect']['lp_hybrid'][c] = {
        'reward': get_performance_real(greedy_concepts), 
        'concepts': greedy_concepts
    }
results['imperfect']['lp_hybrid']

# +
results['imperfect']['multiple_log'] = {}

for c in num_concepts_range:
    greedy_concepts = multiple_log_selection_supervised(train_X,train_Y,train_concept_accuracy,c)
    results['imperfect']['multiple_log'][c] = {
        'reward': get_performance_real(greedy_concepts), 
        'concepts': greedy_concepts
    }
results['imperfect']['multiple_log']
# -

# ## Intervention

test = pickle.load(open("../../data/cub/test_error.pkl","rb"))


num_concepts = len(manually_selected_concepts)
manual_selection = manually_selected_concepts
random_selection = random.sample(list(range(312)),112)
entropy_selection = basic_greedy_selection_supervised(train_X,train_Y,112)
greedy_selection = greedy_selection_supervised(train_X,train_Y,112)
lp_selection  = lp_selection_supervised(train_X,train_Y,112)
multiple_selection  = multiple_log_selection_supervised(train_X,train_Y,train_concept_accuracy,112)


def get_performance_real(selected_concepts):
    mlp = MLPClassifier(
        hidden_layer_sizes=(256),
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
                                entropy_selection
                                ],[
                                    "manual","lp_hybrid","multiple_log",
                                    'greedy','random',
                                    'entropy'
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


