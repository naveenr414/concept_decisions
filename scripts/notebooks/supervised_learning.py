# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: food
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

from concept_abstraction.training import train_ppo_model, SimpleQEstimator
from concept_abstraction.selection import greedy_selection_supervised
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


is_jupyter = 'ipykernel' in sys.modules

# +
if is_jupyter: 
    seed        = 42
    num_concepts_selected = 2
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

dataset = json.load(open("../../data/cub/preprocessed.json"))

# ## Concept Selection

train_X = np.array([row['attributes'] for row in dataset['train']])
test_X = np.array([row['attributes'] for row in dataset['test']])

all_rows_train = set([''.join([str(int(j)) for j in row]) for row in train_X])
all_rows_test = set([''.join([str(int(j)) for j in row]) for row in test_X])


def get_performance(selected_concepts,accuracy_by_concept):
    train_X = np.array([row['attributes'] for row in dataset['train']])
    test_X = np.array([row['attributes'] for row in dataset['test']])

    flip_probs = 1 - np.array(accuracy_by_concept)
    rand_vals = np.random.rand(*train_X.shape)
    flip_mask = rand_vals < flip_probs  # True means flip
    train_X = np.where(flip_mask, 1 - train_X, train_X)
    train_X = train_X[:,selected_concepts]

    flip_probs = 1 - np.array(accuracy_by_concept)
    rand_vals = np.random.rand(*test_X.shape)
    flip_mask = rand_vals < flip_probs  # True means flip
    test_X = np.where(flip_mask, 1 - test_X, test_X)
    test_X = test_X[:,selected_concepts]


    train_Y = np.array([row['label'] for row in dataset['train']])
    test_Y = np.array([row['label'] for row in dataset['test']])

    mlp = MLPClassifier(
        hidden_layer_sizes=(50),
        activation='relu',
        solver='adam',
        max_iter=1000,  # increase if needed
        random_state=0,
        alpha=1e-3,  # instead of 0.0001,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20
    )

    # Train the model
    mlp.fit(train_X, train_Y)

    # Predict on the test set
    y_pred = mlp.predict(test_X)

    # Compute accuracy
    acc = accuracy_score(test_Y, y_pred)
    return acc

train_X = np.array([row['attributes'] for row in dataset['train']])
test_X = np.array([row['attributes'] for row in dataset['test']])
train_Y = np.array([row['label'] for row in dataset['train']])
test_Y = np.array([row['label'] for row in dataset['test']])


# +
random_concept_list = [np.random.choice(list(range(train_X.shape[1])),k,replace=False).tolist() for k in range(1,num_concepts_selected)]
random_average_reward = [get_performance(c,np.ones(312)) for c in random_concept_list]

results['random_selection'] = {
    'concepts': random_concept_list, 
    'values': random_average_reward,
}
# -

greedy_concept_list = greedy_selection_supervised(train_X,train_Y,num_concepts_selected)
greedy_reward = [get_performance(c,np.ones(312)) for c in greedy_concept_list]
results['greedy_selection'] = {
    'concepts': greedy_concept_list, 
    'values': greedy_reward,
}

manually_selected_concepts = [
    1,
    4,
    6,
    7,
    10,
    14,
    15,
    20,
    21,
    23,
    25,
    29,
    30,
    35,
    36,
    38,
    40,
    44,
    45,
    50,
    51,
    53,
    54,
    56,
    57,
    59,
    63,
    64,
    69,
    70,
    72,
    75,
    80,
    84,
    90,
    91,
    93,
    99,
    101,
    106,
    110,
    111,
    116,
    117,
    119,
    125,
    126,
    131,
    132,
    134,
    145,
    149,
    151,
    152,
    153,
    157,
    158,
    163,
    164,
    168,
    172,
    178,
    179,
    181,
    183,
    187,
    188,
    193,
    194,
    196,
    198,
    202,
    203,
    208,
    209,
    211,
    212,
    213,
    218,
    220,
    221,
    225,
    235,
    236,
    238,
    239,
    240,
    242,
    243,
    244,
    249,
    253,
    254,
    259,
    260,
    262,
    268,
    274,
    277,
    283,
    289,
    292,
    293,
    294,
    298,
    299,
    304,
    305,
    308,
    309,
    310,
    311,
]
results['manually_selected_concepts'] = manually_selected_concepts

# +
random_from_manual_list = [np.random.choice(manually_selected_concepts,k,replace=False).tolist() for k in range(1,num_concepts_selected)]
random_from_manual_reward = [get_performance(c,np.ones(312)) for c in random_concept_list]

results['random_manual_selection'] = {
    'concepts': random_from_manual_list, 
    'values': random_from_manual_reward,
}
# -

results['attribute_names'] = open("../../data/cub/attributes.txt").read().split("\n")

results['random_selection']['values'], results['greedy_selection']['values'], results['random_manual_selection']['values']

# +
# TODO: Actually Train Two-Stage Data
# -

# ## Save Data

save_path = get_save_path(out_folder,save_name)

delete_duplicate_results(out_folder,"",results)

json.dump(results,open('../../results/'+save_path,'w'))


