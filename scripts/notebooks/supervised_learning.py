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

# +
from concept_abstraction.training import train_ppo_model, SimpleQEstimator
from concept_abstraction.selection import greedy_selection_supervised, lp_selection_supervised, lp_selection_supervised_imperfect, multiple_selection_supervised
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



# -

is_jupyter = 'ipykernel' in sys.modules

# +
if is_jupyter: 
    seed        = 42
    num_concepts_selected = 112
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


def get_performance(selected_concepts,accuracy_by_concept):
    mlp = MLPClassifier(
        hidden_layer_sizes=(128),
        activation='relu',
        solver='adam',
        max_iter=1000,  # increase if needed
        random_state=0,
        alpha=1e-3,  # instead of 0.0001,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20
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

train_X = np.array([row['attributes'] for row in dataset['train']])
test_X = np.array([row['attributes'] for row in dataset['test']])
train_Y = np.array([row['label'] for row in dataset['train']])
test_Y = np.array([row['label'] for row in dataset['test']])


# +
# Denoising
total_X = np.concatenate([train_X,test_X])
total_Y = np.concatenate([train_Y,test_Y])

rows_by_label = {}
for label in set(total_Y):
    relevant_subset = total_X[total_Y == label]
    relevant_subset = np.round(np.mean(relevant_subset,axis=0))
    rows_by_label[label] = relevant_subset

    train_X[train_Y == label] = relevant_subset
    test_X[test_Y == label] = relevant_subset
# -

results['perfect']['lp'] = {}
for num_concepts_selected in [10,20,30,40,50,60,70,80,90,100,110]:
    lp_concept_list = lp_selection_supervised(train_X,train_Y,num_concepts_selected)
    results['perfect']['lp'][num_concepts_selected] = {
        'reward': get_performance(lp_concept_list,np.ones(312)),
        'concepts': lp_concept_list
    }
    print("LP Performance {}: {}".format(num_concepts_selected,results['perfect']['lp'][num_concepts_selected] ))

manually_selected_concepts = open("../../data/cub/manual_concepts.txt").read().strip().split("\n")
manually_selected_concepts = [int(i) for i in manually_selected_concepts]
results['perfect']['manual'] = {
    'reward': get_performance(manually_selected_concepts,np.ones(312)),
    'concepts': manually_selected_concepts
}
print("Manual Performance {}".format(results['perfect']['manual']['reward']))

# #### Imperfect Concepts

img_locations = ["../../data/cub/images/{}".format(i['location']) for i in dataset['train']]
img_locations_test = ["../../data/cub/images/{}".format(i['location']) for i in dataset['test']]

# +
import torch
from torch.utils.data import Dataset
from PIL import Image

class CUBArrayDataset(Dataset):
    def __init__(self, img_locations, attributes, transform=None):
        """
        img_locations : list of image file paths
        attributes    : numpy array or torch tensor of shape (N, 312)
        transform     : torchvision transforms
        """
        self.img_locations = img_locations
        self.attributes = torch.tensor(attributes, dtype=torch.float32)
        self.transform = transform

    def __len__(self):
        return len(self.img_locations)

    def __getitem__(self, idx):
        img = Image.open(self.img_locations[idx]).convert("RGB")
        label = self.attributes[idx]
        if self.transform:
            img = self.transform(img)
        return img, label



# +
resol = 224          # final crop size (change if needed)
resized_resol = 256  # initial resize before crop (if used)

transform = transforms.Compose([
    transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
    transforms.RandomResizedCrop(resol),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), # implicitly divides by 255
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
])


# +
from torch.utils.data import DataLoader

train_dataset = CUBArrayDataset(img_locations, train_X, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)


# +

model = models.resnet50(pretrained=True)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 312),
    nn.Sigmoid()  # probabilities for each attribute
)

# +

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.BCELoss()  # Binary Cross Entropy for multi-label
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -

threshold = 0.5  # probability cutoff for positive prediction
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_attrs = 0
    total_attrs   = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

        # ---- Average attribute accuracy ----
        preds = (outputs > threshold).float()      # shape: (batch, 312)
        correct_attrs += (preds == labels).sum().item()  # count correct per attribute
        total_attrs   += labels.numel()                  # total predictions = batch * 312

    epoch_loss = running_loss / len(train_loader.dataset)
    attr_accuracy = correct_attrs / total_attrs          # average across 312 attributes

    print(f"Epoch [{epoch+1}/{num_epochs}]  "
          f"Loss: {epoch_loss:.4f}  "
          f"Attr-Accuracy (avg over 312): {attr_accuracy:.4f}")


# +
from sklearn.metrics import f1_score

model.eval()
all_labels = []
all_preds  = []

with torch.no_grad():
    for imgs, labels in train_loader:  # or a separate val/test loader
        imgs  = imgs.to(device)
        labels = labels.cpu().numpy()           # (batch, 312)

        outputs = model(imgs).cpu().numpy()     # probabilities
        preds = (outputs > 0.5).astype(np.int32)  # threshold

        all_labels.append(labels)
        all_preds.append(preds)

# Stack all batches
all_labels = np.vstack(all_labels)  # shape (N, 312)
all_preds  = np.vstack(all_preds)   # shape (N, 312)

# F1 per attribute (macro across 312)
f1_macro = f1_score(all_labels, all_preds, average="macro")
print(f"Macro F1 (avg over 312 attributes): {f1_macro:.4f}")


# +
from torch.utils.data import DataLoader

train_pred_dataset = CUBArrayDataset(img_locations, train_X, transform=transform)
test_pred_dataset  = CUBArrayDataset(img_locations_test, 
                                     torch.zeros((len(img_locations_test), 312)),  # dummy labels
                                     transform=transform)

train_pred_loader = DataLoader(train_pred_dataset, batch_size=32, shuffle=False, num_workers=4)
test_pred_loader  = DataLoader(test_pred_dataset,  batch_size=32, shuffle=False, num_workers=4)


# +

resol = 224  # same resolution as used in training
test_transform = transforms.Compose([
    transforms.CenterCrop(resol),
    transforms.ToTensor(),  # divides by 255
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
])


# +
train_pred_dataset = CUBArrayDataset(
    img_locations,
    torch.zeros((len(img_locations), 312)), 
    transform=test_transform
)
test_pred_dataset = CUBArrayDataset(
    img_locations_test,
    torch.zeros((len(img_locations_test), 312)),
    transform=test_transform
)

train_pred_loader = DataLoader(train_pred_dataset, batch_size=32, shuffle=False, num_workers=4)
test_pred_loader  = DataLoader(test_pred_dataset,  batch_size=32, shuffle=False, num_workers=4)

# -

def get_predictions(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)           # probabilities after sigmoid
            preds.append(outputs.cpu().numpy())
    return np.vstack(preds)                 # shape: (N, 312)



# +
pred_train_X = get_predictions(model, train_pred_loader, device)
pred_test_X  = get_predictions(model, test_pred_loader,  device)

print("pred_train_X:", pred_train_X.shape)
print("pred_test_X :", pred_test_X.shape)


# +
from sklearn.neural_network import MLPClassifier

def get_performance_real(selected_concepts):
    mlp = MLPClassifier(
        hidden_layer_sizes=(128),
        activation='relu',
        solver='adam',
        max_iter=1000,  # increase if needed
        random_state=0,
        alpha=1e-3,  # instead of 0.0001,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20
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

results['imperfect']['multiple_lp'] = {}
for c in results['perfect']['lp']:
    imperfect_concepts = multiple_selection_supervised(train_X,train_Y,c)
    results['imperfect']['multiple_lp'][c] = {
        'reward': get_performance_real(imperfect_concepts), 
        'concepts': imperfect_concepts
    }
results['imperfect']['multiple_lp']

# ## Save Data

save_path = get_save_path(out_folder,save_name)

delete_duplicate_results(out_folder,"",results)

json.dump(results,open('../../results/'+save_path,'w'))


