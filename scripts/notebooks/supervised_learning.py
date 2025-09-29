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

from concept_abstraction.training import train_ppo_model, SimpleQEstimator
from concept_abstraction.selection import greedy_selection_supervised, lp_selection_supervised, lp_selection_supervised_imperfect, multiple_selection_supervised, iterative_selection_supervised, greedy_selection_supervised
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

train_X = np.load("../../data/cub/cub_train_x.npy")
test_X = np.load("../../data/cub/cub_test_x.npy")
train_Y = np.array([row['label'] for row in dataset['train']])
test_Y = np.array([row['label'] for row in dataset['test']])
val_X = test_X[:1000,:]
val_Y = test_Y[:1000]
test_X = test_X[1000:,:]
test_Y = test_Y[1000:]

results['perfect']['lp'] = {}
for num_concepts_selected in range(10,311,10):
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
img_locations_val = img_locations_test[:1000]
img_locations_test = img_locations_test[1000:]

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
resol = 299 # Inception V3 requires 299x299
resized_resol = 299 # not needed for RandomResizedCrop

# Training transforms
train_transform = transforms.Compose([
    transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),  # Match paper's exact params
    transforms.RandomResizedCrop(resol),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet normalization
])

# Validation/test transforms (center crop + resize)
val_transform = transforms.Compose([
    transforms.Resize(resol),
    transforms.CenterCrop(resol),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# +
from torch.utils.data import DataLoader

train_dataset = CUBArrayDataset(img_locations, train_X, transform=train_transform)
val_dataset = CUBArrayDataset(img_locations_val, val_X, transform=val_transform)  # Add validation set

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)  # batch_size=64
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)


# +
model = models.inception_v3(pretrained=True, aux_logits=True)  # Use Inception V3

# Replace final FC layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 312)  # No sigmoid here - apply in loss

# Also replace auxiliary classifier if using aux_logits
if model.aux_logits:
    num_aux_features = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = nn.Linear(num_aux_features, 312)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# +
pos_weight = []

for i in range(train_X.shape[1]):
    pos_count = train_X[:, i].sum().item()
    neg_count = len(train_X) - pos_count
    
    if pos_count == 0:
        weight = 1.0
    else:
        weight = neg_count / pos_count
        # Cap at a reasonable maximum (paper says ~9 average, so cap at 3-4x that)
        weight = min(weight, 30.0)  # or 50.0 max
    
    pos_weight.append(weight)

pos_weight = torch.tensor(pos_weight).to(device)

print(f"Average pos_weight: {pos_weight.mean().item():.2f}")
print(f"Min: {pos_weight.min().item():.2f}, Max: {pos_weight.max().item():.2f}")

# +

# Use BCEWithLogitsLoss with pos_weight for class imbalance
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# SGD with momentum 0.9 (start with one LR, will do hyperparameter search)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0004)


# +
threshold = 0.5
num_epochs = 50
best_val_acc = 0.0

for epoch in range(num_epochs):
    # ===== Training Phase =====
    model.train()
    running_loss = 0.0
    correct_attrs = 0
    total_attrs = 0
    
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # Inception V3 returns (outputs, aux_outputs) during training
        if model.training and model.aux_logits:
            outputs, aux_outputs = model(imgs)
            loss1 = criterion(outputs, labels)
            loss2 = criterion(aux_outputs, labels)
            loss = loss1 + 0.4 * loss2  # Standard Inception V3 aux loss weighting
        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * imgs.size(0)
        
        # Accuracy calculation (apply sigmoid to logits)
        preds = (torch.sigmoid(outputs) > threshold).float()
        correct_attrs += (preds == labels).sum().item()
        total_attrs += labels.numel()
    
    epoch_loss = running_loss / len(train_loader.dataset)
    train_attr_accuracy = correct_attrs / total_attrs
    
    # ===== Validation Phase =====
    model.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = (torch.sigmoid(outputs) > threshold).float()
            val_correct += (preds == labels).sum().item()
            val_total += labels.numel()
    
    val_attr_accuracy = val_correct / val_total
    
    # Track best model
    if val_attr_accuracy > best_val_acc:
        best_val_acc = val_attr_accuracy
        torch.save(model.state_dict(), 'best_model.pth')
    
    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Train Loss: {epoch_loss:.4f} "
          f"Train Acc: {train_attr_accuracy:.4f} "
          f"Val Acc: {val_attr_accuracy:.4f}")


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
resol = 299  # Inception V3 uses 299x299, not 224

test_transform = transforms.Compose([
    transforms.Resize(resol),  # First resize the image
    transforms.CenterCrop(resol),  # Then center crop to 299x299
    transforms.ToTensor(),  # divides by 255
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

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
train_pred_loader = DataLoader(train_pred_dataset, batch_size=64, shuffle=False, num_workers=4)  # batch_size=64
test_pred_loader = DataLoader(test_pred_dataset, batch_size=64, shuffle=False, num_workers=4)  # batch_size=64

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
        hidden_layer_sizes=(128,128),
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


# +
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
mlp.fit(pred_train_X, train_Y)

# Predict on the test set
y_pred = mlp.predict(pred_test_X)

# Compute accuracy
acc = accuracy_score(test_Y.reshape(-1,1), y_pred)
acc
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
results['imperfect']['iterative'] = {}
all_imperfect_concepts = iterative_selection_supervised(pred_train_X,train_Y,310)

for c in results['perfect']['lp']:
    imperfect_concepts = all_imperfect_concepts[:c]
    results['imperfect']['iterative'][c] = {
        'reward': get_performance_real(imperfect_concepts), 
        'concepts': imperfect_concepts
    }
results['imperfect']['iterative']

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

# ## Save Data

save_path = get_save_path(out_folder,save_name)

delete_duplicate_results(out_folder,"",results)

json.dump(results,open('../../results/'+save_path,'w'))


