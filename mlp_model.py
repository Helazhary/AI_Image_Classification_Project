""" mlp_model.py description
    ==> multi-layer perceptron model that is defined by 3 layers

    1. Input Layer: Linear(50, 512) + ReLU activation
    2. Hidden Layer: Linear(512, 512) + batch normalization + ReLU activation
    3. Output Layer: Linear(512, 10)


"""
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize._tstutils import f1
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from load_cifar10set import load_cifar10_data
from feature_extraction import extract_features, load_resnet18_model
from pca_reduction import apply_pca
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# this function serves as a definition of the MLP model
class MLP(nn.Module):
    def __init__(self, input_size=50, hidden_size1=512, hidden_size2=512, output_size=10):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.BatchNorm1d(hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, output_size)
        )

    def forward(self, x):
        return self.model(x)


# prepare the data for the MLP model to train/test on
def prepare_data():
    # load the CIFAR-10 set data
    train_loader, test_loader = load_cifar10_data()

    # load pre-trained ResNet-18 model
    model, device = load_resnet18_model()

    # extract features
    train_features = extract_features(model, train_loader, device).numpy()
    test_features = extract_features(model, test_loader, device).numpy()

    # apply PCA ==> reduce feature size 512 --> 50
    train_features_reduced, test_features_reduced = apply_pca(train_features, test_features)

    # create labels from the data loaders
    train_labels = torch.cat([labels for _, labels in train_loader]).numpy()
    test_labels = torch.cat([labels for _, labels in test_loader]).numpy()

    # convert features/labels to TensorDatasets
    train_dataset = TensorDataset(torch.tensor(train_features_reduced, dtype=torch.float32),
                                  torch.tensor(train_labels, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(test_features_reduced, dtype=torch.float32),
                                 torch.tensor(test_labels, dtype=torch.long))

    # create data loaders for training/testing in pytorch
    train_data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_data_loader, test_data_loader, device


# function for model training ==> train MLP model...
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()

    # ...by iterating over training data
    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(features)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()


# funciton to evaluate  MLP model on test data
def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            outputs = model(features)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # return MLP model accuracy, confusion matrix, precision & recall
    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')

    # performance summary table
    performance_summary = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Score': [accuracy, precision, recall, f1]
    })

    # output the performance summary table created
    print("\nPerformance Summary:")
    print(performance_summary.to_string(index=False))

    return accuracy, cm, precision, recall


# function that will plot confusion matrix
def plot_confusion_matrix(model, test_loader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            outputs = model(features)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # generate cm
    cm = confusion_matrix(all_labels, all_preds)

    # plot cm
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    plt.title('Confusion Matrix')
    plt.show()


# main function to train and evaluate the MLP
if __name__ == "__main__":
    # prepare all data
    train_loader, test_loader, device = prepare_data()

    # prepare the model
    model = MLP().to(device)

    # prepare the loss + optimizer functions
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # using the SGD optimizer with momentum=0.9 as specified in the project outline

    # train model for x amount of epochs
    num_epochs = 20
    for epoch in range(num_epochs):
        train_model(model, train_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch + 1}/{num_epochs}] completed.")

    # output the evaluation of our model
    accuracy, cm, precision, recall = evaluate_model(model, test_loader, device)
    print(f"Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(cm)
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
