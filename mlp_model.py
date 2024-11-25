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


# this function serves as a definition of the MLP model, it holds 3 layers
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


# 2 new classes will serve as experimentation for adding/removing layers to the model & annotating the differences
# remove 1 layer --> 2 layers
class ShallowMLP(nn.Module):
    def __init__(self, input_size=50, hidden_size=256, output_size=10):
        super(ShallowMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        return self.model(x)


# add 1 layer --> 4 layers
class DeepMLP(nn.Module):
    def __init__(self, input_size=50, hidden_size=512, output_size=10):
        super(DeepMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
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

    # metrics for future analysis
    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')  # Compute the F1 score

    # make a performance summary table
    performance_summary = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Score': [accuracy, precision, recall, f1]
    })

    # Output the performance summary table
    print("\nPerformance Summary:")
    print(performance_summary.to_string(index=False))

    return accuracy, cm, precision, recall


# training and evaluation function
def train_and_evaluate(mlp_class, model_name):
    print(f"\nCurrently training and evaluating: {model_name}")
    model = mlp_class().to(device)  # Initialize the model
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # Optimizer
    criterion = nn.CrossEntropyLoss()  # Loss function

    # train our model
    for epoch in range(num_epochs):
        train_model(model, train_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch + 1}/{num_epochs}] completed for {model_name}.")

    # print the model metrics evaluations
    accuracy, cm, precision, recall = evaluate_model(model, test_loader, device)
    print(f"Final Results for {model_name}:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

    return accuracy


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
    # prepare dataset
    train_loader, test_loader, device = prepare_data()

    # number of epochs
    num_epochs = 20

    # all 3 models (base + variants)
    models = [
        (MLP, "Base MLP (3 Layers)"),
        (ShallowMLP, "Shallow MLP Variant (2 Layers)"),
        (DeepMLP, "Deep MLP Variant (4 Layers)")
    ]

    # Store results
    results = []

    # train each model + show results
    for model_class, name in models:
        acc = train_and_evaluate(model_class, name)
        results.append((name, acc))

    # collect all results
    import pandas as pd
    results_df = pd.DataFrame(results, columns=["Model", "Accuracy"])
    print("\nExperiment Results:")
    print(results_df)

    # plot results
    import matplotlib.pyplot as plt
    names, accuracies = zip(*results)
    plt.bar(names, accuracies)
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.title("Comparison of Model Depths")
    plt.xticks(rotation=45)
    plt.show()


OUTPUT
__________________________________________________________________

Currently training and evaluating: Base MLP (3 Layers)
Epoch [1/20] completed for Base MLP (3 Layers).
Epoch [2/20] completed for Base MLP (3 Layers).
Epoch [3/20] completed for Base MLP (3 Layers).
Epoch [4/20] completed for Base MLP (3 Layers).
Epoch [5/20] completed for Base MLP (3 Layers).
Epoch [6/20] completed for Base MLP (3 Layers).
Epoch [7/20] completed for Base MLP (3 Layers).
Epoch [8/20] completed for Base MLP (3 Layers).
Epoch [9/20] completed for Base MLP (3 Layers).
Epoch [10/20] completed for Base MLP (3 Layers).
Epoch [11/20] completed for Base MLP (3 Layers).
Epoch [12/20] completed for Base MLP (3 Layers).
Epoch [13/20] completed for Base MLP (3 Layers).
Epoch [14/20] completed for Base MLP (3 Layers).
Epoch [15/20] completed for Base MLP (3 Layers).
Epoch [16/20] completed for Base MLP (3 Layers).
Epoch [17/20] completed for Base MLP (3 Layers).
Epoch [18/20] completed for Base MLP (3 Layers).
Epoch [19/20] completed for Base MLP (3 Layers).
Epoch [20/20] completed for Base MLP (3 Layers).

Performance Summary:
   Metric    Score
 Accuracy 0.758000
Precision 0.761611
   Recall 0.758000
 F1-Score 0.758444
                       
Final Results for Base MLP (3 Layers):
Accuracy: 0.76
Precision: 0.76
Recall: 0.76
                       
Confusion Matrix:
[[74  1  4  3  1  0  0  1 11  5]
 [ 2 79  1  2  0  0  0  0  4 12]
 [ 6  1 60  7  8  8  9  1  0  0]
 [ 0  3  5 64  4 15  6  0  1  2]
 [ 0  1  3  9 76  3  1  7  0  0]
 [ 0  1  5 18  3 71  2  0  0  0]
 [ 0  0  4  6  0  4 85  0  0  1]
 [ 4  0  0  5  8  2  0 79  1  1]
 [ 8  3  2  1  0  0  1  0 84  1]
 [ 1 12  0  0  0  0  0  0  1 86]]


Currently training and evaluating: Shallow MLP (2 Layers)
Epoch [1/20] completed for Shallow MLP (2 Layers).
Epoch [2/20] completed for Shallow MLP (2 Layers).
Epoch [3/20] completed for Shallow MLP (2 Layers).
Epoch [4/20] completed for Shallow MLP (2 Layers).
Epoch [5/20] completed for Shallow MLP (2 Layers).
Epoch [6/20] completed for Shallow MLP (2 Layers).
Epoch [7/20] completed for Shallow MLP (2 Layers).
Epoch [8/20] completed for Shallow MLP (2 Layers).
Epoch [9/20] completed for Shallow MLP (2 Layers).
Epoch [10/20] completed for Shallow MLP (2 Layers).
Epoch [11/20] completed for Shallow MLP (2 Layers).
Epoch [12/20] completed for Shallow MLP (2 Layers).
Epoch [13/20] completed for Shallow MLP (2 Layers).
Epoch [14/20] completed for Shallow MLP (2 Layers).
Epoch [15/20] completed for Shallow MLP (2 Layers).
Epoch [16/20] completed for Shallow MLP (2 Layers).
Epoch [17/20] completed for Shallow MLP (2 Layers).
Epoch [18/20] completed for Shallow MLP (2 Layers).
Epoch [19/20] completed for Shallow MLP (2 Layers).
Epoch [20/20] completed for Shallow MLP (2 Layers).

Performance Summary:
   Metric    Score
 Accuracy 0.782000
Precision 0.785481
   Recall 0.782000
 F1-Score 0.782501
                      
Final Results for Shallow MLP (2 Layers):
Accuracy: 0.78
Precision: 0.79
Recall: 0.78

Confusion Matrix:
[[80  1  1  1  2  0  0  1  9  5]
 [ 3 80  1  1  0  0  0  0  3 12]
 [ 5  0 70  5  3  5 12  0  0  0]
 [ 1  1  5 69  4 12  7  0  1  0]
 [ 1  0  7  6 74  4  2  5  1  0]
 [ 0  0  6 18  1 71  2  1  1  0]
 [ 1  1  3  8  0  3 83  0  1  0]
 [ 2  0  1  3 10  3  0 79  1  1]
 [ 9  3  2  0  0  0  0  0 85  1]
 [ 0  6  0  0  0  0  0  0  3 91]]


Currently training and evaluating: Deep MLP (4 Layers)
Epoch [1/20] completed for Deep MLP (4 Layers).
Epoch [2/20] completed for Deep MLP (4 Layers).
Epoch [3/20] completed for Deep MLP (4 Layers).
Epoch [4/20] completed for Deep MLP (4 Layers).
Epoch [5/20] completed for Deep MLP (4 Layers).
Epoch [6/20] completed for Deep MLP (4 Layers).
Epoch [7/20] completed for Deep MLP (4 Layers).
Epoch [8/20] completed for Deep MLP (4 Layers).
Epoch [9/20] completed for Deep MLP (4 Layers).
Epoch [10/20] completed for Deep MLP (4 Layers).
Epoch [11/20] completed for Deep MLP (4 Layers).
Epoch [12/20] completed for Deep MLP (4 Layers).
Epoch [13/20] completed for Deep MLP (4 Layers).
Epoch [14/20] completed for Deep MLP (4 Layers).
Epoch [15/20] completed for Deep MLP (4 Layers).
Epoch [16/20] completed for Deep MLP (4 Layers).
Epoch [17/20] completed for Deep MLP (4 Layers).
Epoch [18/20] completed for Deep MLP (4 Layers).
Epoch [19/20] completed for Deep MLP (4 Layers).
Epoch [20/20] completed for Deep MLP (4 Layers).

Performance Summary:
   Metric    Score
 Accuracy 0.777000
Precision 0.775489
   Recall 0.777000
 F1-Score 0.775275
                      
Final Results for Deep MLP (4 Layers):
Accuracy: 0.78
Precision: 0.78
Recall: 0.78
                      
Confusion Matrix:
[[77  1  3  1  1  1  0  1 10  5]
 [ 2 84  1  0  0  0  0  0  4  9]
 [ 6  1 65  5  6  7  6  1  3  0]
 [ 1  1  3 61  6 17  7  4  0  0]
 [ 1  0  7  9 69  2  2  8  1  1]
 [ 0  0  5 11  3 73  2  4  2  0]
 [ 1  0  3  6  1  4 84  0  1  0]
 [ 2  0  1  3  2  0  0 88  0  4]
 [ 7  2  2  0  0  0  1  0 88  0]
 [ 0 10  0  0  0  0  0  0  2 88]]


Experiment Results:
                    Model  Accuracy
0     Base MLP (3 Layers)     0.758
1  Shallow MLP (2 Layers)     0.782
2     Deep MLP (4 Layers)     0.777

Process finished with exit code 0
