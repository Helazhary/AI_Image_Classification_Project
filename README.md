# AI_Image_Classification_Project

## Files Overview

__________________________________________________________________________________________________________________________________________________________________________________________________
- **mlp_model.py**: Contains the complete implementation of the base MLP model, the shallow MLP variant, and the deep MLP variant, the training of these models, aswell as their evaluations. 

#Steps to Run the Code

### 1. Data Preprocessing
The *prepare_data()* function in **mlp_model.py** loads the CIFAR-10 dataset and extracts the features using the pre-trained ResNet-18 model.
The features sizes' are then reduced using PCA reduction, which are then converted to torches and labeled into PyTorch tensor objects.

### 2. Training the Models
Set the parameters to your liking & modify *num_epochs* in **mlp_model.py** to set the number of training epochs. It is set to 20 by default.
Running the script will train each MLP variant and evaluate their performance sequentially: 
MLP model: Original architecture.
Shallow MLP model: 2-layered MLP model.
Deep MLP model: 4-layered MLP model.

### 3. Evaluation
The script prints a summary table of the metrics of each model: accuracy, precision, recall, f1-score. It also prints a comparison chart of accuracy percentages.

### 4. Execution
Outputs include epoch training logs, confusion matrices, a performance summary table, and an accuracy comparison chart.

__________________________________________________________________________________________________________________________________________________________________________________________________
- **CNN_Ai_Image_Classificatoin_Project.ipynb:** Contains the complete implementation of the VGG models, training, evaluation, and data preprocessing.

#Steps to Run the Code

### 1. Data Preprocessing
Install dependencies: torch, torchvision, numpy, matplotlib, pandas, seaborn, and scikit-learn.
Data is automatically downloaded and limited to a specified number of samples per class using get_limited_data_loaders.

### 2. Training the Models
Ensure GPU is available for faster training (uses CUDA if available).

Three models are trained sequentially:
VGG11: Original architecture.
VGG8: Reduced-depth architecture.
VGG11_LargeKernel: Modified with larger convolutional kernels.

### 3. Evaluation
Metrics include accuracy, precision, recall, F1-score, and confusion matrix visualization.
Results are summarized in a performance table.

### 4. Execution
Run all cells for the code to work properly.
Outputs include training logs, confusion matrices, and a performance summary table.
