# AI_Image_Classification_Project

## Files Overview
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