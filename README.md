# Binary Classification on UNSW-NB15 Dataset

This project implements binary and multi-class classification models on the UNSW-NB15 network intrusion detection dataset. The project explores multiple machine learning algorithms to identify and classify network attacks.

## Overview

The UNSW-NB15 dataset is a comprehensive network security dataset containing nine types of attacks (DoS, Exploits, Backdoors, Analysis, Fuzzers, Shellcode, Reconnaissance, Worms, and Normal traffic). This project focuses on:
- **Binary Classification**: Normal vs. Attack traffic
- **Multi-class Classification**: Distinguishing between different attack types and normal traffic

## Project Structure

```
.
├── datasets/
│   ├── UNSW_NB15_whole.csv          # Complete dataset
│   ├── UNSW_NB15_features.csv       # Dataset features
│   ├── bin_data.csv                 # Binary classification dataset
│   └── multi_data.csv               # Multi-class classification dataset
├── models/
│   ├── decision_tree_binary.pkl     # Binary DT model
│   ├── decision_tree_multi.pkl      # Multi-class DT model
│   ├── knn_binary.pkl               # Binary KNN model
│   ├── knn_multi.pkl                # Multi-class KNN model
│   ├── logistic_regressor_binary.pkl
│   ├── logistic_regressor_multi.pkl
│   ├── linear_regressor_binary.pkl
│   ├── linear_regressor_multi.pkl
│   ├── lsvm_binary.pkl              # Binary Linear SVM model
│   ├── lsvm_multi.pkl               # Multi-class Linear SVM model
│   ├── mlp_binary.pkl               # Binary MLP model
│   ├── mlp_multi.pkl                # Multi-class MLP model
│   ├── random_forest_binary.pkl     # Binary RF model
│   └── random_forest_multi.pkl      # Multi-class RF model
├── predictions/
│   ├── dt_real_pred_bin.csv         # Decision Tree binary predictions
│   ├── dt_real_pred_multi.csv       # Decision Tree multi-class predictions
│   ├── knn_real_pred_bin.csv        # KNN binary predictions
│   └── ...                          # (Other model predictions)
├── plots/
│   ├── Pie_chart_binary.png         # Binary class distribution
│   ├── Pie_chart_multi.png          # Multi-class distribution
│   ├── correlation_matrix_bin.png   # Feature correlation (binary)
│   ├── correlation_matrix_multi.png # Feature correlation (multi-class)
│   └── ...                          # (Model performance plots)
├── labels/
│   ├── le1_classes.npy              # Label encoder for binary classification
│   └── le2_classes.npy              # Label encoder for multi-class classification
└── UNSW_Project - Copy.ipynb        # Main Jupyter notebook with analysis and training
```

## Models Implemented

The project includes the following machine learning models:

1. **Decision Tree (DT)** - Fast, interpretable tree-based classifier
2. **K-Nearest Neighbors (KNN)** - Distance-based classifier
3. **Logistic Regression** - Linear probabilistic classifier
4. **Linear Regression** - Baseline regression approach
5. **Linear SVM (LSVM)** - Linear support vector machine for classification
6. **Multi-Layer Perceptron (MLP)** - Neural network classifier
7. **Random Forest (RF)** - Ensemble tree-based classifier

Each model is trained on both binary and multi-class datasets.

## Dataset Information

- **Total Records**: ~250,000 network flow samples
- **Features**: 49 network traffic attributes including:
  - Source/destination IP addresses and ports
  - Protocol information
  - Connection duration and bytes transferred
  - Packet counts and rates
  - TCP flags and connection states
  
- **Classes**:
  - Binary: Normal (0) vs. Attack (1)
  - Multi-class: Normal + 8 attack types

## Usage

### Prerequisites

- Python 3.x
- Jupyter Notebook
- Required libraries: pandas, numpy, scikit-learn, matplotlib, seaborn

### Running the Project

1. Clone or open this repository
2. Open the Jupyter notebook:
   ```bash
   jupyter notebook "UNSW_Project - Copy.ipynb"
   ```
3. Run the cells to:
   - Load and explore the dataset
   - Preprocess and normalize features
   - Train all models
   - Evaluate performance metrics
   - Generate visualizations

### Making Predictions

Use the trained models from the `models/` folder to make predictions on new network traffic data:

```python
import pickle
import pandas as pd

# Load a trained model
with open('models/random_forest_binary.pkl', 'rb') as f:
    model = pickle.load(f)

# Load label encoder
with open('labels/le1_classes.npy', 'rb') as f:
    label_encoder = pickle.load(f)

# Make predictions
predictions = model.predict(X_new)
```

## Results

Performance predictions and metrics are stored in:
- `predictions/` - CSV files containing model predictions for each classifier
- `plots/` - Visualization plots including confusion matrices and performance comparisons

## Key Files

- **UNSW_Project - Copy.ipynb**: Complete analysis pipeline with data exploration, preprocessing, model training, and evaluation
- **output (1).html**: Generated HTML report of analysis results

## License

This project is based on the UNSW-NB15 dataset. Please refer to the original dataset documentation for usage rights.

## References

- UNSW-NB15 Dataset: https://www.unsw.adfa.edu.au/unsw-canberra-cyber/cybersecurity/UNSW-NB15-Datasets/
- Original Paper: Moustafa, N., & Slay, J. (2015). UNSW-NB15: A comprehensive data set for network intrusion detection systems.

## Contributing

Feel free to fork, modify, and improve this project.

---

**Last Updated**: December 2025
