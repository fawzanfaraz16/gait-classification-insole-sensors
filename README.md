# Gait Classification Using Insole Sensor Data
A neural-network approach to classifying five distinct gait behaviours from insole pressure-sensor time-series data.

---
## Overview

Gait analysis is a key area of biomechanical research with applications in injury detection, rehabilitation monitoring, and movement science. This project trains and evaluates a Multi-Layer Perceptron (MLP) on windowed insole sensor data to classify the following gait classes:

| Class | Description |
|---|---|
| **Normal_walking** | Standard walking gait |
| **Injury_walking** | Gait pattern simulating or reflecting injury |
| **Stepping** | Stationary or in-place stepping |
| **Swaying** | Postural sway / balance movements |
| **Jumping** | Vertical jump movements |

The final model achieves **94.92% test accuracy** across all five classes.

---

## Highlights

- End-to-end pipeline from raw sensor data to trained classifier
- Sliding-window segmentation of time-series insole data
- Simple MLP architecture with strong classification performance
- Per-class precision, recall, and F1-score reported via full classification report
- Honest discussion of limitations and misclassification patterns

---

## Methodology

### Preprocessing

1. **Missing values** removed using `dropna()`.
2. **Label encoding** applied via a manual label-mapping dictionary.
3. **Feature scaling** using `StandardScaler`, fitted on the training set only and then applied to the validation and test sets to prevent data leakage.
4. **Sliding window segmentation** applied to the time-series data:
   - Window size: **20 frames**
   - Stride: **15 frames** (5-frame overlap per window)

### Data Split

The dataset was split randomly into three partitions:

| Partition | Share |
|---|---|
| Train | 80% |
| Validation | 10% |
| Test | 10% |

> **Note:** A random split was used rather than a stratified split. This is acknowledged as a limitation (see below).

### Model Architecture

| Component | Detail |
|---|---|
| Type | Multi-Layer Perceptron (MLP) |
| Layers | 3 fully connected (`Linear`) layers total |
| Hidden dimension | 64 |
| Output dimension | 5 (one per gait class) |
| Optimizer | Adam |
| Loss function | CrossEntropyLoss |
| Learning rate | 0.0001 |
| Batch size | 32 |
| Epochs | 20 |

Some limited experimentation informed the final configuration: the hidden dimension was increased from 8 to 64, learning rates of 0.001 and 0.0001 were compared, and the effect of standardisation on convergence was evaluated. These were not exhaustive searches; the values above represent the final selected configuration.

---

## Results

### Top-Line Metrics

| Metric | Train | Validation | Test |
|---|---|---|---|
| Loss | 0.1989 | 0.2069 | — |
| Accuracy | 93.67% | 94.31% | **94.92%** |

### Classification Report (Test Set)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Normal_walking | 0.91 | 0.96 | 0.94 | 516 |
| Injury_walking | 0.99 | 0.92 | 0.95 | 590 |
| Stepping | 0.86 | 0.98 | 0.91 | 322 |
| Swaying | 1.00 | 0.93 | 0.96 | 535 |
| Jumping | 0.97 | 0.98 | 0.98 | 341 |
| **Accuracy** |  |  | **0.95** | **2304** |
| **Macro avg** | 0.95 | 0.95 | 0.95 | 2304 |
| **Weighted avg** | 0.95 | 0.95 | 0.95 | 2304 |

**Jumping** was the strongest-performing class (F1 0.98), likely because vertical jump movements produce the most distinctive insole pressure signatures. **Stepping** had the lowest precision (0.86), indicating some confusion with other classes, while **Normal_walking** and **Injury_walking** also showed some overlap, likely due to similar pressure-distribution characteristics.

---

## Misclassification and Limitations

- **Similar gait classes:** Normal_walking, Injury_walking, and Stepping share overlapping biomechanical features, leading to some inter-class confusion.
- **Random split:** The train/validation/test split was not stratified, meaning class proportions may vary slightly across partitions.
- **Limited hyperparameter exploration:** Only a small number of configurations were tested. More systematic tuning (e.g., grid or Bayesian search) could improve results.
- **Architecture scope:** Only an MLP was evaluated. Temporal models such as LSTMs or 1D-CNNs may better capture sequential patterns in gait data.

---

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/fawzanfaraz16/gait-classification-insole-sensors.git
   cd gait-classification-insole-sensors
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the dataset**
   Obtain a compatible copy of the insole sensor dataset and update the dataset path in the notebook to match your local setup.

4. **Run the notebook**
   Open `gait_classification.ipynb` in Jupyter Notebook or JupyterLab and execute the cells sequentially.

---

## Requirements

- Python 3.8+
- PyTorch
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- torchinfo
- tqdm

A full list of pinned versions is provided in `requirements.txt`.

---

## Future Improvements

- Implement stratified splitting to ensure balanced class representation across partitions
- Explore temporal architectures (LSTM, 1D-CNN) that can model sequential dependencies within each window
- Conduct more systematic hyperparameter search over learning rate, hidden dimensions, and window parameters
- Evaluate on held-out or externally collected gait data to assess generalisation
- Apply data augmentation techniques suited to time-series sensor data

---

## Dataset Note

The original dataset is not included in this repository. To reproduce training and evaluation, users must provide their own compatible copy of the data and update the dataset path in the notebook accordingly.

---

## Academic Use

This project was developed for academic purposes. Please respect any dataset-specific usage restrictions if reusing or adapting this work.
```
