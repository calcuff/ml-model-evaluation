# Machine Learning Model Evaluation

This project provides a comprehensive evaluation of several supervised machine learning algorithms on challenging datasets. The goal is to compare the generalization performance of different models and analyze their strengths and weaknesses under varying data conditions.

## Overview

We benchmark the following algorithms:

- **K-Nearest Neighbors**
- **Decision Tree**
- **Random Forest**
- **Neural Network**

Each model is evaluated across one or more datasets with characteristics such as class imbalance, high dimensionality, and noisy features.

---
## Objectives

- Assess model performance using accuracy, precision, recall, F1-score, and confusion matrices.
- Explore the effect of dataset complexity on model generalization.
- Highlight trade-offs in bias-variance and model interpretability.
- Provide reproducible code and structured analysis for future comparison studies.

---
## Running

### Installation
Install required dependencies:

```bash
pip install -r requirements.txt
```

### Project Structure

- Model implementations are located in [src/](src/)

- Experiment scripts are located in [experiments/](experiments/), organized by dataset

### Running an Experiment

Each Python script under experiments/dataset/ will:

1. Load the appropriate dataset
    - Normalize numerical features
    - One-hot encode categorical features
    - Split into train/validation sets using stratified 10-fold cross-validation

2. Perform a hyperparameter search
    - Search criteria depend on the model (e.g., k in KNN, depth in Decision Trees, etc.)

3. Train and evaluate the model
    - For each hyperparameter configuration
    - Evaluate performance on the validation set using accuracy and F1-score

4. Store results
    - Results are saved as .csv files, ex: [/experiments/credit/results/credit-nn-results.csv](/experiments/credit/results/credit-nn-results.csv)

5. Generate plots
    - Performance graphs are saved as .png files, ex: [/experiments/credit/results/credit-nn-results.png](/experiments/credit/results/credit-nn-results.png)

#### Note: python scripts must be run from with the experiments/dataset directory

``` bash
cd experiments/digits
python3 knn_digits.py
```

---
### 📚 Datasets

This project benchmarks classification models across four datasets, each offering distinct challenges in terms of data structure, dimensionality, and learning complexity.

---

### 🔢 1. Handwritten Digits Recognition

|                                |                                  |
|--------------------------------|----------------------------------|
| **Task**     | Multiclass (0–9 digit classification)       |
| **Instances** | 1,797 samples                              |
| **Features**  | 64 grayscale pixels (8×8 image)             |
| **Goal**      | Predict the digit depicted in the image     |
| **Source**    | [`sklearn.datasets.load_digits`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) |

<p align="center">
  <img src="https://scikit-learn.org/stable/_images/sphx_glr_plot_digits_classification_001.png" width="300" alt="Handwritten Digits Example"/>
</p>

---

### 🧬 2. Oxford Parkinson’s Disease Detection

|                                |                                  |
|--------------------------------|----------------------------------|
| **Task**     | Binary classification (healthy vs Parkinson’s) |
| **Instances** | 195 voice recordings                        |
| **Features**  | 22 vocal frequency-based attributes         |
| **Goal**      | Predict disease status from voice data       |
| **Source**    | Provided in assignment zip file             |

> Small-scale biomedical dataset used to evaluate generalization on noisy, real-world data.

---

### 🌾 3. Rice Grains Dataset

|                                |                                  |
|--------------------------------|----------------------------------|
| **Task**     | Binary classification (Cammeo vs Osmancik)   |
| **Instances** | 3,810 rice grain samples                    |
| **Features**  | 7 morphological measurements                |
| **Goal**      | Identify rice species by grain shape        |
| **Source**    | Provided in assignment zip file             |

> Medium-scale dataset used to evaluate classification performance on compact, well-behaved numerical features.

---

### 💳 4. Credit Approval Dataset

|                                |                                  |
|--------------------------------|----------------------------------|
| **Task**     | Binary classification (approve or deny)     |
| **Instances** | 653 applications                           |
| **Features**  | 6 numerical + 9 categorical features        |
| **Goal**      | Predict credit approval outcome             |
| **Source**    | Provided in assignment zip file             |

> This dataset includes categorical features that require one-hot encoding prior to training.

---

### 📊 Dataset Summary Table

| Dataset              | Task               | # Samples | # Features | Feature Types             |
|----------------------|--------------------|-----------|------------|----------------------------|
| Handwritten Digits   | Multiclass (10-way) | 1,797     | 64         | Numerical (image pixels)   |
| Parkinson’s Voice    | Binary              | 195       | 22         | Numerical (biomedical)     |
| Rice Grains          | Binary              | 3,810     | 7          | Numerical (morphological)  |
| Credit Approval      | Binary              | 653       | 15         | Numerical + Categorical    |

---

**Note:** Datasets containing categorical attributes (e.g., the Credit Approval dataset) were preprocessed using **one-hot encoding** to ensure compatibility with models like neural networks that require fully numerical input.
