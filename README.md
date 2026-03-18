# KNN Performance Analysis on Fashion-MNIST

This project presents a comprehensive analysis of the **K-Nearest Neighbors (KNN)** algorithm on the Fashion-MNIST dataset. The study evaluates how different factors such as **training size (N)**, **number of neighbors (k)**, **sampling strategy**, and **distance metrics** affect model performance.

---

## Project Overview

The goal of this project is to systematically analyze KNN performance across:

- Varying training dataset sizes (N)
- Different values of k (number of neighbors)
- Sampling strategies:
  - Random (stratified)
  - Sequential (index-based)
- Distance metrics:
  - Euclidean (L2)
  - Manhattan (L1)

Both **global accuracy** and **local (per-class) accuracy** are evaluated.

---

## Dataset

- **Dataset**: Fashion-MNIST
- **Total Samples**: 70,000
- **Classes**: 10
- **Image Size**: 28×28 (flattened to 784 features)

### Classes:
- T-shirt/top  
- Trouser  
- Pullover  
- Dress  
- Coat  
- Sandal  
- Shirt  
- Sneaker  
- Bag  
- Ankle boot  

---

## Methodology

### 1. Data Preprocessing
- Combined train + test sets
- Flattened images (28×28 → 784)
- Normalized pixel values to [0,1]

---

### 2. KNN Implementation

- Distance computed using `pairwise_distances`
- Batch processing used for memory efficiency
- Predictions via **majority voting**
- Evaluated for:
  - `k ∈ {1, 2, 5, 10, 15, 20}`
  - `N ∈ [1000, 6000]` (per class)

---

### 3. Experiments

#### Question 1: Effect of Training Size (N) and k

- Used **random stratified sampling**

**Observations:**
- Accuracy improves as **N increases**
- Optimal k ≈ **5** across all N
- Very small k → noisy predictions
- Very large k → oversmoothing

---

#### Question 2: Sampling Strategy (Random vs Sequential)

- Compared:
  - Random sampling (shuffled per class)
  - Sequential sampling (original order)

**Key Findings:**
- Random sampling consistently performs **slightly better**
- Difference increases with larger N
- Best k remains **5** for both methods
- Sequential sampling introduces **mild bias**

---

#### Question 3: Distance Metric (L1 vs L2)

- Compared:
  - Euclidean (L2)
  - Manhattan (L1)

**Key Findings:**
- L1 performs slightly better at **smaller N**
- L2 catches up or slightly outperforms at **larger N**
- Differences are **small but consistent**
- Certain classes (e.g., Sandal, Shirt) are more sensitive to metric choice

---

## Results Summary

### Global Trends
- Increasing training size → better performance
- Optimal k = **5**
- Random sampling > Sequential sampling
- L1 and L2 differences are minor but noticeable

---

### Best Accuracy Achieved
- **~86.8%** (Random sampling, L2, k=5, N=6000)

---

### Per-Class Insights
- High accuracy classes:
  - Trouser, Sneaker
- Challenging classes:
  - Shirt, Coat (due to visual similarity)

---

## Output Files

- `q1_results.csv` → Random sampling results  
- `q2_results.csv` → Sequential sampling results  
- `q3_results.csv` → Manhattan distance results  

---

## Visualizations

The project includes:
- Global accuracy vs N plots
- Local (per-class) accuracy plots
- Random vs Sequential comparison graphs
- L1 vs L2 comparison graphs

---

## How to Run

```bash
# Install dependencies
pip install numpy pandas matplotlib scikit-learn keras scipy

# Run the script
python knn_analysis.py
