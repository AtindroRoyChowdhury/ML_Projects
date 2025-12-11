
## Predicting Term Deposit Subscription — Bank Marketing Analysis

## Project Overview
This project analyzes the **Bank Marketing dataset (UCI Repository ID 222)** to identify the factors influencing whether a client subscribes to a term deposit.  
Dataset size: **41,188 records, 17 attributes**

**Goal:**  
Perform **EDA and ML modeling** to predict term deposit subscription and derive actionable marketing insights.

---

## 1. Data Loading & Setup
```python
!pip install ucimlrepo
from ucimlrepo import fetch_ucirepo
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from scipy.stats import pointbiserialr, chi2_contingency

bank_marketing = fetch_ucirepo(id=222)
X = bank_marketing.data.features
y = bank_marketing.data.targets
df = pd.concat([X, y], axis=1)
```
---

## 2. Data Cleaning & Feature Engineering
- Converted categorical → numeric
- Engineered `contacted_before` feature
- Filled missing values logically

**Insight:**  
Clients contacted previously are **more likely to subscribe**.

---

## 3. Exploratory Data Analysis

### Categorical Features (Cramér’s V)
| Feature | V | Impact |
|----------|--|---------|
| `poutcome` | 0.31 | Past campaign success |
| `month` | 0.26 | Seasonality |
| `contact` | 0.15 | Type of contact |
| `housing` | 0.14 | Loan indicator |

### Numerical Features (Point-Biserial)
| Feature | Corr | Observation |
|----------|------|--------------|
| `campaign` | -0.07 | Too many contacts reduce success |
| `balance` | +0.05 | Higher balance = slight increase |
| `previous` | +0.09 | Past engagement helps |

---

## 4. Modeling — Predicting Subscription

### Random Forest
Accuracy: **0.89**, ROC-AUC: **0.80**  
Recall (Positive class): **0.23** → Poor for minority class

### XGBoost (with class weights)
ROC-AUC: **0.80**, Recall improved to **0.66**  
Optimal threshold **≈ 0.48** balanced recall and precision.

---

## 5. Insights
- **Past contact and campaign outcome** are strong predictors.
- **Over-contacting** reduces conversion.
- **Balanced XGBoost model** increases recall by 40%.

---

## 6. Takeaways
- Validated marketing effectiveness via statistics & ML.
- Improved subscriber prediction using **threshold tuning**.
- Can evolve into a **real-time campaign optimization tool**.

---

## 7. Tech Stack
**Python, Pandas, NumPy, SciPy, Seaborn, Scikit-Learn, XGBoost, UCI ML Repo**

---
