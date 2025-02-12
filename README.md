# Data Classification and Visualization Project

## Overview
This project visualises and classifies data using machine learning techniques such as Support Vector Machines (SVM), Principal Component Analysis (PCA), and Parallel Coordinate plots. We aim to explore dataset structures, improve classification performance, and visualize data distributions effectively.

---

## 1️⃣ Data Preprocessing
### Steps Taken:
- **Imported required libraries** (`pandas`, `seaborn`, `matplotlib`, `sklearn`).
- **Loaded the dataset** using `pd.read_csv()`.
- **Checked dataset structure** using:
  ```python
  df.head()
  df.columns
  ```
- **Fixed column name issues** by standardizing names:
  ```python
  df.columns = df.columns.str.lower().str.strip()
  ```
- **Encoded categorical labels** for classification using:
  ```python
  from sklearn.preprocessing import LabelEncoder
  encoder = LabelEncoder()
  df['species'] = encoder.fit_transform(df['species'])
  ```

---

## 2️⃣ Support Vector Machine (SVM) Decision Boundary
### Steps Taken:
- **Trained an SVM model:**
  ```python
  from sklearn.svm import SVC
  model = SVC(kernel='linear')
  model.fit(X_train, y_train)
  ```
- **Plotted the decision boundary** using a meshgrid and `plt.contourf()`.
- **Fixed missing imports (`ListedColormap`)**:
  ```python
  from matplotlib.colors import ListedColormap
  ```

---

## 3️⃣ Pairplot for Data Distribution
### Steps Taken:
- **Used `sns.pairplot()` to visualize class separability**:
  ```python
  import seaborn as sns
  sns.pairplot(df, hue='species')
  ```
- Helped understand how features are distributed and related.

---

## 4️⃣ PCA for 3D Visualization
### Steps Taken:
- **Reduced data dimensions using PCA**:
  ```python
  from sklearn.decomposition import PCA
  X_reduced = PCA(n_components=3).fit_transform(X)
  ```
- **Plotted 3D data visualization**:
  ```python
  from mpl_toolkits.mplot3d import Axes3D
  fig = plt.figure(figsize=(8, 6))
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y, cmap='viridis')
  ```

---

## 5️⃣ Parallel Coordinates Plot
### Steps Taken:
- **Used `parallel_coordinates()` to visualize multiple features:**
  ```python
  from pandas.plotting import parallel_coordinates
  parallel_coordinates(df, 'species', color=('#556270', '#4ECDC4', '#C7F464'))
  ```
- **Fixed KeyError (`'species' not found`)** by checking column names.

---

## 6️⃣ Train, Test, and Validation Splits
### Purpose:
To prevent overfitting and evaluate model performance.

- **Training Set (60-80%)** → Used to train the model.
- **Testing Set (20-30%)** → Evaluates model performance.
- **Validation Set (10-20%)** → Tunes hyperparameters.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## Conclusion
✅ **Fixed errors and missing dependencies**
✅ **Visualized data distributions using Pairplots, PCA, and Parallel Coordinates**
✅ **Implemented SVM for classification with decision boundary visualization**
✅ **Explained train-test-validation splitting for improved model performance**

🚀 This project provides a strong foundation for **data classification and visualization**.

Would you like me to refine any part further? 😊

