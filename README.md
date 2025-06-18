# 🧠 Machine Learning Project: Classification & Regression with Scikit-Learn & TensorFlow

This project demonstrates the end-to-end application of machine learning techniques on two real-world datasets: the **MNIST Handwritten Digit Dataset** (classification task) and a **Synthetic Housing Dataset** (regression task). It showcases model selection, dimensionality reduction, evaluation metrics, and ethical AI practices.

---

## 📌 Objectives

- Apply supervised learning for **image classification** (MNIST).
- Predict housing prices using **regression models**.
- Compare the performance of different algorithms.
- Reflect on **ethical considerations** in model development.

---

## 🧪 Technologies Used

- Python 3.x  
- [Scikit-Learn](https://scikit-learn.org/)  
- [TensorFlow/Keras](https://www.tensorflow.org/)  
- Matplotlib & Seaborn  
- NumPy & Pandas  

---

## 🖼️ Demo Visualizations

> Replace below with actual plots from your notebook.

- **Confusion Matrix - kNN Classifier**  
  ![Confusion Matrix - kNN](https://github.com/user-attachments/assets/e27ef521-9999-48d9-90ef-c47886e2c46f)

- **Model Comparison Chart**  
  ![Model Performance](https://github.com/user-attachments/assets/8a4d846c-d0e1-4205-8c4d-638bf609b779)

- **Actual vs Predicted (Regression)**  
  ![Regression Scatter Plot](https://github.com/user-attachments/assets/11a9a4f3-392f-4cb7-b47e-f6b35e5512cf)

---

## 🔍 Models & Highlights

### 🧮 Classification (MNIST)
- **Data Preprocessing**: StandardScaler + PCA (95% variance)
- **Models**:
  - k-Nearest Neighbors (k=10)
  - SVM (Linear & RBF with GridSearchCV)
  - Neural Network (2-layer, ReLU + Dropout, Softmax)

> ✅ Best Accuracy: **Neural Network – 96.96%**

### 📈 Regression (Housing Data)
- **Models**:
  - Linear Regression
  - Polynomial Regression (deg=2)
  - Ridge & Lasso Regression

> 🔍 **Best Generalization**: Ridge Regression (R² ≈ 0.61)  
> ⚠️ Polynomial Regression tended to overfit

---

## ✅ Evaluation Metrics

| Model            | Accuracy | Precision | Recall | F1-Score |
|------------------|----------|-----------|--------|----------|
| kNN              | 94.74%   | 94.75%    | 94.66% | 94.69%   |
| SVM (Linear)     | 93.48%   | 93.45%    | 93.41% | 93.42%   |
| SVM (RBF)        | 92.19%   | 92.35%    | 92.10% | 92.18%   |
| Neural Network   | **96.96%** | **96.95%** | **96.91%** | **96.93%** |

---

## ⚖️ Ethical Considerations

This project followed responsible AI principles:
- **Fairness**: Balanced datasets and unbiased preprocessing
- **Transparency**: Model explanations via metrics and visualizations
- **Accountability**: Clear documentation and evaluation logs
- Mentioned tools like SHAP/LIME for model explainability (not implemented but acknowledged)

---

