# 💓 Decoding Arrhythmias: Atrial Fibrillation Detection Using Hybrid Machine Learning Models

## 🌟 Overview
This repository explores the detection of **Atrial Fibrillation (AF)**, the most common cardiac arrhythmia, using advanced machine learning techniques. By leveraging the PhysioNet/Computing in Cardiology Challenge 2017 dataset, multiple models, including Support Vector Machines (SVM), LightGBM, Convolutional Neural Networks (CNN), and a Hybrid approach, were developed and evaluated to classify AF from ECG signals.

The **Hybrid Model**, combining SVM and LightGBM, emerged as the most effective, achieving superior performance in handling the complexities of AF detection.

---

## ✨ Key Features
- **Multi-Model Approach**: Includes SVM, LightGBM, CNN, and a Hybrid model to address imbalanced multi-class classification.
- **Robust Preprocessing**:
  - Noise filtering with bandpass filters.
  - Sliding window segmentation for time-series data.
  - Signal inversion correction and feature scaling.
- **Comprehensive Evaluation**:
  - Metrics: Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
  - Visualization through confusion matrices and ROC curves.

---

## 🛠️ Methodology
### 🔍 **Preprocessing Pipeline**
1. Bandpass filtering (0.5–50 Hz) to remove noise.
2. Z-score normalization for standardizing ECG signals.
3. Sliding window segmentation to capture temporal patterns.
4. Diverse feature extraction:
   - Time-domain features: RR intervals, SDNN, RMSSD.
   - Frequency-domain features: Power spectral density.
   - Statistical features: Skewness, Kurtosis, Entropy.
5. Addressed class imbalance using **stratified sampling** and **median-based imputation** for missing values.

### 🤖 **Models Implemented**
- **Dual Support Vector Machines (SVM)**:
  - Linear and RBF kernels optimized through GridSearchCV.
- **LightGBM**:
  - Gradient-boosting framework optimized for multiclass classification.
- **Convolutional Neural Networks (CNN)**:
  - Deep learning model for feature extraction and classification.
- **Hybrid Model**:
  - Combines SVM and LightGBM, leveraging complementary strengths for superior performance.

---

## 📈 Results
| **Model**        | **Accuracy** | **Precision** | **Recall** | **F1 Score** |
|-------------------|--------------|---------------|------------|--------------|
| Dual SVM         | 72.36%       | 71.13%        | 72.36%     | 69.86%       |
| LightGBM         | 75.95%       | 74.78%        | 75.95%     | 74.48%       |
| CNN              | 70.54%       | 68.01%        | 70.54%     | 66.78%       |
| **Hybrid Model** | **76.15%**   | **74.94%**    | **76.15%** | **74.70%**   |

The **Hybrid Model** demonstrated balanced performance across all metrics, outperforming individual models in handling imbalanced data.

---

## 🌍 Real-World Applications
- **Atrial Fibrillation Diagnostics**:
  - Supports early and accurate detection of AF to reduce stroke and heart failure risks.
- **Clinical and Remote Monitoring**:
  - Facilitates real-time detection through wearable devices.
- **Generalized Cardiac Analysis**:
  - Applicable to other arrhythmia detection tasks.

---

## 🚀 Future Scope
- **Incorporate Deep Learning**:
  - Explore advanced architectures like RNNs and Transformers for raw ECG signal analysis.
- **Address Class Imbalance**:
  - Experiment with cost-sensitive learning and synthetic data augmentation.
- **Personalized Models**:
  - Develop patient-specific models for tailored healthcare solutions.
- **Integration with Multimodal Data**:
  - Combine ECG with heart rate, blood pressure, and other physiological signals for improved accuracy.

---

## 🔧 Tools and Technologies
- **Programming**: Python
- **Libraries**: NumPy, SciPy, LightGBM, scikit-learn, TensorFlow/Keras
- **Dataset**: PhysioNet/Computing in Cardiology Challenge 2017

---

## 📜 Authors
- **Likhitha Marrapu**  
- Team Members: Chandini Karrothu, Shivani Battu  

