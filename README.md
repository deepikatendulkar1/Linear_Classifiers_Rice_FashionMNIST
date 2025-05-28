# Linear Classifiers from Scratch: Rice & Fashion-MNIST

## Project Overview
This project implements fundamental linear classification algorithms from scratch using NumPy and scientific Python libraries. It includes both binary classification (Rice dataset) and multi-class classification (Fashion-MNIST). Final results are evaluated on train/val/test splits and Kaggle submissions.

## Implemented Models
- logistic.py – Logistic Regression (binary)
- perceptron.py – Perceptron Classifier
- svm.py – Support Vector Machine
- softmax.py – Softmax Regression

## 📁 Project Structure

Linear_Classifiers_Rice_FashionMNIST/  
├── Linear_Classifiers_Rice_FashionMNIST.ipynb     # Main notebook  
├── data_process.py                                # Dataset loading & preprocessing  
├── kaggle_submission.py                           # Utility to create Kaggle submission files  
├── models/                                        # All classifier implementations  
│   ├── logistic.py  
│   ├── perceptron.py  
│   ├── svm.py  
│   └── softmax.py  
├── kaggle/                                        # Kaggle submission outputs  
│   ├── perceptron_submission_fashion.csv  
│   ├── svm_submission_fashion.csv  
│   └── softmax_submission_fashion.csv  

## Datasets
- **Rice Dataset**: Binary classification dataset with 0/1 labels.  
  ⚠️ *Note: This dataset is not included in the repository*

- **Fashion-MNIST**: Multi-class image dataset (0–9 categories).  
  ⚠️ *Note: Fashion-MNIST files are not included. 

## How to Run
1. Install required libraries:
    pip install numpy pandas matplotlib scikit-learn
2. Download datasets as needed.
3. Open the notebook:
    jupyter notebook Linear_Classifiers_Rice_FashionMNIST.ipynb

## Results

### Rice Dataset (Binary Classification)
Perceptron: 99.92% test accuracy  
SVM: 83.15% test accuracy  
Softmax: 83.12% test accuracy  
Logistic Regression: 75.20% test accuracy  

### Fashion-MNIST (Multi-class Classification)
Perceptron: 79.41% test accuracy  
SVM: 81.02% test accuracy  
Softmax: 79.33% test accuracy  

Kaggle submission files for Fashion-MNIST are available in the kaggle/ folder.

## Skills Demonstrated
- Machine Learning from scratch
- Binary and Multi-class classification
- NumPy-based optimization
- Hyperparameter tuning: learning rate, epochs, regularization
- Scientific computing with Python
- Jupyter Notebook for experimentation
- Kaggle submission workflow

## Contact
Name: Deepika Hemant Tendulkar  
Email: deepikatenduulkar5@gmail.com  
LinkedIn: https://www.linkedin.com/in/deepika-tendulkar-a88bb8166/
