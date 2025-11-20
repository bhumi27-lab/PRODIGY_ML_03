# Cat vs Dog Image Classification using SVM

A machine learning project that classifies images of cats and dogs using a Support Vector Machine (SVM) model with complete preprocessing, training, evaluation, and saved-model prediction workflow.

---

## Project Overview

This project demonstrates how classical machine learning algorithms like SVM can be applied to image classification by preprocessing images and converting them into numerical feature vectors.

The pipeline includes:
- Image loading
- Resizing and grayscale conversion
- Flattening image pixels
- Feature scaling
- SVM model training
- Model evaluation
- Saving the trained model
- Predicting new uploaded images

---

## Dataset

Source: Kaggle — Cat and Dog Dataset by tongpython

Folder structure after extraction:

cat_and_dog/
 └── training_set/
      └── training_set/
           ├── cats/
           └── dogs/

---

## Technologies Used

- Python
- Google Colab
- NumPy
- OpenCV
- Scikit-learn
- Matplotlib
- joblib
- tqdm

---

## Workflow

### 1. Image Preprocessing
- Resized images to 64×64
- Converted to grayscale
- Flattened to 4096 features

### 2. Train/Test Split
- 80% training
- 20% testing
- Stratified sampling

### 3. Feature Scaling
Used StandardScaler() to normalize pixel values.

### 4. Model Training
SVM with RBF kernel:

SVC(kernel='rbf', C=3, gamma='scale')

### 5. Model Evaluation
Computed:
- Accuracy
- Precision
- Recall
- F1-score
- Classification report

### 6. Visualization
Displayed a random test image with its predicted label.

---

## Saving the Model

joblib.dump(model, "svm_cat_dog_model.joblib")
joblib.dump(scaler, "scaler_cat_dog.joblib")

---

## Loading the Model

loaded_model = joblib.load("svm_cat_dog_model.joblib")
loaded_scaler = joblib.load("scaler_cat_dog.joblib")

---

## Prediction on New Images

Steps:
- Upload image
- Preprocess
- Scale
- Predict (Cat/Dog)
- Visualize using Matplotlib

---

## Repository Structure

Task3/
├── notebook.ipynb
├── svm_cat_dog_model.joblib
├── scaler_cat_dog.joblib
├── .gitignore
├── LICENSE
└── README.md

---

## Future Improvements

- Use HOG features
- Train a CNN for better accuracy
- Deploy using Streamlit
- Add data augmentation
