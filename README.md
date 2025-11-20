

🐾 Cat vs Dog Image Classification using SVM

A machine learning project that classifies images of cats and dogs using a Support Vector Machine (SVM) model with complete preprocessing, training, evaluation, and saved-model prediction workflow.

📌 Project Overview

This project demonstrates how classical machine learning algorithms like SVM can be applied to image classification by preprocessing images and converting them into numerical feature vectors.

The pipeline includes:

Image loading

Resizing & grayscale conversion

Flattening

Feature scaling

SVM model training

Model evaluation

Saving the trained model

Predicting new uploaded images

📂 Dataset

Source: Kaggle — Cat and Dog Dataset by tongpython

Folder structure after extraction:

cat_and_dog/
    └── training_set/
          └── training_set/
                ├── cats/
                └── dogs/


Only the labeled training set was used for training and testing.

🔧 Technologies Used

Python

Google Colab

NumPy

OpenCV

Scikit-learn

Matplotlib

joblib

tqdm

🧠 Workflow
1️⃣ Image Preprocessing

Resize images to 64×64

Convert to grayscale

Flatten to 4096-pixel vectors

2️⃣ Train/Test Split

80% training

20% testing

Stratified sampling

3️⃣ Feature Scaling

SVM requires normalized inputs → used StandardScaler().

4️⃣ Model Training

SVM with RBF kernel:

SVC(kernel='rbf', C=3, gamma='scale')

5️⃣ Model Evaluation

Includes:

Accuracy

Precision

Recall

F1-score

Classification report

6️⃣ Visualization

Random test image is shown along with predicted label.

💾 Saving the Model
joblib.dump(model, "svm_cat_dog_model.joblib")
joblib.dump(scaler, "scaler_cat_dog.joblib")

🔁 Loading the Model
loaded_model = joblib.load("svm_cat_dog_model.joblib")
loaded_scaler = joblib.load("scaler_cat_dog.joblib")

🖼 Predicting on New Images

Upload image

Preprocess

Flatten

Scale

Predict (Cat/Dog)

Display result via Matplotlib

📁 Repository Structure
Task3/
│
├── notebook.ipynb
├── svm_cat_dog_model.joblib
├── scaler_cat_dog.joblib
├── .gitignore
├── LICENSE
└── README.md

📌 Learning Outcomes

Image preprocessing

Applying SVM to high-dimensional data

Scaling and normalization

Model evaluation

Saving & loading ML models

Prediction on new data

End-to-end ML workflow

🚀 Future Improvements

Add HOG features

Replace SVM with CNN

Deploy using Streamlit

Add data augmentation
