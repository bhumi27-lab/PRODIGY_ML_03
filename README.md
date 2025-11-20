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


Only the labeled training set was used.

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

Resize to 64×64

Convert to grayscale

Flatten to 4096 pixels

2️⃣ Train/Test Split

80% train

20% test

Stratified sampling

3️⃣ Scaling

StandardScaler used for normalization.

4️⃣ Model Training

Using SVM with RBF kernel:

SVC(kernel='rbf', C=3, gamma='scale')

5️⃣ Evaluation

Includes accuracy + full classification report.

6️⃣ Visualization

Random image from test set displayed with predicted label.

💾 Saving the Model
joblib.dump(model, "svm_cat_dog_model.joblib")
joblib.dump(scaler, "scaler_cat_dog.joblib")

🔁 Loading the Model
loaded_model = joblib.load("svm_cat_dog_model.joblib")
loaded_scaler = joblib.load("scaler_cat_dog.joblib")

🖼 Predicting New Images

Uploaded images are:

Resized

Grayscaled

Flattened

Scaled

Classified

📁 Repository Structure
Task3/
│
├── notebook.ipynb
├── svm_cat_dog_model.joblib
├── scaler_cat_dog.joblib
├── .gitignore
├── LICENSE
└── README.md

🚀 Future Improvements

Use HOG features

Replace SVM with CNN

Deploy with Streamlit

Add augmentation
