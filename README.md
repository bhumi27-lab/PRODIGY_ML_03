🐾 Cat vs Dog Image Classification using SVM

A machine learning project that classifies images of cats and dogs using a Support Vector Machine (SVM) model with complete preprocessing, training, evaluation, and saved-model prediction workflow.

📌 Project Overview

This project demonstrates how classical machine learning algorithms like SVM can be applied to image classification by preprocessing images and converting them into numerical feature vectors.

The pipeline includes:

Image loading

Resizing and grayscale conversion

Flattening image pixels

Feature scaling

SVM model training

Model evaluation

Saving the trained model

Predicting uploaded images

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

Resize images to 64×64

Convert to grayscale

Flatten to a 4096-pixel vector

2️⃣ Train/Test Split

80% for training

20% for testing

Stratified split

3️⃣ Feature Scaling

Standardized using:

StandardScaler()

4️⃣ Model Training

Used an RBF-kernel SVM:

SVC(kernel='rbf', C=3, gamma='scale')

5️⃣ Evaluation

Accuracy

Precision

Recall

F1-score

Classification report

6️⃣ Visualization

Random test image displayed with predicted label.

💾 Saving the Model
joblib.dump(model, "svm_cat_dog_model.joblib")
joblib.dump(scaler, "scaler_cat_dog.joblib")

🔁 Loading the Model
loaded_model = joblib.load("svm_cat_dog_model.joblib")
loaded_scaler = joblib.load("scaler_cat_dog.joblib")

🖼 Predicting New Images

Uploaded images are:

Resized

Converted to grayscale

Flattened

Scaled

Classified as Cat or Dog

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

Add HOG features

Replace SVM with a CNN

Deploy using Streamlit

Add data augmentation
