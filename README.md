🐾 Cat vs Dog Image Classification using SVM

A machine learning project that classifies images of cats and dogs using a Support Vector Machine (SVM) model with complete preprocessing, training, evaluation, and saved-model prediction workflow.

📌 Project Overview

This project demonstrates how classical machine learning algorithms like SVM can be applied to image classification by preprocessing images and converting them into numerical feature vectors.

The pipeline includes:

Image loading

Resizing & grayscale conversion

Flattening image pixels

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

Resize to 64×64

Convert to grayscale

Flatten → 4096 features

Store data + labels

2️⃣ Train/Test Split

80% training, 20% testing

Stratified for balanced classes

3️⃣ Feature Scaling

Standardized using StandardScaler() — essential for SVM performance.

4️⃣ Model Training

Used RBF kernel:

SVC(kernel='rbf', C=3, gamma='scale')

5️⃣ Model Evaluation

Printed:

Accuracy

Classification report

Precision, Recall, F1-score

6️⃣ Visualization

Displayed a random test image with predicted label.

💾 Saving the Model
joblib.dump(model, "svm_cat_dog_model.joblib")
joblib.dump(scaler, "scaler_cat_dog.joblib")


Both model and scaler are saved for future predictions.

🔁 Loading the Model
loaded_model = joblib.load("svm_cat_dog_model.joblib")
loaded_scaler = joblib.load("scaler_cat_dog.joblib")

🖼 Predicting on New Images

Users can upload any image, which is then:

Resized

Grayscaled

Flattened

Scaled

Classified as Cat or Dog

A Matplotlib preview shows the uploaded image with the predicted label.

📁 Repository Structure
Task3/
│
├── notebook.ipynb
├── svm_cat_dog_model.joblib
├── scaler_cat_dog.joblib
└── README.md

📌 Learning Outcomes

Applying SVM to high-dimensional data

Image preprocessing using OpenCV

Feature scaling and preparation

Training and evaluating ML models

Saving and loading models

Making predictions on new data

Building a clear ML project workflow

🚀 Future Improvements

Add HOG features for better performance

Train a CNN to achieve higher accuracy

Deploy model using Streamlit

Add data augmentation
