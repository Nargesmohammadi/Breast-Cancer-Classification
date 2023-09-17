

## Breast Cancer Classification with Neural Network

This repository contains code for a breast cancer classification project using a neural network. The goal of this project is to predict whether a tumor is malignant or benign based on a set of input features.



### Dataset

The breast cancer dataset used in this project is sourced from the [sklearn.datasets](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) module in scikit-learn. It consists of samples with various features related to breast cancer tumors.



### Installation

In this repository I followed these steps:


1. Install the required dependencies:

   ```
   pip install numpy pandas matplotlib scikit-learn tensorflow
   ```


### Usage

1. Import the necessary libraries:

   ```python
   import numpy as np
   import pandas as pd
   import matplotlib.pyplot as plt
   import sklearn.datasets
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import StandardScaler
   import tensorflow as tf
   from tensorflow import keras
   ```

2. Load the breast cancer dataset:

   ```python
   breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
   ```

3. Prepare the data:

   ```python
   # Create a DataFrame from the dataset
   data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)
   data_frame['label'] = breast_cancer_dataset.target

   # Split the data into training and testing sets
   X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

   # Standardize the input data
   scaler = StandardScaler()
   X_train_std = scaler.fit_transform(X_train)
   X_test_std = scaler.transform(X_test)
   ```

4. Build and train the neural network model:

   ```python
   model = keras.Sequential([
       keras.layers.Flatten(input_shape=(30,)),
       keras.layers.Dense(20, activation='relu'),
       keras.layers.Dense(2, activation='sigmoid')
   ])

   model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])

   history = model.fit(X_train_std, Y_train, validation_split=0.1, epochs=10)
   ```

5. Evaluate the model:

   ```python
   loss, accuracy = model.evaluate(X_test_std, Y_test)
   print("Accuracy:", accuracy)
   ```

6. Make predictions:

   ```python
   input_data = np.array([17.14, 16.4, 116, 912.7, 0.1186, 0.2276, 0.2229, 0.1401, 0.304, 0.07413, 1.046, 0.976,
                          7.276, 111.4, 0.008029, 0.03799, 0.03732, 0.02397, 0.02308, 0.007444,
                          22.25, 21.4, 152., 1461, 0.1545, 0.3949, 0.3853, 0.255, 0.4066, 0.1059])

   input_data_reshaped = input_data.reshape(1, -1)
   input_data_std = scaler.transform(input_data_reshaped)
   prediction = model.predict(input_data_std)

   if np.argmax(prediction) == 0:
       print("The tumor is Malignant")
   else:
       print("The tumor is Benign")
   ```

   

### Accuracy Graph ![image](https://github.com/Nargesmohammadi/Breast-Cancer-Classification_with_Neural_Network/assets/96385230/3a8b19b0-8c47-4cb4-a819-0c39b82d904f)



### Loss Graph  ![Uploading image.png…]()



### Acknowledgments

- This project was inspired by the breast cancer classification problem.
- The breast cancer dataset used in this project was sourced from scikit-learn.


