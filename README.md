# SVM-emotion-face-recognition
A simple way to create a SVM classifier to emotion recognition. 
There is a feature extractor that take into account only feature about 
some euclidean distance between landmarks point.

# Getting Started
#### launch main.py from prompt command line
```
python main.py
```
# Requirements
#### dlib 19.21.0
#### scikit-learn 0.23.2
#### seaborn 0.11.0
#### opencv-python 4.4.0.46

# Instruction to train the model

### 1. once clone the repository download shape_predictor_68_face_landmarks.dat
### 2. create the Dataset of images containing a part of training and a part of test respecting this tree diagram

![](img-readme1.png)
