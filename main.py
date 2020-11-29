
from extraction_classification_SVM import *
from classification_from_model import *

"""
Note:
- use only constant as parameter of function start_classification_extraction_SVM (advice)
- the gamma parameter is setted empirically depending of the first Analysis that 
  evaluate the model predicting the training test: in this case a good gamma parameter 
  will produce 1.0 of Accuracy
"""


#classification from a  pre-trained model and a test dataset converted in CSV

start_classification_from_model(PATH_NAME_TO_MODEL,PATH_NAME_TO_TEST_SET)


#classification with feature extraction and then the evaluation of model
"""
N_FEATURES_DEFAULT = 26

GAMMA_DEFAULT_2 = 0.11111111 # good parameter for 26 features


ROOT_PATH_DATASET_TRAINING = "Dataset/train"
ROOT_PATH_DATASET_TEST = "Dataset/test"

training_X, labels_Y,test_X,labels_Y_test,model = start_classification_extraction_SVM(
    ROOT_PATH_DATASET_TRAINING,
    ROOT_PATH_DATASET_TEST,
    N_FEATURES_DEFAULT,
    GAMMA_DEFAULT_2)


model = load_model(PATH_NAME_TO_SAVE_MODEL)

print("--------28 FEATURES ANALYSIS OF TRAINING SET---------\n")
startAnalysis(training_X, model, labels_Y)
print("\n-----------------ANALYSIS DONE-----------------------\n\n")

print("--------28 FEATURES ANALYSIS OF TEST SET---------\n")
startAnalysis(test_X, model,labels_Y_test)
print("\n-----------------ANALYSIS DONE-----------------------\n\n")

"""

