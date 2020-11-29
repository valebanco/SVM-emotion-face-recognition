from sklearn import svm
from sklearn import metrics
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns

import dlib
import cv2

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import distance

import pickle
from constant import *


"""
ho 7,10 e 15 e 22 features

"""


FEATURE_PART_NUMBER_CUR = FEATURE_PART_NUMBER_2 #this constant is used to change set (in constant.py) of custom feature to work

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # model to recognize faces in input images

def get_distance(shape_p1,shape_p2):
    a = (shape_p1.x,shape_p1.y)
    b = (shape_p2.x,shape_p2.y)
    return distance.euclidean(a,b)


def initialize_list_column_feature(n_features):
    
    list_column_feature = []

    for i in range(n_features):
        l = []
        list_column_feature.append(l)
    
    return list_column_feature

def n_features_is_in_range(n_features,feature_set):
    return n_features <= len(feature_set)

def convert_to_np_array(features):
    for el in features:
        el = np.array(el)

    return features
    
def get_data_X(feature_column,N_FEATURE_TO_EXTRACT):
    f = convert_to_np_array(feature_column)
    f_tuplate = tuple(f)
    training_X = np.vstack(f_tuplate).T
    return training_X

"""
INPUT:
path file --> string that represent the path of the dataset composed of image file
N_FEATURE_TO_EXTRACT --> integer number constant 

OUTPUT:
set_X --> vstack representation of feature_map
labelstmp --> classification labels for each image
feature_columns --> list of list of features of dimension n_samples x n_feature

function that calculate euclidean distance among pairs of feature point landmarks
"""
def extract_N_feature_from_path(path_file,N_FEATURE_TO_EXTRACT):

    if(n_features_is_in_range(N_FEATURE_TO_EXTRACT,FEATURE_PART_NUMBER_CUR)):

        feature_column = initialize_list_column_feature(N_FEATURE_TO_EXTRACT)
        feature_part = FEATURE_PART_NUMBER_CUR[0:N_FEATURE_TO_EXTRACT]
        switch_class = 0
        labelstmp = []

        for root,subdirs,fi in os.walk(path_file): 
            for s in subdirs:
                p = root + '/' + s + '\\*'
                files = glob.glob( p )
                for f in files:
                    item = cv2.imread(f)
                    detect = detector(item,1)
                    if(np.shape(detect) != (0,)):
                        shape = predictor(item,detect[0])
                        for i_feature in range(N_FEATURE_TO_EXTRACT): 
                            feature_column[i_feature].append(get_distance(shape.part(feature_part[i_feature][0]),shape.part(feature_part[i_feature][1])))
                        labelstmp.append(switch_class)
            
                switch_class = switch_class + 1
            
        
        set_X = get_data_X(feature_column,N_FEATURE_TO_EXTRACT)
        print("total img extracted----> ", len(set_X))
        return set_X,labelstmp,feature_column
    else:
        print("OUT OF NUMBER OF FEATURES")





"""
function that generate the string to write into file.test 
chosen for the representation of dataset
"""
def get_content_dataset(num_features,features,labels):
    # creo la stringa da scrivere nel file
    final_string = ""
    size_row_dataset = len(features[1])

    for i in range(size_row_dataset):
        final_string = final_string + str(labels[i])

        for j in range(num_features):
            final_string = final_string + " "+ str(features[j][i])
        
        final_string = final_string + "\n"

    return final_string


def print_confusion_matrix(y_predictions,x_predictions):
    ax = plt.subplot()
    cm = confusion_matrix(x_predictions,y_predictions)
    sns.heatmap(cm,annot = True, ax = ax)
    ax.set_xlabel("Label Predette")
    ax.set_ylabel("Label previste")
    ax.set_title("Matrice di confusione")
    plt.show()

def print_complexive_accuracy(test_labels,y_predictions):
    print("Accuracy ===>", metrics.accuracy_score(test_labels,y_predictions))


"""
The function start the evaluation of the model 
visualising accuracy,confusion matrix and classification report

INPUT:
set_X --> set of features for each sample in nvstack format
true_labels --> labels of the classification of set_X
"""
def startAnalysis(set_X, model,true_labels):
    label_predicted = model.predict(set_X)
    print_complexive_accuracy(true_labels,label_predicted)
    print_confusion_matrix(true_labels,label_predicted)
    print(classification_report(true_labels,label_predicted))



def save_dataset(content,name_file):
    f = open(name_file, "w")
    f.write(content)
    f.close()

def generate_dataset(path_to_save,n_features,feature_column,labels_Y):
    content_dataset = get_content_dataset(n_features,feature_column,labels_Y)
    save_dataset(content_dataset,path_to_save)

def generate_model(type_kernel,C_parameter, gamma_parameter,training_X,labels_Y):
    model = svm.SVC(kernel= type_kernel , C= C_parameter, gamma = gamma_parameter)
    model.fit(training_X, labels_Y)
    return model

def save_model(file_name,model):
   pickle.dump(model, open(file_name, 'wb'))

def load_model(file_name):
    loaded_model = pickle.load(open(file_name, 'rb'))
    return loaded_model

"""
INPUT:
path_directory_train --> path where we can find training images
path_directory_test --> path where we can find testing images
n_features --> number of feature we want to extract depending of the constant in file constant.py
gamma_parameter --> parameter that help us to generate a gaussian model SVM in a multiclassification issue

OUTPUT:
training_X --> data about feature of set of training
labels_Y --> labels of training_X set
test_X --> data about feature of set of test
labels_test_Y --> labels of test_X set
model_generated --> information of model generated
"""
 
def start_classification_extraction_SVM(path_directory_train,path_directory_test,n_features,gamma_parameter):
    

    training_X, labels_Y,fc= extract_N_feature_from_path(path_directory_train,n_features)
    
    """if you want to generate CSV file of test dataset"""
    #generate_dataset(PATH_NAME_TO_TRAIN_SET,n_features,fc,labels_Y)

    model_generated = generate_model('rbf',1.0,gamma_parameter,training_X,labels_Y)

    test_X, labels_test_Y,fc_test = extract_N_feature_from_path(path_directory_test,n_features)

    """if you want to generate CSV file of test dataset"""
    generate_dataset(PATH_NAME_TO_TEST_SET,n_features,fc_test,labels_test_Y)

    save_model(PATH_NAME_TO_MODEL,model_generated)
     
   

    return training_X,labels_Y,test_X,labels_test_Y,model_generated

