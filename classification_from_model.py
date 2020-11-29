import numpy as np
import pickle
from constant import *
from extraction_classification_SVM import *

N_FEATURES = len(FEATURE_PART_NUMBER_CUR)

def get_data_test_X(feature_column,n_feature):
    f = convert_to_np_array(feature_column)
    f_tuplate = tuple(f)
    training_X = np.vstack(f_tuplate).T
    return training_X

def get_value(data):
    value = initialize_list_column_feature(N_FEATURES)

    for i in range(len(data)):
        for j in range (N_FEATURES):
            value[j].append(data[i][j+1])
    
    return get_data_test_X(value,N_FEATURES)

def get_labels(data):
    labels = []
    for i in range(len(data)):
        labels.append(data[i][0])
    return labels

def start_classification_from_model(file_name_model, file_name_csv_test):
    
    model = load_model (file_name_model)
    data =  np.loadtxt(fname = file_name_csv_test, delimiter = ' ')
    test_X = get_value(data)
    label_test_Y = get_labels(data)
    startAnalysis(test_X,model,label_test_Y)

