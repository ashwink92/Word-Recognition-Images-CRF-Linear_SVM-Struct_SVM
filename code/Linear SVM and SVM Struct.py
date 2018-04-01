import re
import collections
from subprocess import run, PIPE
import os
import matplotlib.pyplot as plt
import scipy, math
from scipy.ndimage import rotate
from crf_prediction import crf_predict
import numpy as np
from sklearn.svm import LinearSVC

#Initializing path variables to various training, test and intermediate result files
struct_model_path = "../data/model_trained.txt"
struct_test_predictions_path = "../data/test_predictions.txt"

struct_train_path = "../data/train_struct.txt"
struct_test_path = "../data/test_struct.txt"

linear_train_path = "../data/train.txt"
linear_test_path = "../data/test.txt"

plt.style.use('seaborn-paper')

letter_accuracies = []
word_accuracies = []

def train_svm_struct(C=1.0):
    #Training model Using Cornell's SVM HMM library
    args = ['../svm_hmm_windows/svm_hmm_learn',
            '-c', str(C),
            struct_train_path,
            struct_model_path]

    result = run(args, stdin=PIPE)

def evaluate_struct_perf(test_data_path, test_pred_path):
    #Prepping data to evaluate word and letter accuracy for SVM-Struct
    y = []
    word_ids = []
    with open(test_data_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.split(" ")
        y.append(int(parts[0]))
        word_ids.append(int(parts[1].split(":")[1]))
        
    
    with open(test_pred_path, 'r') as f:
        y_pred = list(map(int, f.readlines()))
    
    return evaluate_linear_svm(y, y_pred, word_ids)


def evaluate_svm_struct():
    # Classification method
    args = ['../svm_hmm_windows/svm_hmm_classify',
            struct_test_path,
            struct_model_path,
            struct_test_predictions_path]

    result = run(args, stdin=PIPE)

    char_acc, word_acc = evaluate_struct_perf(struct_test_path, struct_test_predictions_path)

    letter_accuracies.append(char_acc)
    word_accuracies.append(word_acc)
        

def evaluate_linear_svm(y, y_pred, word_ids):
    # Calculating letter and word accuracy
    correct = 0
    word_correct = [-1]*max(word_ids)
    
    for i, actual in enumerate(y):
        if(actual == y_pred[i]):
            correct += 1
            word_correct[word_ids[i] - 1] = 1 if word_correct[word_ids[i] - 1] != 0 else 0
        else:
            word_correct[word_ids[i] - 1] = 0
    
    char_acc = round(float(correct/len(y)) * 100, 3)
    word_acc = round(float(sum(word_correct)/len(word_correct)) * 100, 3)
    
    return char_acc, word_acc
    

def train_evaluate_linear_svm(C=1.0, transform=False, limit=None):
    # Prepping training data for model training
    X = []
    y = []
    word_ids = []
    with open(linear_train_path, 'r') as f:
        lines = f.readlines()
    for l in lines:
        letter = re.findall(r'[a-z]', l)
        letter = letter[0]
        l = re.findall(r'\d+', l)
        l = list(map(int, l))
        word_ids.append(l[2])
        X.append(l[4:])
        y.append(letter)
        
    #Reusing this method for Q4..
    if(transform):
        X = transform_data(X, word_ids, limit)
    
    model = LinearSVC(C=C, max_iter=1000, verbose=10, random_state=0)
    model.fit(X, y)
    
    #Prepping data to test the model
    X = []
    y = []
    word_ids = []
    with open(linear_test_path, 'r') as f:
        lines = f.readlines()
    for l in lines:
        letter = re.findall(r'[a-z]', l)
        letter = letter[0]
        l = re.findall(r'\d+', l)
        l = list(map(int, l))
        word_ids.append(l[2])
        X.append(l[4:])
        y.append(letter)

    y_predicted = model.predict(X)
    
    char_acc, word_acc = evaluate_linear_svm(y, y_predicted, word_ids)
    
    letter_accuracies.append(char_acc)
    word_accuracies.append(word_acc)

def accuracy_plots(X_range, scale='log', lbl=''):
    #Plotting graphs for visualization

    plt.plot(X_range, letter_accuracies, label='char-level acc')
    plt.title(lbl + ' - Character level accuracy')
    plt.legend()
    plt.xlabel('C/limit(q4)')
    if scale is not None: plt.xscale(scale)
    plt.xticks(X_range)
    plt.ylabel('accuracy')
    plt.show()

    plt.plot(X_range, word_accuracies, label='word-level acc')
    plt.title(lbl + ' - Word level accuracy')
    plt.legend()
    plt.xlabel('C/limit(q4)')
    if scale is not None: plt.xscale(scale)
    plt.xticks(X_range)
    plt.ylabel('accuracy')
    plt.show()   
    
    
def transform_data(X, word_ids, limit):

    # Method to rotate and translate the images
    
    w_np = np.array(word_ids)
    transforms = []
    with open("../data/transform.txt", 'r') as f:
        lines = f.readlines()
    for i in range(limit):
        line = lines[i].split(" ")
        if(line[0] == "r"):
            #Rotate image

            indices = np.where(w_np == int(line[1]))[0]
            
            for index in indices:
                x_np_arr = np.array(X[index])
                x_np_arr = x_np_arr.reshape(16, 8)
                y_np_arr = rotate(x_np_arr, int(line[2]), reshape=False)
                
                xsize = x_np_arr.shape
                ysize = y_np_arr.shape
                fromx = int((ysize[0] + 1 - xsize[0]) // 2) #// - floored division
                fromy = int((ysize[1] + 1 - xsize[1]) // 2)
                y_np_arr = y_np_arr[fromx:fromx + xsize[0], fromy:fromy + xsize[1]]                
                
                ind_0 = np.where(y_np_arr == 0)
                y_np_arr[ind_0] = x_np_arr[ind_0]
                
                X[index] = y_np_arr.reshape(1, 128).tolist()[0]
        elif(line[0] == "t"):
            #Translate image

            indices = np.where(w_np == int(line[1]))[0]
            
            for index in indices:
                x_np_arr = np.array(X[index])
                x_np_arr = x_np_arr.reshape(16, 8)
                xsize = x_np_arr.shape
                
                ox = int(line[2])
                oy = int(line[3])
                y = np.copy(x_np_arr)
                
                y[max(0, ox): min(xsize[0], xsize[0] + ox),
                    max(0, oy): min(xsize[1], xsize[1] + oy)] = x_np_arr[max(0, 1 - ox): min(xsize[0], xsize[0] - ox),
                     max(0, 1 - oy): min(xsize[1], xsize[1] - oy)]

                y[ox: xsize[0], oy: xsize[1]] = x_np_arr[0: xsize[0] - ox, 0: xsize[1] - oy]
                
                X[index] = y.reshape(1, 128).tolist()[0]            
    
    return X


if __name__ == "__main__":

    """ Question - 3"""
    """ Linear SVC """
    #Cs are inverted for this lib's SVM
    Cs = [1e-3, 1e-2, 1e-1, 1.0]  # [1.0, 10.0, 100.0, 1000.0]
        
    letter_accuracies.clear()
    word_accuracies.clear()
    
    for C in Cs:
        train_evaluate_linear_svm(C=C)
    
    accuracy_plots(Cs, lbl='3) SVM-MC')    
    
    """ Structured SVM """
    Cs = [1.0, 10.0, 100.0, 1000.0]

    letter_accuracies.clear()
    word_accuracies.clear()
    
    for C in Cs:
        train_svm_struct(C=C)
        evaluate_svm_struct()
    
    accuracy_plots(Cs, lbl='3) SVM-Struct')

    """ CRF """

    Cs = [1.0, 10.0, 100.0, 1000.0]

    letter_accuracies.clear()
    word_accuracies.clear()

    for C in Cs:
        if(C != 1000.0):
            char_acc, word_acc = crf_predict('result\solution-' + str(C) + '.txt')
        else:
            char_acc, word_acc = crf_predict('result\solution.txt') #reusing already computed params
        letter_accuracies.append(char_acc)
        word_accuracies.append(word_acc)

    accuracy_plots(Cs, lbl='3) CRF')
    

    """ Question 4 """
    """ Linear SVC """
    limits = [0, 500, 1000, 1500, 2000]  # [1.0, 10.0, 100.0, 1000.0]

    # Chosen C = 1.0 == 1000.0

    letter_accuracies.clear()
    word_accuracies.clear()

    for limit in limits:
        train_evaluate_linear_svm(C=1.0, transform=True, limit=limit)

    accuracy_plots(limits, scale=None, lbl='4) SVM-MC')

    
    """ CRF """
    # Chosen C = 1000.0    
    
    letter_accuracies.clear()
    word_accuracies.clear()

    for limit in limits:
        if(limit == 0):
            char_acc, word_acc = crf_predict('result\solution.txt') #reusing already computed params
        else:
            char_acc, word_acc = crf_predict('result\solution-4-' + str(limit) + '.txt')
        letter_accuracies.append(char_acc)
        word_accuracies.append(word_acc)

    accuracy_plots(limits, scale=None, lbl='4) CRF')