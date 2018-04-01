#cd "D:/MS Courses/AML/Project/Project_1(1)/"

# Loading Model W and T matrices 
import numpy
import itertools
import copy
import string
import re
import collections
from scipy.optimize import fmin_bfgs
import time
from scipy.ndimage import rotate
from crf_prediction import crf_predict

C = 1000
# model = numpy.loadtxt(r'data\model.txt')
# warray_m = numpy.array(model[0:3328])
# wmatrix_m = numpy.array(numpy.split(warray_m, 26))
# tarray_m = numpy.array(model[3328:4004])
# tmatrix_m = numpy.array(numpy.split(tarray_m, 26))
# tmatrix_m1 = numpy.matrix(numpy.split(tarray_m, 26))
# tmatrix_m = numpy.transpose(tmatrix_m1)
# final_t_matrix = numpy.array(tmatrix_m)

# Loading the training dataset
# x-train is 128 * total no of letters matrix 
# y-train is 1 * total no of letters matrix with one char
train_data = {}
x_train = []
y_train = []
next_id = []
word_id = []
pos = []
with open(r'..\data\train.txt', 'r') as f:
    lines = f.readlines()
for l in lines:
    letter = re.findall(r'[a-z]', l)
    letter = letter[0]
    l = re.findall(r'\d+', l)
    l = list(map(int, l))
    next_id.append(l[1])
    word_id.append(l[2])
    pos.append(l[3])
    x_train.append(l[4:])
    y_train.append(letter)
word_count = word_id[len(word_id)-1]
x_train_array = numpy.array(x_train)
numpy.shape(x_train_array)


# In[24]:

def objective(model):
    # Pre-calculating Dot Product of X and Node Weights W for all letters
    warray_m = numpy.array(model[0:3328])
    wmatrix_m = numpy.array(numpy.split(warray_m, 26))
    tarray_m = numpy.array(model[3328:4004])
    tmatrix_m = numpy.array(numpy.split(tarray_m, 26))
    tmatrix_m1 = numpy.matrix(numpy.split(tarray_m, 26))
    tmatrix_m = numpy.transpose(tmatrix_m1)
    final_t_matrix = numpy.array(tmatrix_m)
    total_letters = len(x_train)
    #total_letters = 68
    dot_letter_weights = numpy.inner(x_train_array, wmatrix_m)
    i = 0
    unique, counts = numpy.unique(word_id, return_counts=True)
    # Dictionary with list of words and count of letters in each
    word_dict = dict(zip(unique, counts))
    alphabet = dict(zip(string.ascii_lowercase, range(0,26)))
    # Calculating Forward Propagation for all the letters in each word
    Fwd = numpy.zeros((total_letters, 26))
    letter = 0
    for key in word_dict:
        m = word_dict[key]
        letter+=1
        for i in range(1,m):
            fwd_vect = Fwd[letter-1] + final_t_matrix.transpose()
            fwd_vect_max = numpy.max(fwd_vect, axis = 1)
            fwd_vect = (fwd_vect.transpose() - fwd_vect_max).transpose()
            Fwd[letter] = fwd_vect_max + numpy.log(numpy.sum(numpy.exp(fwd_vect + dot_letter_weights[letter-1]), axis = 1))
            letter+=1
    # Calculating Backward Propagation for all the letters in each word
    total_letters = len(x_train)
    Bwd = numpy.zeros((total_letters, 26))
    letter = -1
    letter1 = -1
    for key in word_dict:
        m = word_dict[key]
        letter1+=m
        letter = letter1
        letter-=1
        for i in range(1,m):
            bwd_vect = Bwd[letter+1] + final_t_matrix
            bwd_vect_max = numpy.max(bwd_vect, axis = 1)
            bwd_vect = (bwd_vect.transpose() - bwd_vect_max).transpose()
            Bwd[letter] = bwd_vect_max + numpy.log(numpy.sum(numpy.exp(bwd_vect + dot_letter_weights[letter+1]), axis = 1))
            letter-=1
    # Calculating Z value for each word based on forward passes
    word = 0
    letter=-1
    z = [0.0 for key in word_dict]
    for key in word_dict:
        m = word_dict[key]
        letter +=m
        z[word] = numpy.sum(numpy.exp(Fwd[letter] + dot_letter_weights[letter]))
        word+=1
    # Calcualting log_p(y | X) for each word
    log_py_x = [0.0 for key in word_dict]
    letter = 0
    word = 0
    for key in word_dict:
        sum = 0
        m = word_dict[key]
        for first in range(m):
            yb = alphabet[y_train[letter]]
            yprevb = alphabet[y_train[letter-1]]
            sum += dot_letter_weights[letter][yb]
            if first > 0:
                sum+= final_t_matrix[yprevb][yb]
            letter+=1
        log_py_x[word] = numpy.log(numpy.exp(sum)/z[word])
        word+=1
    # Average Log p(y | X) across all words
    avg_log_py_x = numpy.sum(log_py_x)/len(word_dict)
    reg = 1/2 * numpy.sum(model ** 2)
    return -C * avg_log_py_x + reg


# In[25]:

def gradient(model):
    # Pre-calculating Dot Product of X and Node Weights W for all letters
    warray_m = numpy.array(model[0:3328])
    wmatrix_m = numpy.array(numpy.split(warray_m, 26))
    tarray_m = numpy.array(model[3328:4004])
    tmatrix_m = numpy.array(numpy.split(tarray_m, 26))
    tmatrix_m1 = numpy.matrix(numpy.split(tarray_m, 26))
    tmatrix_m = numpy.transpose(tmatrix_m1)
    final_t_matrix = numpy.array(tmatrix_m)
    total_letters = len(x_train)
    #total_letters = 68
    dot_letter_weights = numpy.inner(x_train_array, wmatrix_m)
    i = 0
    unique, counts = numpy.unique(word_id, return_counts=True)
    # Dictionary with list of words and count of letters in each
    word_dict = dict(zip(unique, counts))
    alphabet = dict(zip(string.ascii_lowercase, range(0,26)))
    # Calculating Forward Propagation for all the letters in each word
    Fwd = numpy.zeros((total_letters, 26))
    letter = 0
    for key in word_dict:
        m = word_dict[key]
        letter+=1
        for i in range(1,m):
            fwd_vect = Fwd[letter-1] + final_t_matrix.transpose()
            fwd_vect_max = numpy.max(fwd_vect, axis = 1)
            fwd_vect = (fwd_vect.transpose() - fwd_vect_max).transpose()
            Fwd[letter] = fwd_vect_max + numpy.log(numpy.sum(numpy.exp(fwd_vect + dot_letter_weights[letter-1]), axis = 1))
            letter+=1
    # Calculating Backward Propagation for all the letters in each word
    total_letters = len(x_train)
    Bwd = numpy.zeros((total_letters, 26))
    letter = -1
    letter1 = -1
    for key in word_dict:
        m = word_dict[key]
        letter1+=m
        letter = letter1
        letter-=1
        for i in range(1,m):
            bwd_vect = Bwd[letter+1] + final_t_matrix
            bwd_vect_max = numpy.max(bwd_vect, axis = 1)
            bwd_vect = (bwd_vect.transpose() - bwd_vect_max).transpose()
            Bwd[letter] = bwd_vect_max + numpy.log(numpy.sum(numpy.exp(bwd_vect + dot_letter_weights[letter+1]), axis = 1))
            letter-=1
    # Calculating Z value for each word based on forward passes
    word = 0
    letter=-1
    z = [0.0 for key in word_dict]
    for key in word_dict:
        m = word_dict[key]
        letter +=m
        z[word] = numpy.sum(numpy.exp(Fwd[letter] + dot_letter_weights[letter]))
        word+=1
    # Calculating Gradient with respect to W and T
    letter = 0
    lettert = 0
    lettert1=0
    word = 0
    overallgrad = []
    total = numpy.zeros(128 * 26)
    for key in word_dict:
        m = word_dict[key]
        grad_w = numpy.zeros((26,128))
        for i in range(m):
            yb = alphabet[y_train[letter]]
            yprevb = alphabet[y_train[letter-1]]
            grad_w[yb] += x_train_array[letter]
            check = numpy.ones((26,128)) * x_train_array[letter]
            check = check.transpose() * numpy.exp(Fwd[letter] + Bwd[letter] + dot_letter_weights[letter]) / z[word]
            grad_w -= check.transpose()
            letter+=1
        grad_t = numpy.zeros(26*26)
        for i in range(m-1):
            for k in range(26):
                grad_t[k * 26: (k+1) * 26] -= numpy.exp(dot_letter_weights[lettert] + final_t_matrix.transpose()[k]
                                              + dot_letter_weights[lettert+1][k] + Bwd[lettert+1][k] + Fwd[lettert])
            lettert+=1
        lettert+=1
        grad_t /= z[word]
        for i in range(m-1):
            rnd = alphabet[y_train[lettert1]]
            rnd+= 26 * alphabet[y_train[lettert1 + 1]] 
            grad_t[rnd] += 1
            lettert1+=1
        lettert1+=1
        word+=1
        overallgrad.append(numpy.concatenate((grad_w.flatten(), grad_t)))
    # Average Gradient across words
    avg_gradient = (numpy.sum(overallgrad, axis = 0)/(len(word_dict))).tolist()
    return numpy.array(numpy.dot(avg_gradient, -C).tolist() + model)


def optimize_and_store(filename):

    print("===> Printing C, ", C)

    model = numpy.array([0.0 for x in range(4004)])
    #start = time.time()
    xopt = fmin_bfgs(objective, model, gradient, maxiter=100)
    #print(time.time() - start)
    thefile = open(filename, 'w') #'data\predictionWeights.txt'
    for i in range(len(xopt)):
        thefile.write("%s\n" % xopt[i])
    thefile.close()


def transform_data(X, word_ids, limit):
    
    w_np = numpy.array(word_ids)
    transforms = []
    with open("../data/transform.txt", 'r') as f:
        lines = f.readlines()
    for i in range(limit):
        line = lines[i].split(" ")
        if(line[0] == "r"):
            indices = numpy.where(w_np == int(line[1]))[0]
            
            for index in indices:
                x_np_arr = numpy.array(X[index])
                x_np_arr = x_np_arr.reshape(16, 8)
                y_np_arr = rotate(x_np_arr, int(line[2]), reshape=False)
                
                xsize = x_np_arr.shape
                ysize = y_np_arr.shape
                fromx = int((ysize[0] + 1 - xsize[0]) // 2) #// - floored division
                fromy = int((ysize[1] + 1 - xsize[1]) // 2)
                y_np_arr = y_np_arr[fromx:fromx + xsize[0], fromy:fromy + xsize[1]]                
                
                ind_0 = numpy.where(y_np_arr == 0)
                y_np_arr[ind_0] = x_np_arr[ind_0]
                
                X[index] = y_np_arr.reshape(1, 128).tolist()[0]
        elif(line[0] == "t"):
            indices = numpy.where(w_np == int(line[1]))[0]
            
            for index in indices:
                x_np_arr = numpy.array(X[index])
                x_np_arr = x_np_arr.reshape(16, 8)
                xsize = x_np_arr.shape
                
                ox = int(line[2])
                oy = int(line[3])
                y = numpy.copy(x_np_arr)
                
                y[max(0, ox): min(xsize[0], xsize[0] + ox),
                    max(0, oy): min(xsize[1], xsize[1] + oy)] = x_np_arr[max(0, 1 - ox): min(xsize[0], xsize[0] - ox),
                     max(0, 1 - oy): min(xsize[1], xsize[1] - oy)]

                y[ox: xsize[0], oy: xsize[1]] = x_np_arr[0: xsize[0] - ox, 0: xsize[1] - oy]
                
                X[index] = y.reshape(1, 128).tolist()[0]            
    
    return X




# from scipy.optimize import check_grad
# wt = numpy.loadtxt(r'C:\Users\Ashwin PC\Desktop\MS Materials\Advanced Machine Learning\Project 1\data\model.txt')
# print(check_grad(
#             objective, 
#             gradient, 
#             wt))

if __name__ == "__main__":
    """Question 2b"""
    optimize_and_store('result\solution.txt')
    #print("Question 2b, Optimal Objective Funtion value = ", fopt)

    """ Prediction for 2b """
    print(crf_predict('result\solution.txt', 'result\prediction.txt'))


    """Question 3a"""
    Cs = [1.0, 10.0, 100.0] #skipping 1000 since prev. run., use result\solution.txt for params during prediction
    for val in Cs:
        C = val
        optimize_and_store('result\solution-' + str(C) + '.txt')

    """Question 4a"""
    C = 1000.0
    limits = [500, 1000, 1500, 2000] #skipping 0, use result\solution.txt for params during prediction
    for limit in limits:
        next_id.clear()
        word_id.clear()
        pos.clear()
        x_train.clear()
        y_train.clear()

        with open(r'..\data\train.txt', 'r') as f:
            lines = f.readlines()
        for l in lines:
            letter = re.findall(r'[a-z]', l)
            letter = letter[0]
            l = re.findall(r'\d+', l)
            l = list(map(int, l))
            next_id.append(l[1])
            word_id.append(l[2])
            pos.append(l[3])
            x_train.append(l[4:])
            y_train.append(letter)
        word_count = word_id[len(word_id)-1]
        
        x_train = transform_data(x_train, word_id, limit)

        x_train_array = numpy.array(x_train)

        optimize_and_store('result\solution-4-' + str(limit) + '.txt')


"""
Run statistics for 2b and 3a
D:\MS Courses\AML\Project\Project_1(1)\solution>python crf.py
===> Printing C,  1000
Warning: Maximum number of iterations has been exceeded.
         Current function value: 3742.088129
         Iterations: 100
         Function evaluations: 102
         Gradient evaluations: 102
===> Printing C,  1.0
Optimization terminated successfully.
         Current function value: 20.039494
         Iterations: 13
         Function evaluations: 26
         Gradient evaluations: 26
===> Printing C,  10.0
Optimization terminated successfully.
         Current function value: 127.393909
         Iterations: 45
         Function evaluations: 60
         Gradient evaluations: 60
===> Printing C,  100.0
Warning: Maximum number of iterations has been exceeded.
         Current function value: 670.476276
         Iterations: 100
         Function evaluations: 103
         Gradient evaluations: 103


Prediction stats for 2b
(83.686, 47.281)


Run Statistics for 4a
===> Printing C,  1000.0
Warning: Maximum number of iterations has been exceeded.
         Current function value: 4553.929274
         Iterations: 100
         Function evaluations: 102
         Gradient evaluations: 102
===> Printing C,  1000.0
Warning: Maximum number of iterations has been exceeded.
         Current function value: 4962.395315
         Iterations: 100
         Function evaluations: 104
         Gradient evaluations: 104
===> Printing C,  1000.0
Warning: Maximum number of iterations has been exceeded.
         Current function value: 5282.610646
         Iterations: 100
         Function evaluations: 102
         Gradient evaluations: 102
===> Printing C,  1000.0
Warning: Maximum number of iterations has been exceeded.
         Current function value: 5505.896193
         Iterations: 100
         Function evaluations: 102
         Gradient evaluations: 102

"""