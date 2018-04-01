
# coding: utf-8

# In[198]:

# Loading Model W and T matrices 
import numpy
import itertools
import copy
import string
model = numpy.loadtxt(r'C:\Users\Ashwin PC\Desktop\MS Materials\Advanced Machine Learning\Project 1\data\model.txt')
warray_m = numpy.array(model[0:3328])
wmatrix_m = numpy.array(numpy.split(warray_m, 26))
print(numpy.shape(wmatrix_m))
tarray_m = numpy.array(model[3328:4004])
tmatrix_m = numpy.array(numpy.split(tarray_m, 26))
tmatrix_m1 = numpy.matrix(numpy.split(tarray_m, 26))
tmatrix_m = numpy.transpose(tmatrix_m1)
final_t_matrix = numpy.array(tmatrix_m)
print(numpy.shape(tmatrix_m))
import re
import collections
train_data = {}
x_train = []
y_train = []
next_id = []
word_id = []
pos = []
with open(r'C:\Users\Ashwin PC\Desktop\MS Materials\Advanced Machine Learning\Project 1\data\train.txt', 'r') as f:
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

# Pre-calculating Dot Product of X and Node Weights W for all letters
total_letters = len(x_train)
dot_letter_weights = numpy.inner(x_train_array, wmatrix_m)
i = 0
unique, counts = numpy.unique(word_id, return_counts=True)

# Dictionary with list of words and count of letters in each
word_dict = dict(zip(unique, counts))
alphabet = dict(zip(string.ascii_lowercase, range(0,26)))


# In[199]:

# Calculating Forward Propagation for all the letters in each word
total_letters = len(x_train)
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


# In[200]:

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


# In[201]:

# Calculating Z value for each word based on forward passes
word = 0
letter=-1
z = [0.0 for key in word_dict]
for key in word_dict:
    m = word_dict[key]
    letter +=m
    z[word] = numpy.sum(numpy.exp(Fwd[letter] + dot_letter_weights[letter]))
    word+=1


# In[202]:

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


# In[204]:

# Average Log p(y | X) across all words
avg_log_py_x = numpy.sum(log_py_x)/len(word_dict)
avg_log_py_x


# In[205]:

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
                                              + dot_letter_weights[lettert+1][j] + Bwd[lettert+1][k] + Fwd[lettert])
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


# In[206]:

# Average Gradient across words
avg_gradient = (numpy.sum(overallgrad, axis = 0)/len(word_dict)).tolist()


# In[211]:

thefile = open('random12.txt', 'w')
for item in avg_gradient:
    thefile.write("%s\n" % item)


# In[188]:




# In[212]:



