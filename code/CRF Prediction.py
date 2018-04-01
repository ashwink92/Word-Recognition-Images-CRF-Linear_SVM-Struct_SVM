
# coding: utf-8

# In[1]:

# Loading Model W and T matrices 
import numpy
import itertools
import copy
model = numpy.loadtxt(r'C:\Users\Ashwin PC\Desktop\MS Materials\Advanced Machine Learning\Project 1\Final\predictionWeights.txt')
warray_m = numpy.array(model[0:3328])
wmatrix_m = numpy.matrix(numpy.split(warray_m, 26))
print(numpy.shape(wmatrix_m))
tarray_m = numpy.array(model[3328:4004])
tmatrix_m = numpy.matrix(numpy.split(tarray_m, 26))
tmatrix_m1 = numpy.matrix(numpy.split(tarray_m, 26))
tmatrix_m = numpy.transpose(tmatrix_m1)
print(numpy.shape(tmatrix_m))
# Loading the training dataset
# x-train is 128 * total no of letters matrix 
# y-train is 1 * total no of letters matrix with one char
import re
import collections
train_data = {}
x_train = []
y_train = []
next_id = []
word_id = []
pos = []
with open(r'C:\Users\Ashwin PC\Desktop\MS Materials\Advanced Machine Learning\Project 1\data\test.txt', 'r') as f:
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
x_train_matrix = numpy.matrix(x_train)
numpy.shape(x_train_matrix)


# In[2]:

# Create dictionary with letters and increasing numbers
import string
alphabet = dict(zip(string.ascii_lowercase, range(0,26)))


# In[3]:

i = 0
unique, counts = numpy.unique(word_id, return_counts=True)
# Dictionary with list of words and count of letters in each
word_dict = dict(zip(unique, counts))
letter = 0
total_letters = len(x_train)
dict1 = {}
words = 0
letter = 0


# In[4]:

def predict(m,letter):
    # Initialization
    nodeweight = []
    nodeweight1 = []
    opt = []
    for j in range(0,26):
        nodeweight.append(numpy.dot(x_train_matrix[letter], wmatrix_m[j].transpose()).tolist())
    opt.append(nodeweight)
    letter+=1
    # Dynamic Programming
    optnodes = [[0 for x in range(26)] for y in range(m)] 
    for i in range(1,m):
        tempnodeweight = []
        tempedgeweight = []
        nodeweight1 = []
        nodeweight1 = copy.deepcopy(nodeweight)
        opt.append(nodeweight1)
        for j in range(0,26):
            tempnodeweight.append(numpy.dot(x_train_matrix[letter], wmatrix_m[j].transpose()).tolist())
        for j in range(0,26):
            opt[i][j][0][0] = -10000
            for k in range(0,26):
                if(opt[i][j][0][0] < opt[i-1][k][0][0] + tempnodeweight[j][0][0] + tmatrix_m.item(k,j)):
                    opt[i][j][0][0] = opt[i-1][k][0][0] + tempnodeweight[j][0][0] + tmatrix_m.item(k,j)
                    optnodes[i][j] = k
        letter +=1
    # final values
    seq = [0 for x in range(0,m)]
    argmax = -10000
    for i in range(0,26):
        if(argmax < opt[m-1][i][0][0]):
            argmax = opt[m-1][i][0][0]
            seq[m-1] = i + 1;
    print(seq[m-1])
    # Track Back
    k = seq[m-1] - 1
    for i in range(m-1, 0, -1):
        k = optnodes[i][k]
        seq[i-1] = k + 1
    return seq


# In[5]:

# Predicting each word
letter = -1
each_word = []
for key in word_dict:
    m = word_dict[key]
    each_word.append(predict(m,letter+1))
    letter+=m


# In[6]:

# predicting each word
i = 0
eachLetter = []
for key in word_dict:
    m = word_dict[key]
    for x in range(m):
        eachLetter.append(each_word[i][x])
    i+=1


# In[7]:




# In[13]:

# To check Accuracy
alphabetrev = dict(zip(range(1,27), string.ascii_lowercase))


# In[24]:

# Calculating Letter Wise Accuracy
i = 0
correctpred = 0
for item in eachLetter:
    if y_train[i] == alphabetrev[eachLetter[i]]:
        correctpred+=1
    i+=1
print(correctpred)
print(correctpred / len(eachLetter))


# In[29]:

# Calculating word wise accuracy
i = 0
letter = 0
correctpred = 0
wordcorrectpred = 0
for item in word_dict:
    m = word_dict[item]
    for j in range(m):
        if y_train[i] == alphabetrev[eachLetter[i]]:
            correctpred+=1
        i+=1
    if correctpred == m:
        wordcorrectpred+=1
    correctpred = 0
print(wordcorrectpred)
print(wordcorrectpred / len(word_dict))


# In[33]:

thefile = open('prediction.txt', 'w')
for item in eachLetter:
    thefile.write("%s\n" % item)

