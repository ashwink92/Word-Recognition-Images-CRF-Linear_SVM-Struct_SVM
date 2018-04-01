
# coding: utf-8

# In[2]:

# load Data
import numpy
import itertools
import copy
a = numpy.loadtxt(r'decode_input.txt')
i =0
j =127
y = list(range(1,27))
xarray = numpy.array(a[0:12800])
xmatrix = numpy.matrix(numpy.split(xarray, 100))
print(numpy.shape(xmatrix))
warray = numpy.array(a[12800:16128])
wmatrix = numpy.matrix(numpy.split(warray, 26))
print(numpy.shape(wmatrix))
tarray = numpy.array(a[16128:16804])
tmatrix1 = numpy.matrix(numpy.split(tarray, 26))
tmatrix = numpy.transpose(tmatrix1)
print(numpy.shape(tmatrix))


# In[3]:

# Initialization
nodeweight = []
nodeweight1 = []
opt = []
for j in range(0,26):
    nodeweight.append(numpy.dot(xmatrix[0], wmatrix[j].transpose()).tolist())
opt.append(nodeweight)


# In[4]:

# Dynamic Programming
optnodes = [[0 for x in range(26)] for y in range(100)] 
for i in range(1,100):
    tempnodeweight = []
    tempedgeweight = []
    nodeweight1 = []
    nodeweight1 = copy.deepcopy(nodeweight)
    opt.append(nodeweight1)
    for j in range(0,26):
        tempnodeweight.append(numpy.dot(xmatrix[i], wmatrix[j].transpose()).tolist())
    for j in range(0,26):
        opt[i][j][0][0] = 0
        for k in range(0,26):
            if(opt[i][j][0][0] < opt[i-1][k][0][0] + tempnodeweight[j][0][0] + tmatrix.item(k,j)):
                opt[i][j][0][0] = opt[i-1][k][0][0] + tempnodeweight[j][0][0] + tmatrix.item(k,j)
                optnodes[i][j] = k


# In[5]:

# final values
seq = [0 for x in range(0,100)]
argmax = 0
for i in range(0,26):
    if(argmax < opt[99][i][0][0]):
        argmax = opt[99][i][0][0]
        seq[99] = i + 1;
print(seq[99])
argmax


# In[6]:

# Track Back
k = seq[99] - 1
for i in range(99, 0, -1):
    k = optnodes[i][k]
    seq[i-1] = k + 1
seq
argmax


# In[11]:

thefile = open('decode_output.txt', 'w')
for item in seq:
    thefile.write("%s\n" % item)


