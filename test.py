from numpy import *
import numpy

a = numpy.loadtxt('data3.txt')
train = a[:, 2:]
dataSet = array(train)

f = open('test.txt', 'w')
f.write(str(train, encoding='utf8'))
f.close()

