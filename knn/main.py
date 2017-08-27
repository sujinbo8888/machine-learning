from numpy import *
import operator

import matplotlib
import matplotlib.pyplot as plt
from os import listdir


def createDataSet():
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels

def classify0(intx,dataset,labels,k):
    datasetsize=dataset.shape[0]
    diffmat=tile(intx,(datasetsize,1))-dataset
    sqdiffmat=diffmat**2
    sqdistances=sqdiffmat.sum(axis=1)
    distances=sqdistances**0.5
    sorteddistIndicies=distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel=labels[sorteddistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1

    sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename='datingTestSet2.txt'):
    fr=open(filename)
    arrayOlines=fr.readlines()
    numberOfLines=len(arrayOlines)
    returnMat=zeros((numberOfLines,3))
    labels=[]
    index=0
    for line in arrayOlines:
        line=line.strip()
        listFromLine=line.split('\t')
        returnMat[index,:]=listFromLine[0:3]
        labels.append(int(listFromLine[-1]))
        index+=1

    return returnMat,labels

def img2vector(filename):
    returnVect=zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        line=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(line[j])
    return returnVect

def handwritingClassTest():
    labels=[]
    files=listdir('digits/trainingDigits')
    m=len(files)
    traingMat=zeros((m,1024))
    for i in range(m):
        filenameStr=files[i]
        classNumber=filenameStr.split('.')[0].split('_')[0]
        labels.append(classNumber)
        traingMat[i,:]=img2vector('digits/trainingDigits/%s'%filenameStr)

    testfiles=listdir('digits/testDigits')
    m=len(files)
    errorCount=0.0
    for i in range(m):
        filenameStr=files[i]
        classNumber=filenameStr.split('.')[0].split('_')[0]

        testMat=img2vector('digits/trainingDigits/%s'%filenameStr)

        classfiyResult=classify0(testMat,traingMat,labels,10)

        print "the calssifier result:%s, the real result:%s"%(classfiyResult,classNumber)

        if classNumber!=classfiyResult:errorCount+=1.0

    print "the total number of error:%d"%errorCount
    print "the percent:%f"%(errorCount/float(m))





def autoNorm(data):
    minv=data.min(0)
    maxv=data.max(0)

    ranges=maxv-minv

    normal=zeros(shape(data))
    m=data.shape[0]
    normal=data-tile(minv,(m,1))
    normal=normal/tile(ranges,(m,1))

    return normal,ranges,minv

if __name__=='__main__':

    handwritingClassTest()
'''    data,label= createDataSet()
    print classify0([1.3,0.3],data,label,3)

    mat,label= file2matrix()
    mat,s,m= autoNorm(mat)

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(mat[:,0],mat[:,1],15.0*array(label),15.0*array(label))
    plt.show()
'''

