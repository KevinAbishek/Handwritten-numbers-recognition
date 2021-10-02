# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 16:32:54 2021

@author: hp
"""
import numpy
import scipy.special
from matplotlib import pyplot as pp
import cv2

class network:
    
    def __init__(self, ni, nh, no, lr = 1):
        self.InputNodes = ni
        self.HiddenNodes = nh
        self.OutputNodes = no
        self.LearnRate = lr
        self.Epoch = 0
        self.Performance = 0.0
        
        self.wih = numpy.random.normal(0.0, pow(self.HiddenNodes, -0.5), (self.HiddenNodes, self.InputNodes))
        self.who = numpy.random.normal(0.0, pow(self.OutputNodes, -0.5), (self.OutputNodes, self.HiddenNodes))
        self.wihi = self.wih.copy()
        self.whoi = self.who.copy()
        
        self.act_func = lambda x: scipy.special.expit(x)
    
    def FP (self, input_vals):
        
        self.input_vals = numpy.array(input_vals, ndmin = 2).T
        
        self.HiddenNodeInputs = numpy.dot(self.wih, self.input_vals)
        self.HiddenNodeOutputs = self.act_func(self.HiddenNodeInputs)
        self.OutputNodeInputs = numpy.dot(self.who, self.HiddenNodeOutputs)
        self.OutputNodeOutputs = self.act_func(self.OutputNodeInputs)
        
        return (self.OutputNodeOutputs)
    
    def Train (self, input_vals, training_vals):
        
        self.training_vals = numpy.array(training_vals, ndmin = 2).T
        
        self.OutputNodeOutputs = self.FP(input_vals)
        
        self.OutputErrors = self.training_vals - self.OutputNodeOutputs
        self.HiddenErrors = numpy.dot(self.who.T, self.OutputErrors)
        
        self.who += self.LearnRate * numpy.dot (self.OutputErrors*self.OutputNodeOutputs*(1-self.OutputNodeOutputs), numpy.transpose(self.HiddenNodeOutputs))
        self.wih += self.LearnRate * numpy.dot (self.HiddenErrors*self.HiddenNodeOutputs*(1-self.HiddenNodeOutputs), numpy.transpose(self.input_vals))
        

def TrainNetwork(file_name, net, prog = True):
    file = open(file_name,'r')
    i, j = 0, 0
    filesize = file.seek(0,2)
    file.seek(0)
    while (i < filesize):
        data = file.readline()
        i = file.tell()
        data = data.split(',')
        #data_im = numpy.asfarray(data[1:]).reshape((28,28))
        training_input_vals = ((numpy.asfarray(data[1:]))/255*0.99) + 0.01
        training_output_vals = numpy.zeros((1,net.OutputNodes)) + 0.01
        training_output_vals[0][int(data[0])] = 0.99
        net.Train(training_input_vals,   training_output_vals)
        j += 1
        if (prog):
            print (str(net.Epoch+1) + '_' + str(j))
    net.Epoch += 1
    file.close()

def OpenFile (file_name):
    file = open (file_name,'r')
    return (file)

def CloseFile (file):
    file.close()

def GetNumber (file_name):
    file = open(file_name,'r')
    i, j = 0, 0
    filesize = file.seek(0,2)
    file.seek(0)
    while (i < filesize):
        file.readline()
        i = file.tell()
        j += 1
    file.close()
    return (j)

def GetNumberFast (file):
    i = 0
    readpositions = []
    filesize = file.seek(0,2)
    file.seek(0)
    while (i < filesize):
        readpositions.append(i)
        file.readline()
        i = file.tell()
    return (readpositions)

def GetData (instance, file_name, Datatype = 'FP', normalize = True):
    file_size = GetNumber(file_name)
    if ((instance < 0) and (instance > file_size)):
        print("Instance does no exist")
        return None
    else:
        file = open(file_name,'r')
        i = 0
        file.seek(0)
        while (i < instance):
            file.readline()
            i += 1
        data = file.readline()
        file.close()
        data = data.split(',')
        if (normalize == True):
            input_data = (numpy.asfarray(data[1:])/255*0.99)+0.01
        elif (normalize == False):
            input_data = numpy.asfarray(data[1:])
            image_data = input_data.reshape((28,28))
        if (Datatype == 'FP'):
            output_data = int(data[0])
            if (normalize == False):
                pp.imshow(image_data)
                print(output_data)
        elif (Datatype == 'Train'):
            output_data = numpy.zeros((1,10)) + 0.01
            output_data[0][int(data[0])] = 0.99
        return (input_data, output_data)
   
def GetDataFast (readpos, file):
    file.seek(readpos)
    data = file.readline()
    data = data.split(',')
    input_data = (numpy.asfarray(data[1:])/255*0.99)+0.01
    output_data = int(data[0])
    return (input_data, output_data)    

def GenerateCustomImageData(image_file_name):
    #write in red over blue BG
    img = cv2.imread(image_file_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    i, j = 0, 0
    maxv, minv = gray.max(), gray.min()
    for i in range(0,len(gray)):
        for j in range(0,len(gray[0])):
            gray[i][j] = 255.0/(maxv-minv)*(gray[i][j]-minv)
    return(gray)

def FeedCustomImageData(custom_image_dir, number, instance, net):
    path = custom_image_dir + "\\" + str(number) + "\\" + str(instance) + ".png"
    gray = GenerateCustomImageData(path)
    input_data = (numpy.asfarray(gray)/255*0.99)+0.01
    input_data = numpy.asfarray(input_data).reshape((1, net.InputNodes))
    pp.imshow(gray)
    return(net.FP(input_data))
    
#open and close file

def GetPerformance(file_name, net, prog = True):
    file = OpenFile(file_name)
    readpositions = GetNumberFast(file)
    results = []
    for instance in range (0, len(readpositions)):
        RA = []
        res = []
        input_vals, target_val = GetDataFast (readpositions[instance], file)
        output_vals = net.FP (input_vals)
        max_val = max (output_vals)
        for i in range (0,len(output_vals)):
            if (output_vals[i] == max_val):
                RA.append(1)
            else:
                RA.append(0)
        i = 0
        for i in range(0, len(RA)):
            if (RA[i] == 1):
                res.append(i)
        if ((len(res) == 1) and (res[0] == target_val)):
            results.append(1)
        else: 
            results.append(0)
        if (prog):
            print (instance+1)
    net.Performance = performance = sum(results)/len(results)
    CloseFile(file)
    return (results, performance)