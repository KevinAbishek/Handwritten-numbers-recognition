# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 12:55:38 2021

@author: hp
"""
from ANN import network, TrainNetwork, GetPerformance
from HDF5 import SaveN, CreateNF, CloseF, OpenF, LoadN, DeleteN, ReturnN_Net

test_file_name = "mnist_test.csv"
test10_file_name = "mnist_test_10.csv"
train_file_name = "mnist_train.csv"
custom_image_dir = r'C:\Users\hp\Desktop\Study\Python\practice\NN\Test Images'
project_dir = r"C:\Users\hp\Desktop\Study\Python\practice\NN"

memory_file_name = r'\NNmem.hdf5'
Epoch2_file_name = r'\NNmem_E2.hdf5'
Epoch3_file_name = r'\NNmem_E3.hdf5'
Epoch4_file_name = r'\NNmem_E4.hdf5'
Epoch5_file_name = r'\NNmem_E5.hdf5'
Epoch6_file_name = r'\NNmem_E6.hdf5'
Epoch7_file_name = r'\NNmem_E7.hdf5'
Epoch8_file_name = r'\NNmem_E8.hdf5'
Epoch9_file_name = r'\NNmem_E9.hdf5'
Epoch10_file_name = r'\NNmem_E10.hdf5'

Eit = [memory_file_name, Epoch2_file_name, Epoch3_file_name, Epoch4_file_name, Epoch5_file_name, Epoch6_file_name, Epoch7_file_name, Epoch8_file_name, Epoch9_file_name, Epoch10_file_name]

inodes = 28*28
onodes = 10
hnodes = 100
learnrate = 0.1


def CreateAndStoreRandomNetworks (Nitems):
    file = CreateNF (project_dir, memory_file_name)
    i = 0
    for i in range(0,Nitems):
        NN = network (inodes, hnodes, onodes, learnrate)
        print ('Training Model: ' + str(i+1))
        TrainNetwork (train_file_name,  NN, prog = False)
        print ('Performance Test Model: ' + str(i+1))
        GetPerformance (test_file_name, NN)
        print ('Storing Model: ' + str(i+1))
        SaveN (file, NN)
    CloseF (file)


def IncrementEpoch(mem_filename_1, mem_filename_2):
    file2 = CreateNF (project_dir, mem_filename_2)
    file1 = OpenF(project_dir, mem_filename_1)
    Nnet1 = ReturnN_Net(file1)
    for i in range(1,Nnet1+1):
        nn = LoadN(file1, str(i))
        print ('Epoch: ' + str(nn.Epoch+1) + ' ,Training Model: ' + str(i))
        TrainNetwork(train_file_name, nn, prog = False)
        print ('Performance Test Model: ' + str(i))
        GetPerformance (test_file_name, nn, prog = False)
        print ('Storing Model: ' + str(i))
        SaveN (file2, nn)
    CloseF (file1)
    CloseF (file2)
        
def ProduceEpochIterations (nit):
    for i in range(0, nit-1):
        IncrementEpoch(Eit[i], Eit[i+1])
        

def EpochPlots (files, grp):
    Pvals = []
    for i in files:
        f = OpenF (project_dir, i)
        nn = LoadN (f, grp)
        Pvals.append (nn.Performance)
        CloseF (f)
    return (Pvals)

            