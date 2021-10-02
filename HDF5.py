# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 15:49:55 2021

@author: hp
"""
import h5py as hdf
from ANN import network
import numpy

def CreateNewFile (project_dir, file_name, net = None):
    file = hdf.File(project_dir + file_name, 'x')
    grp = file.create_group('1') 
    per = grp.create_dataset('Performance', (1,), dtype= 'float')
    lr = grp.create_dataset('LearnRate', (1,), dtype= 'float')
    Epoch = grp.create_dataset('Epoch', (1,), dtype= 'int')
    Ninput = grp.create_dataset('InputNodes', (1,), dtype= 'int')
    Nhidden = grp.create_dataset('HiddenNodes', (1,), dtype= 'int')
    Noutput = grp.create_dataset('OutputNodes', (1,), dtype= 'int')
    if (net != None):
        per[:] = net.Performance
        lr[:] = net.LearnRate
        Epoch[:] = net.Epoch
        Ninput[:] = net.InputNodes
        Nhidden[:] = net.HiddenNodes
        Noutput[:] = net.OutputNodes
        grp.create_dataset('wihi', data = net.wihi)
        grp.create_dataset('whoi', data = net.whoi)
        grp.create_dataset('wih', data = net.wih)
        grp.create_dataset('who', data = net.who)
    file.close()
   
def CreateNF (project_dir, file_name):
    file = hdf.File(project_dir + file_name, 'x')
    return (file)   

def OpenF (project_dir, file_name):
    file = hdf.File(project_dir + file_name, 'r+')
    return (file)

def LoadNetwork (project_dir, file_name, grp):
    """
    Function to fetch a particular network from the file directory set.
    """
    file = hdf.File(project_dir + file_name, 'r')
    OutputNodes = file[grp]['OutputNodes'][0]
    HiddenNodes = file[grp]['HiddenNodes'][0]
    InputNodes = file[grp]['InputNodes'][0]
    LearnRate = file[grp]['LearnRate'][0]
    net = network(InputNodes, HiddenNodes, OutputNodes, LearnRate)
    net.Epoch = file[grp]['Epoch'][0]
    net.Performance = file[grp]['Performance'][0]
    net.wih = file[grp]['wih'][:]
    net.who = file[grp]['who'][:]
    net.wihi = file[grp]['wihi'][:]
    net.whoi = file[grp]['whoi'][:]
    file.close()
    return(net)

def LoadN (file, grp):
    OutputNodes = file[grp]['OutputNodes'][0]
    HiddenNodes = file[grp]['HiddenNodes'][0]
    InputNodes = file[grp]['InputNodes'][0]
    LearnRate = file[grp]['LearnRate'][0]
    net = network(InputNodes, HiddenNodes, OutputNodes, LearnRate)
    net.Epoch = file[grp]['Epoch'][0]
    net.Performance = file[grp]['Performance'][0]
    net.wih = file[grp]['wih'][:]
    net.who = file[grp]['who'][:]
    net.wihi = file[grp]['wihi'][:]
    net.whoi = file[grp]['whoi'][:]
    return(net)

def ReturnN_Networks(project_dir, file_name):
    """
    Function to return number of networks in a given file directory set.
    """
    file = hdf.File(project_dir + file_name, 'r')
    Ngrps = 0
    for grp in file:
        Ngrps += 1
    return (Ngrps)
    file.close()

def ReturnN_Net(file):
    Ngrps = 0
    for grp in file:
        Ngrps += 1
    return (Ngrps)

def SaveNetwork(project_dir, file_name, net):
    """
    Function to save network "net" in a given file directory set.
    """
    i = ReturnN_Networks(project_dir, file_name)
    file = hdf.File(project_dir + file_name, 'r+')
    Newgrp = file.create_group(str(i+1))
    Newgrp.create_dataset('Performance', data = numpy.array(net.Performance, ndmin = 1))
    Newgrp.create_dataset('LearnRate', data = numpy.array(net.LearnRate, ndmin = 1))
    Newgrp.create_dataset('Epoch', data = numpy.array(net.Epoch, ndmin = 1))
    Newgrp.create_dataset('InputNodes', data = numpy.array(net.InputNodes, ndmin = 1))
    Newgrp.create_dataset('HiddenNodes', data = numpy.array(net.HiddenNodes, ndmin = 1))
    Newgrp.create_dataset('OutputNodes', data = numpy.array(net.OutputNodes, ndmin = 1))
    Newgrp.create_dataset('wihi', data = numpy.array(net.wihi, ndmin = 1))
    Newgrp.create_dataset('whoi', data = numpy.array(net.whoi, ndmin = 1))
    Newgrp.create_dataset('wih', data = numpy.array(net.wih, ndmin = 1))
    Newgrp.create_dataset('who', data = numpy.array(net.who, ndmin = 1))
    file.close()
    
def SaveN(file, net, grp = None):
    if (grp == None):
        i = ReturnN_Net(file)
        Newgrp = file.create_group(str(i+1))
    else:
        Newgrp = file.create_group(grp)
    Newgrp.create_dataset('Performance', data = numpy.array(net.Performance, ndmin = 1))
    Newgrp.create_dataset('LearnRate', data = numpy.array(net.LearnRate, ndmin = 1))
    Newgrp.create_dataset('Epoch', data = numpy.array(net.Epoch, ndmin = 1))
    Newgrp.create_dataset('InputNodes', data = numpy.array(net.InputNodes, ndmin = 1))
    Newgrp.create_dataset('HiddenNodes', data = numpy.array(net.HiddenNodes, ndmin = 1))
    Newgrp.create_dataset('OutputNodes', data = numpy.array(net.OutputNodes, ndmin = 1))
    Newgrp.create_dataset('wihi', data = numpy.array(net.wihi, ndmin = 1))
    Newgrp.create_dataset('whoi', data = numpy.array(net.whoi, ndmin = 1))
    Newgrp.create_dataset('wih', data = numpy.array(net.wih, ndmin = 1))
    Newgrp.create_dataset('who', data = numpy.array(net.who, ndmin = 1))   

def DeleteNetwork (project_dir, file_name, grp):
    """
    Function to delete a network from a given file directory set.
    """
    i = ReturnN_Networks(project_dir, file_name)
    if (int(grp) < i+1):
        file = hdf.File(project_dir + file_name, 'r+')
        del file[grp]
        file.close()

def DeleteN (file, grp):
    i = ReturnN_Net(file)
    if (int(grp) < i+1):
        del file[grp]

def DeleteAllN (file):
    for i in file:
        del file[i]

def CloseF (file):
    file.close()
    
