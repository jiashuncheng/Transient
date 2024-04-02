from asyncio import transports
from itertools import count
from turtle import color
import numpy as np
from parameters import *
import pickle
import matplotlib.pyplot as plt
import copy
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE, Isomap
from scipy.stats import entropy
import torch

def normalization(data):
    _range = np.max(data, axis=1) - np.min(data, axis=1)
    _range = _range[:,np.newaxis] + 0.00001
    return (data - np.min(data, axis=1)[:, np.newaxis]) / _range

def normalization1(data):
    data = data - np.mean(data, axis=1)[:,np.newaxis]
    data = data/(np.max(np.abs(data), axis=1)[:,np.newaxis]+0.00001)
    return data

def mean_var(A):
    mean = np.mean(A)
    var = np.var(A)
    std = np.std(A)
    return mean, var, std

def c_mass(data):
    interval = 1
    mass_max = 0
    mass_loc = 0
    for i in range(len(data)-interval):
        if np.sum(data[i:i+interval]) > mass_max:
            mass_max = np.sum(data[i:i+interval])
            mass_loc = i + interval/2
    return (mass_loc, mass_max)

def sorting(base):
    loc_all_0 = []
    for i in base[:, :]: #20:80
        loc, mass = c_mass(i)
        loc_all_0.append(loc) 
        # mass_all.append(mass)
    B = np.argsort(loc_all_0)  

    return B, loc_all_0

def norm_sorting(data):
    data = normalization1(data)
    B = sorting(data[:, :])
    return data[B, :]

def trajectory_weights(data, base, weights0, weights, epoch, acc, outfile, delay_time, save=True):
    a = sorting(base[:par['sensor_region'], 20:80])
    b = sorting(base[par['sensor_region']:par['sensor_region']+par['association_region'], 20:80]) + par['sensor_region']
    c = sorting(base[-par['motor_region']:, 20:80]) + par['sensor_region'] + par['association_region']
    C = np.concatenate((a,b,c))
    original = copy.deepcopy(data)
    original[0] = original[0][C, :]

    B = sorting(base[:, 20:80])
    for j in range(len(data)):
        data[j] = data[j][B, :]
    data0 = data[0]
    data1 = data[1]
    # STI for transient
    # entropy, r2b_ratio, SI_trial_vec, tol_sum, TI = STI(abs(data0[:,20:20+delay_time].T))
    # print('Entropy is {}, ratio is {}, SI is {}, transfer is {}'.format(entropy, r2b_ratio, SI_trial_vec, tol_sum))

    num = len(data0)
    
    weights_sort = weights[:, B]
    weights_sort = weights_sort[B, :]

    # weights in/between regions   par['sensor_region']:par['sensor_region']+par['association_region']
    s2s = weights[:par['sensor_region'], :par['sensor_region']]
    s2a = weights[:par['sensor_region'], par['sensor_region']:par['sensor_region']+par['association_region']]
    s2m = weights[:par['sensor_region'], -par['motor_region']:]

    a2s = weights[par['sensor_region']:par['sensor_region']+par['association_region'], :par['sensor_region']]
    a2a = weights[par['sensor_region']:par['sensor_region']+par['association_region'], par['sensor_region']:par['sensor_region']+par['association_region']]
    a2m = weights[par['sensor_region']:par['sensor_region']+par['association_region'], -par['motor_region']:]

    m2s = weights[-par['motor_region']:, :par['sensor_region']]
    m2a = weights[-par['motor_region']:, par['sensor_region']:par['sensor_region']+par['association_region']]
    m2m = weights[-par['motor_region']:, -par['motor_region']:]
    
    if weights.max()>weights0.max():
        max_legend = weights.max()
    else:
        max_legend = weights0.max()

    if weights.min()< weights0.min():
        min_legend = weights.min()
    else:
        min_legend = weights0.min()    

    list_weights = [s2s, s2a, s2m, a2s, a2a, a2m, m2s, m2a, m2m]
    weights_name = ['s2s', 's2a', 's2m', 'a2s', 'a2a', 'a2m', 'm2s', 'm2a', 'm2m']
    plt.figure(1, figsize=(10*3, 8*3))
    
    for j in range(len(list_weights)):
        loc =  330 + j + 1
        mean, var, std = mean_var(list_weights[j])
        if weights_name[j][0] == weights_name[j][2]:
            sparse = (len(np.where(list_weights[j]==0)[0])-list_weights[j].shape[0])/(list_weights[j].shape[0] * list_weights[j].shape[1]-list_weights[j].shape[0])
        else:
            sparse = len(np.where(list_weights[j]==0)[0])/(list_weights[j].shape[0] * list_weights[j].shape[1])
        plt.subplot(loc)
        fig=plt.imshow(list_weights[j], aspect='auto', vmax=max_legend, vmin=min_legend)# ,vmax=10#,'jet' vmin=0, vmax=1cmap=plt.cm.winter, , cmap='inferno' magma , cmap='jet', vmax=1, vmin=-1
        plt.xlabel('Neuron_index')
        plt.gca().invert_yaxis()
        plt.ylabel('Neuron_index')
        plt.title('Connectivity matrix:' + weights_name[j] + '\n' + 'mean:' + str(mean) + ' var:' + str(var) + ' std:' + str(std) + '\n' + 'sparse:' + str(sparse))
        if (j+1)%3==0:
            plt.colorbar(fig, label="Weights value")
    if save:
        plt.savefig('./result/' + outfile + '.pdf') #, dpi=300
        plt.close()
    else:
        plt.show() 

    plt.figure(2, figsize=(35, 8))
    plt.subplot(141)
    fig=plt.imshow(data0, aspect='auto',cmap='jet')# ,vmax=10#,'jet' vmin=0, vmax=1cmap=plt.cm.winter, , cmap='inferno' magma , cmap='jet', vmax=1, vmin=-1
    plt.xlabel('Steps/100ms')
    plt.gca().invert_yaxis()
    plt.ylabel('Sorting neuron_index')
    plt.title('Neural activity in trail ' + str(epoch) +'\n' +'ACC:' + str(acc))
    plt.colorbar(fig, label="Normalized firing rate")
    plt.plot([10, 10], [10, num-10], c='w', linestyle='--')
    plt.plot([20, 20], [10, num-10], c='w', linestyle='--')
    plt.plot([80, 80], [10, num-10], c='w', linestyle='--')

    plt.subplot(142)
    fig=plt.imshow(original[0], aspect='auto',cmap='jet')# ,vmax=10#,'jet' vmin=0, vmax=1cmap=plt.cm.winter, , cmap='inferno' magma , cmap='jet', vmax=1, vmin=-1
    plt.xlabel('Steps/100ms')
    plt.gca().invert_yaxis()
    plt.ylabel('Sorting neuron_index')
    plt.title('Neural activity in trail ' + str(epoch) +'\n' + 'region sorting')
    plt.colorbar(fig, label="Normalized firing rate")
    plt.plot([10, 10], [10, num-10], c='w', linestyle='--')
    plt.plot([20, 20], [10, num-10], c='w', linestyle='--')
    plt.plot([80, 80], [10, num-10], c='w', linestyle='--')

    # plt.subplot(142)
    # fig=plt.imshow(normalization1(weights_sort), aspect='auto',vmin=-1, vmax=1)    
    # plt.xlabel('Neuron_index')
    # plt.gca().invert_yaxis()
    # plt.ylabel('Neuron_index')
    # plt.colorbar(fig, label="Weights value")
    # plt.title('Connection matrix (sorting)')

    con_strenght = np.mean(normalization1(weights_sort), axis=0)
    con_strenght = con_strenght[:, np.newaxis]
    plt.subplot(143)
    fig=plt.imshow(con_strenght, aspect='auto')    
    plt.ylabel('Neuron_index')
    plt.colorbar(fig, label="Value")
    plt.gca().invert_yaxis()
    plt.xticks([])
    plt.title('Connection matrix (column mean-output connection)')

    con_strenght = np.mean(normalization1(weights_sort), axis=1)
    con_strenght = con_strenght[:, np.newaxis]
    plt.subplot(144)
    # fig=plt.imshow(con_strenght, aspect='auto')    
    for i in ([10,20]):
        plt.plot(normalization1(weights_sort)[i, :], label=i, linewidth=0.5)
    plt.legend()
    # plt.ylabel('Neuron_index')
    # plt.colorbar(fig, label="Value")
    # plt.gca().invert_yaxis()
    # plt.xticks([])
    plt.title('Output connection')

    if save:
        plt.savefig('./result/' + outfile + 'sorting' +  '.pdf') #, dpi=300
        plt.close()
    else:
        plt.show() 

def acc_act(acc, hidden_act, k, odors, outfile):
    activity = np.mean(np.mean(np.mean(np.array(hidden_act), axis=1),1),1) #(epochs, time, batch, neurons)
    # Firing_rate0 = np.mean(np.mean(np.array(hidden_act[k])[:, odors[0], :], axis=1), axis=1)
    # Firing_rate1 = np.mean(np.mean(np.array(hidden_act[k])[:, odors[1], :], axis=1), axis=1)
    # sensor neurons FR 
    Firing_rate0 = np.mean(np.mean(np.array(hidden_act[k])[:, odors[0], :], axis=1)[:, :par['sensor_region']], axis=1)
    Firing_rate1 = np.mean(np.mean(np.array(hidden_act[k])[:, odors[0], :], axis=1)[:, par['sensor_region']:par['sensor_region']+par['association_region']], axis=1)
    Firing_rate2 = np.mean(np.mean(np.array(hidden_act[k])[:, odors[0], :], axis=1)[:, -par['motor_region']:], axis=1)
    num = len(hidden_act[k])

    # fig, ax1 = plt.subplots()
    # ax1.plot(acc, color='blue', label='Accuracy')
    # ax1.set_xlabel('Epochs/10')
    # ax1.set_ylabel('Accuracy(%)')

    # ax2 = ax1.twinx()
    # ax2.plot(activity*10, color='red',label='Mean firing rate')
    # ax2.set_ylabel('Mean firing rate(Hz)')
    # fig.legend()

    plt.plot(Firing_rate0, label='Sensor neurons')
    plt.plot(Firing_rate1, label='Association neurons')
    plt.plot(Firing_rate2, label='Motor neurons')

    plt.plot([10, 10], [0, 1.5], c='k', linestyle='--', linewidth=0.5)
    plt.plot([20, 20], [0, 1.5], c='k', linestyle='--', linewidth=0.5)
    plt.plot([80, 80], [0, 1.5], c='k', linestyle='--', linewidth=0.5)

    plt.xlabel('Steps/100ms')
    plt.ylabel('Firing_rate')
    plt.xticks([0, 10, 20, 80])
    plt.legend()
    plt.savefig('./result/' + outfile +'FR.pdf') #, dpi=300 _acc_act
    plt.close()

def selecticity(hidden_act, k, odors, outfile):
    Firing_rate0 = np.mean(np.array(hidden_act[k])[:, odors[0], :], axis=1).T  #(time, neurons)
    Firing_rate1 = np.mean(np.array(hidden_act[k])[:, odors[1], :], axis=1).T #(time, neurons)
    Difference = Firing_rate0 - Firing_rate1
    sel_0 = list(set(np.where(Difference>0.1)[0]))
    sel_1 = list(set(np.where(Difference<-0.1)[0]))
    sel_0_1 = list(set(sel_0) & set(sel_1))
    sel_0 = list(set(sel_0) - set(sel_0_1))
    sel_1 = list(set(sel_1) - set(sel_0_1))
    no_memory = list(set(np.arange(len(Difference)))-set(sel_0)-set(sel_1)-set(sel_0_1))

    return sel_0, sel_1, sel_0_1
    Fir00_nor = normalization1(Firing_rate0[sel_0, :])
    index00, _ = sorting(Fir00_nor)
    Fir10_nor = normalization1(Firing_rate1[sel_0, :])
    Fir11_nor = normalization1(Firing_rate1[sel_1, :])
    Fir01_nor = normalization1(Firing_rate0[sel_1, :])
    index11, _ = sorting(Fir11_nor)

    # plt.figure(figsize=(32, 16))
    # plt.subplot(221)
    # fig=plt.imshow(Fir00_nor[index00,:], aspect='auto',cmap='jet')
    # plt.xlabel('Steps/100ms')
    # plt.gca().invert_yaxis()
    # plt.ylabel('Neuron_index')
    # plt.title('Selective neurons of odor0: ' + str(len(sel_0)))
    # plt.colorbar(fig, label="Normalized firing rate")
    # plt.plot([10, 10], [10, len(index00)-10], c='w', linestyle='--')
    # plt.plot([20, 20], [10, len(index00)-10], c='w', linestyle='--')
    # plt.plot([80, 80], [10, len(index00)-10], c='w', linestyle='--')
    

    # plt.subplot(222)
    # # fig1=plt.imshow(norm_sorting(Firing_rate1[sel_0, :]), aspect='auto',cmap='jet')
    # fig1=plt.imshow(Fir10_nor[index00,:], aspect='auto',cmap='jet')
    # plt.xlabel('Steps/100ms')
    # plt.gca().invert_yaxis()
    # plt.ylabel('Neuron_index')
    # plt.title('Selective neurons of odor0 in odor1 activity: ' + str(len(sel_0)))
    # plt.colorbar(fig1, label="Normalized firing rate")
    # plt.plot([10, 10], [10, len(index00)-10], c='w', linestyle='--')
    # plt.plot([20, 20], [10, len(index00)-10], c='w', linestyle='--')
    # plt.plot([80, 80], [10, len(index00)-10], c='w', linestyle='--')

    # plt.subplot(243)
    # fig2=plt.imshow(norm_sorting(Firing_rate1[sel_0_1, :]), aspect='auto',cmap='jet')# ,vmax=10#,'jet' vmin=0, vmax=1cmap=plt.cm.winter, , cmap='inferno' magma , cmap='jet', vmax=1, vmin=-1
    # plt.xlabel('Steps/100ms')
    # plt.gca().invert_yaxis()
    # plt.ylabel('Neuron_index')
    # plt.title('Selective neurons of odor0&1: ' + str(len(sel_0_1)))
    # plt.colorbar(fig2, label="Normalized firing rate")
    # plt.plot([10, 10], [10, len(sel_0_1)-10], c='w', linestyle='--')
    # plt.plot([20, 20], [10, len(sel_0_1)-10], c='w', linestyle='--')
    # plt.plot([80, 80], [10, len(sel_0_1)-10], c='w', linestyle='--')

    # plt.subplot(244)
    # fig2=plt.imshow(norm_sorting(Firing_rate1[no_memory, :]), aspect='auto',cmap='jet')# ,vmax=10#,'jet' vmin=0, vmax=1cmap=plt.cm.winter, , cmap='inferno' magma , cmap='jet', vmax=1, vmin=-1
    # plt.xlabel('Steps/100ms')
    # plt.gca().invert_yaxis()
    # plt.ylabel('Neuron_index')
    # plt.title('Non-selctive neurons: ' + str(len(no_memory)))
    # plt.colorbar(fig2, label="Normalized firing rate")
    # plt.plot([10, 10], [10, len(no_memory)-10], c='w', linestyle='--')
    # plt.plot([20, 20], [10, len(no_memory)-10], c='w', linestyle='--')
    # plt.plot([80, 80], [10, len(no_memory)-10], c='w', linestyle='--')

    # plt.subplot(223)
    # fig1=plt.imshow(Fir11_nor[index11,:], aspect='auto',cmap='jet')# ,vmax=10#,'jet' vmin=0, vmax=1cmap=plt.cm.winter, , cmap='inferno' magma , cmap='jet', vmax=1, vmin=-1
    # plt.xlabel('Steps/100ms')
    # plt.gca().invert_yaxis()
    # plt.ylabel('Neuron_index')
    # plt.title('Selective neurons of odor1: ' + str(len(sel_1)))
    # plt.colorbar(fig1, label="Normalized firing rate")
    # plt.plot([10, 10], [10, len(index11)-10], c='w', linestyle='--')
    # plt.plot([20, 20], [10, len(index11)-10], c='w', linestyle='--')
    # plt.plot([80, 80], [10, len(index11)-10], c='w', linestyle='--')    

    # plt.subplot(224)
    # fig1=plt.imshow(Fir01_nor[index11,:], aspect='auto',cmap='jet')# ,vmax=10#,'jet' vmin=0, vmax=1cmap=plt.cm.winter, , cmap='inferno' magma , cmap='jet', vmax=1, vmin=-1
    # plt.xlabel('Steps/100ms')
    # plt.gca().invert_yaxis()
    # plt.ylabel('Neuron_index')
    # plt.title('Selective neurons of odor1 in odor0 activity: ' + str(len(sel_1)))
    # plt.colorbar(fig1, label="Normalized firing rate")
    # plt.plot([10, 10], [10, len(index11)-10], c='w', linestyle='--')
    # plt.plot([20, 20], [10, len(index11)-10], c='w', linestyle='--')
    # plt.plot([80, 80], [10, len(index11)-10], c='w', linestyle='--')
    # plt.savefig('./result/' + 'selectivity' + '.pdf') #, dpi=300
    # plt.close()

def weight_pattern(data, weight, outfile):
    B = sorting(data[:,20:80])
    data = data[B, :]
    num = len(data)
    weight = weight[B, :]
    weight = weight[:, B]
    
    weight_sum0 = []
    weight_sum1 = []
    weight_sum2 = []
    weight_sum3 = []
    for i in range(len(weight)):
        weight_input_bef = sum(weight[:i, i])
        weight_input_aft = sum(weight[i:, i])
        weight_output_bef = sum(weight[i,:i ])
        weight_output_aft = sum(weight[i, i:])
        weight_sum0.append(weight_input_bef)
        weight_sum1.append(weight_input_aft)
        weight_sum2.append(weight_output_bef)
        weight_sum3.append(weight_output_aft)


    plt.figure(figsize=(18,5))
    plt.subplot(131)
    fig=plt.imshow(data, aspect='auto',cmap='jet')# ,vmax=10#,'jet' vmin=0, vmax=1cmap=plt.cm.winter, , cmap='inferno' magma , cmap='jet', vmax=1, vmin=-1
    plt.xlabel('Steps/100ms')
    plt.gca().invert_yaxis()
    plt.ylabel('Sorting neuron_index')
    plt.title('Neural activity' )
    plt.colorbar(fig, label="Normalized firing rate")
    plt.plot([10, 10], [9, num-10], c='w', linestyle='--')
    plt.plot([20, 20], [9, num-10], c='w', linestyle='--')
    plt.plot([80, 80], [9, num-10], c='w', linestyle='--')

    plt.subplot(132)
    plt.bar(x=0, bottom=np.arange(len(weight_sum0)), height=1.5, width=weight_sum0, orientation="horizontal", label='Before')
    plt.bar(x=0, bottom=np.arange(len(weight_sum1)), height=1.5, width=weight_sum1, orientation="horizontal", label='After')
    plt.xlabel('Strength value')
    plt.title('Neural input connection strength in trajectory' )
    plt.legend()

    plt.subplot(133)
    plt.bar(x=0, bottom=np.arange(len(weight_sum2)), height=1.5, width=weight_sum2, orientation="horizontal", label='Before')
    plt.bar(x=0, bottom=np.arange(len(weight_sum3)), height=1.5, width=weight_sum3, orientation="horizontal", label='After')
    plt.xlabel('Strength value')
    plt.title('Neural output connection strength in trajectory' )
    plt.legend()

    plt.savefig('./result/' + outfile + 'weights' + '.pdf') #, dpi=300
    plt.close()

def function(dir1, dir2, hidden_act, weights_rnn, acc, k, odor, outfile):
    netdir  = dir1 + dir2 + '.pkl'
    results = pickle.load(open(netdir, 'rb'))

    delay_time = results['parameters']['delay_time']//100
    odors_num = results['parameters']['num_motion_dirs']
    odor_sample = odor[k]
    weights_rnn0 = par['w_rnn0'] @ par['EI_matrix']

    odors = [[] for i in range(odors_num)]
    hidden_states = [[] for j in range(odors_num)]
    for i in range(odors_num):
            odors[i] = np.where(odor_sample==i)[0]
            hidden_states[i] = hidden_act[k][:, odors[i], :]
            hidden_states[i] = np.mean(hidden_states[i], axis=1).T
            hidden_states[i] = normalization1(hidden_states[i])
    hid_act = np.mean(hidden_act[-1], axis=1).T
    hid_act = normalization1(hid_act)
    plot_tra(hid_act, 60, 'trajectory')
    
    # trajectory_weights(hidden_states, hidden_states[0], weights_rnn0, weights_rnn, k, acc, outfile, delay_time) #+ 'trained'
    # acc_act(acc, hidden_act, k, odors, outfile)
    # selecticity(hidden_act, k, odors, outfile)
    # weight_pattern(hidden_states[0], weights_rnn0, outfile +'original')
    return func(hidden_act, k, odors)

def func(hidden_act, k, odors):
    #计算比例
    Firing_rate0 = np.mean(np.array(hidden_act[k])[:, odors[0], :], axis=1).T  #(time, neurons)
    Firing_rate1 = np.mean(np.array(hidden_act[k])[:, odors[1], :], axis=1).T #(time, neurons)
    Firing_rate = np.mean(np.array(hidden_act[k]), axis=1).T
    Difference = Firing_rate0 - Firing_rate1
    print(np.max(Difference), np.min(Difference))
    sel_0 = list(set(np.where(Difference>1)[0]))
    sel_1 = list(set(np.where(Difference<-1)[0]))
    sel_0_1 = list(set(sel_0) & set(sel_1))
    sel_0or1 = list(set(sel_0) | set(sel_1))
    # sel_0 = list(set(sel_0) - set(sel_0_1))
    # sel_1 = list(set(sel_1) - set(sel_0_1))
    no_memory = list(set(np.arange(len(Difference)))-set(sel_0)-set(sel_1)-set(sel_0_1))

    Fir00_nor = normalization1(Firing_rate0[sel_0, :])
    index00, loc00 = sorting(Fir00_nor)
    Fir11_nor = normalization1(Firing_rate1[sel_1, :])
    index11, loc11 = sorting(Fir11_nor)
    Fir_nor = normalization1(Firing_rate[sel_0_1, :])
    index_both, loc_both = sorting(Fir_nor)
    Fir_nor_or = Firing_rate[sel_0or1, :]
    # Fir_nor_or = normalization1(Firing_rate[sel_0or1, :])
    
    index_or, loc_or = sorting(Fir_nor_or)

    index_sel0 = np.array(sel_0)[index00]
    index_sel1 = np.array(sel_1)[index11]
    index_sel = np.array(sel_0_1)[index_both]

    loc0_sort = np.array(loc00)[index00]
    loc1_sort = np.array(loc11)[index11]
    loc_sort = np.array(loc_both)[index_both]
    num0 = []
    num1 = []
    num = []

    for i in range(6):
        index0 = np.where(loc0_sort < (i+1)*10)
        num0.append(np.array(index0).shape[1])
        index1 = np.where(loc1_sort < (i+1)*10)
        num1.append(np.array(index1).shape[1])
        index_bo = np.where(loc_sort < (i+1)*10)
        num.append(np.array(index_bo).shape[1])
    num0 = np.array(num0)
    num1 = np.array(num1)
    num = np.array(num)
    # rec_num = np.append(num0[0], np.diff(num0))
    x0 = np.split(index_sel0, num0)
    x1 = np.split(index_sel1, num1)
    x = np.split(index_sel, num)
    rec0 = []
    rec1 = []
    rec = []
    sen_region, mot_region, ass_region = 200, 200, 200
    for it in x0:
        num_sen = np.array(np.where(it<sen_region)).shape[1]
        num_asso = np.array(np.where(it<sen_region+mot_region)).shape[1]
        num_mot = np.array(np.where(it<sen_region+mot_region+ass_region)).shape[1]
        sub_rec = np.array([num_sen, num_asso-num_sen, num_mot-num_asso])
        rec0.append(sub_rec)
    for it in x1:
        num_sen = np.array(np.where(it<sen_region)).shape[1]
        num_asso = np.array(np.where(it<sen_region+mot_region)).shape[1]
        num_mot = np.array(np.where(it<sen_region+mot_region+ass_region)).shape[1]
        sub_rec = np.array([num_sen, num_asso-num_sen, num_mot-num_asso])
        rec1.append(sub_rec)
    for it in x:
        num_sen = np.array(np.where(it<sen_region)).shape[1]
        num_asso = np.array(np.where(it<sen_region+mot_region)).shape[1]
        num_mot = np.array(np.where(it<sen_region+mot_region+ass_region)).shape[1]
        sub_rec = np.array([num_sen, num_asso-num_sen, num_mot-num_asso])
        rec.append(sub_rec)
    
    return np.array(rec0), np.array(rec1), np.array(rec), Fir_nor_or

def plot_tra(data, delay_time, outfile, index=None):
    loc_all_0 = []
    mass_rec = []
    if index is None:
        for i in data[:,20:80]:
            loc, mass = c_mass(i)
            # if mass < 0.5:
            #     loc=0
            loc_all_0.append(loc) 
            mass_rec.append(mass)
            # mass_all.append(mass)
        B = np.argsort(loc_all_0)
        data0 = data[:,:][B,:]
    else:
        data0 = data[:,:][index,:]
    plt.figure(figsize=(10, 12))
    fig=plt.imshow(data0, aspect='auto',cmap='jet')
    plt.xlabel('Steps/100ms')
    plt.gca().invert_yaxis()
    plt.ylabel('Neuron_index')
    plt.colorbar(fig, label="Normalized firing rate")
    plt.plot([10, 10], [0, data.shape[0]], c='w', linestyle='--')
    plt.plot([20, 20], [0, data.shape[0]], c='w', linestyle='--')
    plt.plot([20+delay_time, 20+delay_time], [0, data.shape[0]], c='w', linestyle='--')
    plt.xticks(np.linspace(0, data0.shape[1], 10), fontsize=10)
    plt.yticks(np.linspace(0, data.shape[0], 6), fontsize=10)
    plt.xlim(0, data0.shape[1])  
    plt.ylim(0, data.shape[0])
    plt.savefig('./result/' + outfile + '.pdf', dpi=300)
    plt.close()
    if index is None:
        return B

def get_r(mean, std):
    r1 = list(map(lambda x: x[0]-x[1], zip(mean, std)))
    r2 = list(map(lambda x: x[0]+x[1], zip(mean, std))) 
    return r1, r2

def acc_plot(delay_, num_):
    fig, ax = plt.subplots(1,1)
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'gray', 'black']
    x = np.linspace(1, 1000, 250)
    for i in range(delay_):
        acc_rec = []
        for j in range(num_):
            path = './acc/delay{}s{}_acc_dis_noch_readall.txt'.format(i, j)
            if os.path.exists(path):
                acc0 = np.loadtxt(path)
                acc_rec.append(acc0)
                ax.plot(x, acc0, color=colors[i], linewidth=1, label='Delay'+str(i)+'s')
        # acc_rec = np.array(acc_rec)
        # acc_mean = np.mean(acc_rec, axis=0)
        # acc_std = np.std(acc_rec, axis=0)
        # r1, r2 = get_r(acc_mean, acc_std)
        # ax.plot(x, np.array(acc_mean), color=colors[i], linewidth=1, label='Delay'+str(i)+'s')
        # ax.fill_between(x, r1, r2, color=colors[i], alpha=0.2)
    plt.legend()
    plt.ylim(0.6, 1.0)
    # plt.yticks([0.8, 0.9, 1.0])
    plt.savefig('./acc.pdf')
    plt.close()

def plot_mean_fir():
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'gray', 'black']
    fig = plt.figure()
    # for i in range(8):
        # print('Start Delay time {}s'.format(i))
        # fig.add_subplot(2, 4, i+1)
    dir = './savedir/{}0_seed0_readall.pkl'.format(6)
    results = pickle.load(open(dir, 'rb'))
    hidden_act = results['hidden_act'][-1]
    hidden_act = np.mean(np.mean(np.square(hidden_act), axis=1), axis=1)
    steps = np.linspace(0, len(hidden_act)-1, len(hidden_act))
    plt.plot(steps, hidden_act, color=colors[7], label='Delay{}s'.format(6))
    plt.legend()
    plt.xticks(np.linspace(0,90,10))
    plt.savefig('./result/mean_fir.pdf')
    plt.close()

def dim_red():
    act_rec = []
    for i in range(1, 8):
        print('Start Delay time {}s'.format(i))
        dir = './savedir/{}0_seed0_readall.pkl'.format(i)
        results = pickle.load(open(dir, 'rb'))
        hidden_act = results['hidden_act'][-1]
        hidden_act = np.mean(hidden_act[20:i*10+20,:,:], axis=1).reshape(1, -1)
        steps = np.linspace(0, len(hidden_act)-1, len(hidden_act))
        act_rec.append(hidden_act)
    isomap_x0 = TSNE(n_components=3).fit_transform(act_rec)

def relu(inX):
    return np.maximum(0,inX)

def cal_TI(dir):
    results = pickle.load(open(dir, 'rb'))
    hidden_act = results['hidden_act'][-1]
    hidden_act = np.mean(hidden_act, axis=1).T #600,90
    data = relu(normalization1(hidden_act).T) #90,600
    # data = relu(hidden_act.T[20:80,])
    ts = data.shape[0]  # number of time points
    entrpy_bins = 60
    window_size = 4
    r_threshold = 0

    entrpy_max = entropy(np.ones(entrpy_bins)*data.shape[1]/entrpy_bins)
    # selected_indx = np.nonzero(np.mean(data, axis=0) > r_threshold)[0]
    selected_indx = np.where(np.max(data, axis=0) > r_threshold)[0]
    # selected_indx = np.array(selected_indx).squeeze()
    print(len(selected_indx))
    data = data[:, selected_indx]

    peak_times = np.argmax(data, axis=0)
    delay_peak_times = np.argmax(data[20:80,:], axis=0)
    end_times = np.clip(delay_peak_times + window_size + 1, 0, ts)
    start_times = np.clip(delay_peak_times - window_size, 0, ts)
    entrpy = entropy(np.histogram(delay_peak_times, entrpy_bins)[0])
    # entrpy = entropy(np.histogram(peak_times, entrpy_bins)[0] + 0.1 * np.ones(entrpy_bins))
    r2b_ratio = np.zeros(len(selected_indx))
    trans_index = 0
    for nind in range(len(selected_indx)):
        # mask = np.zeros(ts)
        # mask[int(start_times[nind]):int(end_times[nind])] = 1
        data0 = data[20:80, nind]
        # ridge = np.mean(data0[int(start_times[nind]):int(end_times[nind])])
        ridge = np.sum(data0[start_times[nind]:end_times[nind]])
        # backgr = np.mean(np.ma.MaskedArray(data0, mask))
        backgr = np.sum(data0)
        # r2b_ratio[nind] = np.log(ridge) - np.log(backgr)
        if backgr == 0:
            r2b_ratio[nind] = 0
        else:
            r2b_ratio[nind] = ridge/backgr
        trace_sum = ridge/(end_times[nind]-start_times[nind])
        trans_index += trace_sum

    trans_index /= len(selected_indx)
    entrpy = entrpy / entrpy_max
    r2b_ratio = np.nanmean(r2b_ratio)

    index1 = np.where(peak_times>=20)
    index2 = np.where(peak_times<80)
    index_delay = np.intersect1d(index1[0], index2[0])
    trans_len = len(index_delay)/len(selected_indx)
    # transient_index = np.sum(data)/(60*500)
    SI_trial_vec = r2b_ratio + entrpy
    Total = SI_trial_vec + trans_len
    return entrpy, r2b_ratio, round(SI_trial_vec, 3), trans_index, trans_len, round(Total, 3)

def W_heatmap(data, weights_rnn):
    # W权重Heatmap
    loc_all_0 = []
    mass_rec = []
    for i in data[:, 20:20+60]:
        loc, mass = c_mass(i)
        loc_all_0.append(loc) 
        mass_rec.append(mass)
        # mass_all.append(mass)
    index_B = np.argsort(loc_all_0)
    fig = plt.figure(figsize=(10, 8))
    index_B = index_B[300:550]
    weights_rnn = weights_rnn[index_B][:,index_B]
    # weights_rnn_ori = weights_rnn[np.sort(index_B)][:,np.sort(index_B)]
    rec = []
    sum_ = 25
    interval = 250//sum_
    for i in range(sum_):
        for j in range(sum_):
            j_rec = np.sum(weights_rnn[interval*i:interval*(i+1), interval*j:interval*(j+1)])
            rec.append(j_rec)
    rec = np.array(rec).reshape(sum_, sum_)
    norm = colors.Normalize(vmin=-2, vmax=5)
    fig = plt.imshow(rec, cmap='jet', interpolation='hanning', norm=norm)
    # fig = plt.imshow(rec, cmap='jet')
    plt.gca().invert_yaxis()
    plt.colorbar(fig)
    plt.xticks([0, 5, 10, 15, 20])
    plt.yticks([0, 5, 10, 15, 20])
    plt.savefig('./{}_hinning.pdf'.format('wrnn0'))
    plt.close()

    return weights_rnn, rec

def dim_iso():

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'gray', 'black']
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for i in range(1, 7):
        n_sample = 6
        dir = './savedir/{}0_seed0_readall.pkl'.format(i)
        results = pickle.load(open(dir, 'rb'))
        hid = results['hidden_act'][-1][:,:n_sample,20:20+10*i].transpose(1, 0, 2).reshape(n_sample, -1)
        hid_3 = Isomap(n_components=3).fit_transform(hid)
        ax.scatter(hid_3[:,0], hid_3[:,1], hid_3[:,2], color=colors[i])
    plt.savefig('./isomap.pdf', dpi=300)


if __name__ == '__main__':
    # dim_iso()
    # plot_mean_fir()
    # acc_plot(8,1)
    # dim_red()
    dir_rec = ['try_noloss', 'try_-2-1', 'try_-1-1', 'try_-1-2']
    for i in range(6):
        dir = './savedir/{}{}.pkl'.format(6, dir_rec[i])
        results = pickle.load(open(dir, 'rb'))
        hidden_act = np.mean(results['hidden_act'][-1], axis=1).T
        hidden_act = normalization1(hidden_act)
        print(hidden_act.shape)
        plot_tra(hidden_act, 60, dir_rec[i])
#cal TI
    # dirs1 = ['6noise0100_readall', '6noise0001_readall', '6noise0101_readall', '6noise0103_readall', '6noise0110_readall']
    # dirs2 = ['60_seed0_readall', '60_seed0_readall_notrans', '60_seed0_readall_full', '60_seed0_readall_notrans_full']
    # for it in dirs2:
    #     dir = './savedir/{}.pkl'.format(it)
    #     print(cal_TI(dir))
#记忆预测
    # dir = './savedir/{}.pkl'.format('3delay3x3_seed0') 
    # results = pickle.load(open(dir, 'rb'))
        # hidden_act = results['hidden_act'][-1]
        # hidden_act = np.mean(hidden_act, axis=1).T
        # Fir_act = normalization1(hidden_act)
        # B = plot_tra(Fir_act, 60, '0103')
    # w_rnn = results['weights']
    # w_rnn0 = par['w_rnn0']
    # print(w_rnn.shape)
    # hidden_act = results['hidden_act'][-1]
    # hidden_act = np.mean(hidden_act, axis=1).T #600,90
    # hidden_act = normalization1(hidden_act)
    # W_heatmap(hidden_act, w_rnn0)
    # acc = results['accuracy']
    # print(acc)

    # dir1= './savedir/'
    # dir2 = '50'
    # netdir  = dir1 + dir2 
    # results = pickle.load(open(netdir, 'rb'))

    # dir0 = './savedir/60'
    # results0 = pickle.load(open(dir0, 'rb'))

    # hidden_act0 = results0['hidden_act'][-1]
    # Fir_act0 = Fir_act0[B,:]
    # Fir_act = Fir_act[B,:]
    # plt.subplot(121)
    # fig=plt.imshow(Fir_act0, aspect='auto',cmap='jet')
    # plt.xlabel('Steps/100ms')
    # plt.gca().invert_yaxis()
    # plt.ylabel('Neuron_index')
    # plt.colorbar(fig, label="Normalized firing rate")
    # plt.plot([10, 10], [0, Fir_act0.shape[0]], c='w', linestyle='--')
    # plt.plot([20, 20], [0, Fir_act0.shape[0]], c='w', linestyle='--')
    # plt.plot([80, 80], [0, Fir_act0.shape[0]], c='w', linestyle='--')
    # # plt.xticks(np.linspace(0, Fir_act0.shape[1], 7), fontsize=10)
    # # plt.yticks(np.linspace(0, Fir_act0.shape[0], 6), fontsize=10)
    # plt.xlim(0, Fir_act0.shape[1])  
    # plt.ylim(0, Fir_act0.shape[0])


    # plt.subplot(122)
    # fig=plt.imshow(Fir_act, aspect='auto',cmap='jet')
    # plt.xlabel('Steps/100ms')
    # plt.gca().invert_yaxis()
    # plt.ylabel('Neuron_index')
    # plt.colorbar(fig, label="Normalized firing rate")
    # plt.plot([10, 10], [0, Fir_act.shape[0]], c='w', linestyle='--')
    # plt.plot([20, 20], [0, Fir_act.shape[0]], c='w', linestyle='--')
    # plt.plot([Fir_act.shape[1]-10, Fir_act.shape[1]-10], [0, Fir_act.shape[0]], c='w', linestyle='--')
    # plt.xlim(0, Fir_act.shape[1])  
    # plt.ylim(0, Fir_act0.shape[0])
    # plt.savefig('./result/' + '6svs1s' + '.pdf', dpi=300)
    # plt.close()



    # delay_time = results['parameters']['delay_time']//100
    # odor_sample = results['odor_sample'][-1]
    # odors_num = results['parameters']['num_motion_dirs']
    
    # weights_rnn0 = par['w_rnn0'] @ par['EI_matrix']

    # weights_rnn = results['weights']
    # hidden_act = results['hidden_act']   # (90,128,500)
    # acc = results['accuracy']
    # steps = np.linspace(1, 1000, 250)
    # plt.plot(steps, acc, c=colors[i], label=str(i)+'s Delay')
    # odors = [[] for i in range(odors_num)]
    # hidden_states = [[] for j in range(odors_num)]
    # for i in range(odors_num):
    #         odors[i] = np.where(odor_sample==i)[0]
    #         hidden_states[i] = hidden_act[k][:, odors[i], :]
    #         hidden_states[i] = np.mean(hidden_states[i], axis=1).T
    #         hidden_states[i] = normalization1(hidden_states[i])

    # trajectory_weights(hidden_states, hidden_states[1], weights_rnn0, weights_rnn, k, acc[k], dir2, delay_time) #+ 'trained'
    # # trajectory_weights(hidden_states, hidden_states[0], weights_rnn, weights_rnn0, len(acc), acc[-1], dir2 + 'notrain') #+ 'trained'
    # acc_act(acc, hidden_act, k, odors, dir2)
    # selecticity(hidden_act, k, odors, dir2)
    # weight_pattern(hidden_states[0], weights_rnn0, dir2+'original') #
    # func(hidden_act, k, odors)