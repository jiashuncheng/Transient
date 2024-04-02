"""
Nicolas Masse 2017
Contributions from Gregory Grant, Catherine Lee
"""

import torch
import torch.nn.functional as F
import numpy as np
import stimulus as stimulus
import pickle
import time
from parameters import par, args
import os, sys
from scipy.integrate import odeint
from scipy.stats import entropy

# Ignore "use compiled version of TensorFlow" errors
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# os.environ
torch.autograd.set_detect_anomaly(True)
print('Using EI Network:\t', par['EI'])
print('Synaptic configuration:\t', par['synapse_config'], "\n")

print('*'*20)
print('old_model_torch.py')

class Model:

    def __init__(self, device):

        # Load the input activity, the target data, and the training mask for this batch of trials    
        self.dt_tau_v = torch.tensor(par['dt']/par['tau_v'])
        self.dt_tau_r = torch.tensor(par['dt']/par['tau_r'])
        # self.dt_tau_v = torch.tensor(100/600)
        # self.dt_tau_r = torch.tensor(100/100)
        self.tran = par['transient']  
        self.device = device
         
        self.initialize_weights()
        
        # self.optimizer = torch.optim.Adam(params=[self.w_in, self.w_rnn, self.b_rnn, self.w_out, self.b_out], lr=0.01)
    def initialize_weights(self):
        # Initialize all weights. biases, and initial values

        self.var_dict = {}
        # all keys in par with a suffix of '0' are initial values of trainable variables
        for k, v in par.items():
            if k[-1] == '0':
                name = k[:-1]
                if name == 'w_rnn':
                    if par['EI']:
                        # ensure excitatory neurons only have postive outgoing weights,
                        # and inhibitory neurons have negative outgoing weights
                        self.var_dict[name] = par[k] @ par['EI_matrix']
                    else:
                        self.var_dict[name] = par[k]
                else :
                    self.var_dict[name] = torch.tensor(par[k]).to(self.device).requires_grad_(True)
            elif k == 'm':
                self.var_dict[k] = par[k] * np.ones(par['n_hidden'], dtype='float32')
                print(self.var_dict['m'])


        self.syn_x_init = torch.Tensor(par['syn_x_init']).to(self.device)
        self.syn_u_init = torch.Tensor(par['syn_u_init']).to(self.device)
               
        if self.tran:
            if args.learnable_m:
                self.m = torch.nn.Parameter(torch.tensor(self.var_dict['m']).to(self.device))#.to(self.device).requires_grad_(True)
            else:
                self.m = torch.tensor(self.var_dict['m']).to(self.device)
        else:
            self.m = torch.tensor(self.var_dict['m']).to(self.device)

        self.w_in  = self.var_dict['w_in']
        self.w_rnn = torch.tensor(self.var_dict['w_rnn']).to(self.device).requires_grad_(True)
        self.b_rnn = self.var_dict['b_rnn']
        self.w_out = self.var_dict['w_out']
        self.b_out = self.var_dict['b_out']

        # self.theta = torch.zeros(self.var_dict['h'].shape[1]).to(self.device).requires_grad_(True)


    # def run(self, input_data, target_data, mask, hidden_act, theta, index=None):
    def run(self, input_data, target_data, mask, index=None):

        self.input_data = torch.unbind(input_data, axis=0)
        self.target_data = target_data
        self.mask = mask

        self.h = []
        self.syn_x = []
        self.syn_u = []
        self.y = []
        self.theta = []
        
        
        h = self.var_dict['h'] #修改h
        theta = torch.zeros(par['n_hidden']).to(self.device).requires_grad_(True)
        theta = torch.normal(0, torch.ones(par['n_hidden'])).to(self.device).requires_grad_(True)
        # theta = theta.requires_grad_(True)
        # h = hidden_act
        syn_x = self.syn_x_init
        syn_u = self.syn_u_init
        # Loop through the neural inputs to the RNN, indexed in time
        for i, rnn_input in enumerate(self.input_data):
            h, syn_x, syn_u, theta = self.rnn_cell(rnn_input, h, syn_x, syn_u, theta)
            # h, syn_x, syn_u = self.rnn_cell(rnn_input, hidden_act[i], syn_x, syn_u)
            h = torch.clamp(h, 0, 100)
            self.h.append(h)
            self.syn_x.append(syn_x)
            self.syn_u.append(syn_u)
            self.y.append(F.relu(h @ self.w_out) + self.b_out)
            self.theta.append(theta)
            # if index is not None:
            #     w_out = self.w_out
            #     w_out[index,:] = 0
            #     # b_out = self.b_out
            #     # b_out[index,:] = 0
            #     self.y.append(F.relu(h @ w_out) + self.b_out)
            # elif index is None:
            #     self.y.append(F.relu(h @ self.w_out) + self.b_out)

        self.h = torch.stack(self.h)
        self.syn_x = torch.stack(self.syn_x)
        self.syn_u = torch.stack(self.syn_u)
        self.y = torch.stack(self.y)
        self.theta = torch.stack(self.theta)

        return self.y, self.h, self.theta
    

    def rnn_cell(self, rnn_input, h, syn_x, syn_u, theta):
        
        # Update neural activity and short-term synaptic plasticity values
        # Update the synaptic plasticity paramaters5
        if par['synapse_config'] is not None:
            # implement both synaptic short term facilitation and depression
            syn_x = syn_x.clone() + (torch.Tensor(par['alpha_std'])*(1-syn_x.clone()) - torch.tensor(par['dt_sec'])*syn_u.clone()*syn_x.clone()*h)*torch.Tensor(par['dynamic_synapse'])
            syn_u = syn_u.clone() + (torch.Tensor(par['alpha_stf'])*(torch.Tensor(par['U'])-syn_u.clone()) + torch.tensor(par['dt_sec'])*torch.tensor(par['U'])*(1-syn_u.clone())*h)*torch.Tensor(par['dynamic_synapse'])
            syn_x = torch.min(torch.tensor(1.), F.relu(syn_x))
            syn_u = torch.min(torch.tensor(1.), F.relu(syn_u))
            h_post = syn_u*syn_x*h

        else:
            # no synaptic plasticity
            h_post = h

        # Update the hidden state. Only use excitatory projections from input layer to RNN
        # All input and RNN activity will be non-negative
        if self.tran:
            h = F.relu(h * (1-par['alpha_neuron']) \
                + par['alpha_neuron'] * (rnn_input @ F.relu(self.w_in) + h_post @ self.w_rnn + self.b_rnn) \
                #  - self.theta.data \
                - theta)# \
                # + torch.empty(h.shape).normal_(torch.tensor(0.), torch.tensor(par['noise_rnn'])).to(self.device))
                 
            # self.theta = (1-self.dt_tau_v) * self.theta.data + F.relu(self.m) * self.dt_tau_v * h  # self.theta
            theta = (1-self.dt_tau_v) * theta + F.relu(self.m) * self.dt_tau_v * h

        else:
            h = F.relu(h * (1-par['alpha_neuron']) \
                + par['alpha_neuron'] * (rnn_input @ F.relu(self.w_in) + h_post @ self.w_rnn + self.b_rnn))
                # + torch.empty(h.shape).normal_(torch.tensor(0.), torch.tensor(par['noise_rnn'])).to(self.device))

        return h, syn_x, syn_u, theta          # 迭代部分因为h theta 没有梯度所以每次只计算一次没有retain graph
    

    def optimize(self, optimizer):

        # Calculate the loss functions and optimize the weights
        criterion = torch.nn.CrossEntropyLoss() #reduction='none'

        input_data  =  self.y.reshape(-1, par['n_output'])
        target_data =  torch.argmax(self.target_data.reshape(-1, par['n_output']), 1)
        
        perf_loss = criterion(input_data, target_data) # * self.mask
        perf_loss = torch.mean(perf_loss)
        # L2 penalty term on hidden state activity to encourage low spike rate solutions
        n = 2 if par['spike_regularization'] == 'L2' else 1
        spike_loss = torch.mean(self.h**n)
        weight_loss = torch.mean(F.relu(self.w_rnn)**n)
        weight_loss2 = torch.mean(F.relu(self.w_out)**n)

        loss = perf_loss + par['spike_cost'] * spike_loss + par['weight_cost'] * (weight_loss)
        
        optimizer.zero_grad()
        loss.backward(retain_graph=False)
        self.w_rnn.grad *= torch.Tensor(par['w_rnn_mask']).to(self.device)
        self.w_out.grad *= torch.Tensor(par['w_out_mask']).to(self.device)
        self.w_in.grad  *= torch.Tensor(par['w_in_mask']).to(self.device)
        # print(self.w_rnn.grad.mean())
        torch.nn.utils.clip_grad_norm_(self.w_rnn, torch.tensor(par['clip_max_grad_val']))
        torch.nn.utils.clip_grad_norm_(self.w_out, torch.tensor(par['clip_max_grad_val']))
        torch.nn.utils.clip_grad_norm_(self.w_in, torch.tensor(par['clip_max_grad_val']))
        if self.tran:
            torch.nn.utils.clip_grad_norm_(self.m, torch.tensor(par['clip_max_grad_val']/10))

        optimizer.step()

        return loss, perf_loss, spike_loss, weight_loss


def main(th, delay_, epo, gpu_id = None):

    device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu") 
    # if gpu_id is not None:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    # Print key parameters
    print_important_params()
     
    #Create the stimulus class to generate trial paramaters and input activity
    stim = stimulus.Stimulus()
    model = Model(device=device)

    if par['transient'] :
        optimizer = torch.optim.Adam(params=[model.w_in, model.w_out, model.w_rnn, model.b_rnn, model.b_out, model.m], lr=1e-3) # 5e-4
    else:
        optimizer = torch.optim.Adam(params=[model.w_in, model.w_out, model.w_rnn, model.b_rnn, model.b_out], lr=0.001) 
       
    model_performance = {'accuracy': [], 'loss': [], 'perf_loss': [], 'spike_loss': [], \
            'weight_loss': [], 'iteration': [], 'hidden_act':[], 'm':[], 'odor_sample':[]}
    

    
    for i in range(par['num_iterations']):

        # if i%2 == 0:
        #     # duration = 3000
        #     trial_info1 = stim.generate_trial(3000, set_rule = None)
        #     x = trial_info1['neural_input']
        #     t = trial_info1['desired_output'] 
        #     m = trial_info1['train_mask']
        # else:
        #     cue = np.zeros((90,64,24))
        #     # cue = np.random.normal(0.1, 10, size=(90,64,24))
        #     for i in range(64):
        #         if trial_info2['sample'][i] == 0:
        #             np.random.seed(0)
        #             cue[10:20,i,:] = np.random.normal(0., 0.5, size=(10,24))
        #         elif trial_info2['sample'][i] == 1:
        #             np.random.seed(1)
        #             cue[10:20,i,:] = np.random.normal(0., 1, size=(10,24))
        #     trial_info2 = stim.generate_trial(6000, set_rule = None)
        #     x = trial_info2['neural_input']
        #     x = x+cue
        #     t = trial_info2['desired_output'] 
        #     m = trial_info2['train_mask']
            # duration = 6000
        duration = (delay_)*1000
        trial_info1 = stim.generate_trial(duration, set_rule = None)
        x = trial_info1['neural_input']
        t = trial_info1['desired_output'] 
        m = trial_info1['train_mask']
        odor_sample = trial_info1['sample']
        # if i%2 == 0:
        #     # hidden_act = par['h0']
        #     hidden_act = 0.1*np.ones((par['batch_size'], par['n_hidden']))
        #     theta = torch.zeros(par['n_hidden'])
        # else:
        #     # hidden_act = np.mean(h.detach().cpu().numpy()[-1,:,:], axis=0)
        #     hidden_act = h.detach().cpu().numpy()[-1,:,:]
        #     theta = theta.detach().cpu().numpy()[-1,:,:]
        # output, h, theta = model.run(torch.Tensor(x).to(device), torch.Tensor(t).to(device), torch.Tensor(m).to(device), torch.Tensor(hidden_act).to(device), torch.Tensor(theta).to(device))
        output, h, theta = model.run(torch.Tensor(x).to(device), torch.Tensor(t).to(device), torch.Tensor(m).to(device))
        loss, perf_loss, spike_loss, weight_loss = model.optimize(optimizer)
        # odor_sample = trial_info['sample']
        target_max = np.argmax(t, axis=2)
        output_max = np.argmax(output.detach().cpu().numpy(), axis=2)
        accuracy = np.sum(np.float32(target_max == output_max))/len(target_max.reshape(-1,1))
        class_acc1 = np.sum(np.float32(target_max[10:20,:] == output_max[10:20,:]))/len(target_max[10:20,:].reshape(-1,1))
        decision_acc1 = np.sum(np.float32(target_max[20+delay_*10:30+delay_*10,:] == output_max[20+delay_*10:30+delay_*10,:]))/len(target_max[20+delay_*10:30+delay_*10,:].reshape(-1,1))
        # class_acc2 = np.sum(np.float32(target_max[40+delay_*10:50+delay_*10,:] == output_max[40+delay_*10:50+delay_*10,:]))/len(target_max[40+delay_*10:50+delay_*10,:].reshape(-1,1))
        # decision_acc2 = np.sum(np.float32(target_max[50+delay_*20:60+delay_*20,:] == output_max[50+delay_*20:60+delay_*20,:]))/len(target_max[50+delay_*20:60+delay_*20,:].reshape(-1,1))
        # class_acc3 = np.sum(np.float32(target_max[70+delay_*20:80+delay_*20,:] == output_max[70+delay_*20:80+delay_*20,:]))/len(target_max[70+delay_*20:80+delay_*20,:].reshape(-1,1))
        decision_acc3 = np.sum(np.float32(target_max[-10:,:] == output_max[-10:,:]))/len(target_max[-10:,:].reshape(-1,1))
        # Save the network model and output model performance to screen
        if i % par['iters_between_outputs']==0:
            print_results(i, perf_loss, spike_loss, weight_loss, h, accuracy, class_acc1, decision_acc1, decision_acc3)
            model_performance = append_model_performance(model_performance, decision_acc3, loss.detach().cpu().numpy(), perf_loss.detach().cpu().numpy(), spike_loss.detach().cpu().numpy(), weight_loss.detach().cpu().numpy(), i, h.detach().cpu().numpy(), model.m.detach().cpu().numpy(),odor_sample)
        if i == par['num_iterations']-1:
            print(model.m)
            print(torch.mean(h**2))
    #Save model and results
    save_name = '{}'.format(th)
    save_results(model_performance, model.w_rnn.detach().cpu().numpy(), save_fn=save_name + '.pkl')
    # torch.save(model, './savedir/' +  save_name + '.pth' )
    # np.savetxt('./acc/{}{}.txt'.format(th, delay_), model_performance['accuracy'])

def normalization1(data):
    data = data - np.mean(data, axis=1)[:,np.newaxis]
    data = data/(np.max(np.abs(data), axis=1)[:,np.newaxis]+0.00001)
    return data

def relu(inX):
    return np.maximum(0,inX)

def cal_TI(h):
    hidden_act = h
    hidden_act = np.mean(hidden_act, axis=1).T #600,90
    data = relu(normalization1(hidden_act).T) #90,600
    # data = relu(hidden_act.T[20:80,])
    ts = data.shape[0]  # number of time points
    entrpy_bins = 60
    window_size = 4
    r_threshold = 0

    # selected_indx = np.nonzero(np.mean(data, axis=0) > r_threshold)[0]
    selected_indx = np.where(np.max(data, axis=0) > r_threshold)[0]
    # selected_indx = np.array(selected_indx).squeeze()
    data = data[:, selected_indx]

    peak_times = np.argmax(data, axis=0)
    delay_peak_times = np.argmax(data[20:80,:], axis=0)
    index1 = np.where(data[20:80,:]>0.4)
    end_times = np.clip(delay_peak_times + window_size + 1, 0, ts)
    start_times = np.clip(delay_peak_times - window_size, 0, ts)
    # entrpy = entropy(np.histogram(delay_peak_times, entrpy_bins)[0]) 
    entrpy = entropy(np.histogram(delay_peak_times, entrpy_bins)[0])
    entrpy_ori = entropy(np.histogram(delay_peak_times, entrpy_bins)[0], base=2)
    entrpy_max = entropy(np.ones(entrpy_bins)*data.shape[1]/entrpy_bins)
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
    return round(Total, 3), entrpy_ori, entrpy, r2b_ratio, round(SI_trial_vec, 3), trans_index, trans_len

def save_results(model_performance, weights, save_fn = None):

    results = {'weights': weights, 'parameters': par}
    for k,v in model_performance.items():
        results[k] = v
    if save_fn is None:
        fn = par['save_dir'] + par['save_fn']
    else:
        fn = par['save_dir'] + save_fn
    pickle.dump(results, open(fn, 'wb'))
    print('Model results saved in ',fn)

def append_model_performance(model_performance, accuracy, loss, perf_loss, spike_loss, weight_loss, iteration, h, m, odor_sample):

    model_performance['accuracy'].append(accuracy)
    model_performance['loss'].append(loss)
    model_performance['perf_loss'].append(perf_loss)
    model_performance['spike_loss'].append(spike_loss)
    model_performance['weight_loss'].append(weight_loss)
    model_performance['iteration'].append(iteration)
    model_performance['hidden_act'].append(h)
    model_performance['m'].append(m)
    model_performance['odor_sample'].append(odor_sample)
    return model_performance

def print_results(iter_num, perf_loss, spike_loss, weight_loss, h, accuracy, class_acc1, decision_acc1, decision_acc3):

    print(par['trial_type'] + ' Iter. {:4d}'.format(iter_num) + ' | Accuracy {:0.4f}'.format(accuracy) +
        ' |Class Accuracy {:0.4f}'.format(class_acc1) + ' |Dicision Accuracy {:0.4f}'.format(decision_acc3) +
      ' | Perf loss {:0.4f}'.format(perf_loss) + ' | Spike loss {:0.4f}'.format(spike_loss) +
      ' | Weight loss {:0.4f}'.format(weight_loss) + ' | Mean activity {:0.4f}'.format(torch.mean(h)))

def print_important_params():

    important_params = ['num_iterations', 'learning_rate', 'noise_rnn_sd', 'noise_in_sd','spike_cost',\
        'spike_regularization', 'weight_cost','test_cost_multiplier', 'trial_type','balance_EI', 'dt',\
        'delay_time', 'connection_prob','synapse_config','tau_slow','tau_fast']
    for k in important_params:
        print(k, ': ', par[k])

def set_seed():
    np.random.seed(par['seed'])
    torch.manual_seed(par['seed'])
    torch.cuda.manual_seed(par['seed'])

if __name__ == '__main__':
    # record = []
    # for i in range(1,11):
    #     par['m'] = i*0.1
    #     sub_record = []
    #     for j in range(3):
    #         delay_ = 6
    #         # name = 'loss_{}_{}_f_m{}_noise{}_odors{}'.format(par['n_hidden'], par['dt'], par['m'], par['noise_rnn_sd'], par['num_motion_dirs'])
    #         name = 'm{}{}'.format(j, i)
    #         energy = main(name, delay_, i,  gpu_id = 0)
    #         sub_record.append(energy)
    #         print(energy)
    #     print(sub_record)
    #     record.append(sub_record)
    # print(record)
    # np.savetxt('./data/acc.txt', record)
    set_seed()
    name = '{}'.format(args.exp)
    delay_ = 6
    main(name, delay_, 0,  gpu_id = args.gpu)