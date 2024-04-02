"""
Nicolas Masse 2017
Contributions from Gregory Grant, Catherine Lee
"""

import datetime
import json
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import stimulus as stimulus
import pickle
import time
import os, sys
from scipy.integrate import odeint
from scipy.stats import entropy
from parameters import par
from config import args
import logging

class Model(nn.Module):

    def __init__(self, device):
        super(Model, self).__init__()

        # Load the input activity, the target data, and the training mask for this batch of trials    
        self.dt_tau_v = torch.tensor(0.1)
        self.dt_tau_r = torch.tensor(par['dt']/par['tau_r'])
        # self.dt_tau_v = torch.tensor(100/600)
        # self.dt_tau_r = torch.tensor(100/100)
        self.tran = par['transient']  
        self.alpha = par['alpha_neuron']
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
                        self.var_dict[name] = par[k] @ par['EI_matrix'] #权重矩阵
                    else:
                        self.var_dict[name] = par[k]
                else :
                    self.var_dict[name] = torch.tensor(par[k]).to(self.device).requires_grad_(True)
            elif k == 'm':
                self.var_dict[k] = par[k] * np.ones(par['n_hidden'], dtype='float32')

        self.syn_x_init = torch.Tensor(par['syn_x_init']).to(self.device)
        self.syn_u_init = torch.Tensor(par['syn_u_init']).to(self.device)

        self.w_in  = nn.Parameter(self.var_dict['w_in'])
        self.w_rnn = nn.Parameter(torch.tensor(self.var_dict['w_rnn']))
        self.b_rnn = nn.Parameter(self.var_dict['b_rnn'])
        self.w_out = nn.Parameter(self.var_dict['w_out'])
        self.b_out = nn.Parameter(self.var_dict['b_out'])

        input_sz, hidden_sz = self.w_in.shape
        #TODO - version 1,2,6
        stdv = 0.1 / math.sqrt(hidden_sz)
        #TODO - version 3
        # stdv = 1 / math.sqrt(hidden_sz)

        self.w_m = nn.Parameter(torch.Tensor(input_sz,hidden_sz))
        self.u_m = nn.Parameter(torch.Tensor(hidden_sz,hidden_sz))
        # nn.init.xavier_normal_(self.w_m)
        # nn.init.xavier_normal_(self.u_m)
        nn.init.uniform_(self.w_m, a=-stdv, b=stdv)
        nn.init.uniform_(self.u_m, a=-stdv, b=stdv)

        self.w_ga = nn.Parameter(torch.Tensor(input_sz,hidden_sz))
        self.u_ga = nn.Parameter(torch.Tensor(hidden_sz,hidden_sz))
        # nn.init.xavier_normal_(self.w_ga)
        # nn.init.xavier_normal_(self.u_ga)
        nn.init.uniform_(self.w_ga, a=-stdv, b=stdv)
        nn.init.uniform_(self.u_ga, a=-stdv, b=stdv)
        
        self.w_h = nn.Parameter(torch.Tensor(input_sz,hidden_sz))
        # nn.init.xavier_normal_(self.w_h)
        nn.init.uniform_(self.w_h, a=0, b=stdv)

        self.m = torch.tensor(par['v_m']).to(self.device)
        self.gamma = torch.tensor(par['v_gamma']).to(self.device)

        if not args.transient and not args.regions and args.connection_prob == 1.0:
            self.b_rnn = nn.Parameter(torch.ones(1,hidden_sz) * 1)

        # self.theta = torch.zeros(self.var_dict['h'].shape[1]).to(self.device).requires_grad_(True)


    # def run(self, input_data, target_data, mask, hidden_act, theta, index=None):
    def run(self, input_data, target_data, mask, index=None):

        self.input_data = torch.unbind(input_data, axis=0)
        self.target_data = target_data
        self.mask = mask #delay期间mask掉

        self.h = []
        self.syn_x = []
        self.syn_u = []
        self.y = []
        self.theta = []
        
        
        h = self.var_dict['h'] #修改h
        theta = torch.normal(0, torch.ones(par['n_hidden'])).to(self.device) if args.noise_h else torch.zeros(par['n_hidden']).to(self.device)
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
            self.y.append(F.relu(h @ self.w_out + self.b_out))
            self.theta.append(theta)

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
            
        noise = torch.empty(h.shape).normal_(torch.tensor(0.), torch.tensor(par['noise_rnn'])).to(self.device) if args.noise_h else 0.
        if self.tran:
            #TODO - 20240126 old version
            # self.m = F.relu(rnn_input @ self.w_m + h @ self.u_m + self.b_m)
            # # self.m = torch.min(self.m, torch.ones_like(self.m).to(self.device) * 1.0)
            # self.gamma = F.relu(rnn_input @ self.w_ga + h @ self.u_ga + self.b_ga)
            # # self.gamma = torch.min(self.gamma, torch.ones_like(self.gamma).to(self.device) * 1.0)
            # h = (1-self.alpha) * h + \
            #     self.alpha * F.relu((rnn_input @ self.w_in + h_post @ self.w_rnn + self.b_rnn - self.gamma * theta) + noise)
            # theta = (1-self.dt_tau_v) * theta + self.dt_tau_v * F.relu(self.m * h+ rnn_input @ self.w_h)#抑制强度
            #TODO - 20240126_2 old version
            # h = (1-self.alpha) * h + \
            #     self.alpha * F.relu((rnn_input @ self.w_in + h_post @ self.w_rnn + self.b_rnn - theta @ self.gamma) + noise)
            # theta = (1-self.dt_tau_v) * theta + self.dt_tau_v * F.relu(h @ self.m + rnn_input @ self.w_h)
            #TODO - version 1
            # self.m = F.relu(rnn_input @ self.w_m + h @ self.u_m)
            # self.gamma = F.relu(rnn_input @ self.w_ga + h @ self.u_ga)
            # h = (1-self.alpha) * h + \
            #     self.alpha * F.relu((rnn_input @ self.w_in + h_post @ self.w_rnn + self.b_rnn - self.gamma * theta) + noise)
            # theta = (1-self.dt_tau_v) * theta + self.dt_tau_v * F.relu(self.m * h+ rnn_input @ self.w_h)
            #TODO - version 2,3
            # self.m = F.relu(rnn_input @ self.w_m + h @ self.u_m)
            # self.gamma = F.relu(rnn_input @ self.w_ga + h @ self.u_ga)
            # h = (1-self.alpha) * h + \
            #     self.alpha * F.relu((rnn_input @ self.w_in + h_post @ self.w_rnn + self.b_rnn - self.gamma * theta) + noise)
            # theta = (1-self.dt_tau_v) * theta + self.dt_tau_v * F.relu(self.m * h)
            #TODO - version 4
            h = (1-self.alpha) * h + \
                self.alpha * F.relu((rnn_input @ self.w_in + h_post @ self.w_rnn + self.b_rnn - self.gamma * theta) + noise)
            theta = (1-self.dt_tau_v) * theta + self.dt_tau_v * F.relu(self.m * h)
        else:
            h = (1-self.alpha) * h + \
                self.alpha * F.relu((rnn_input @ self.w_in + h_post @ self.w_rnn + self.b_rnn) + args.rnn_bias + noise)

        return h, syn_x, syn_u, theta          # 迭代部分因为h theta 没有梯度所以每次只计算一次没有retain graph
    

    def optimize(self, optimizer):

        # Calculate the loss functions and optimize the weights
        if args.delay == 'random':
            criterion = torch.nn.CrossEntropyLoss(reduction='none')

            input_data  =  self.y.reshape(-1, par['n_output'])
            target_data =  torch.argmax(self.target_data.reshape(-1, par['n_output']), 1)
            
            perf_loss = criterion(input_data, target_data)  * self.mask.reshape(-1)
            perf_loss = torch.sum(perf_loss)/self.mask.sum()
        else:
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

        if args.connection_prob != 1.0:
            loss = perf_loss + par['spike_cost'] * spike_loss + par['weight_cost'] * (weight_loss) 
        else:
            loss = perf_loss + par['spike_cost'] * spike_loss
        
        optimizer.zero_grad()
        loss.backward(retain_graph=False)
        self.w_rnn.grad *= torch.Tensor(par['w_rnn_mask']).to(self.device)
        self.w_out.grad *= torch.Tensor(par['w_out_mask']).to(self.device)
        self.w_in.grad  *= torch.Tensor(par['w_in_mask']).to(self.device)
        torch.nn.utils.clip_grad_norm_(self.w_rnn, torch.tensor(par['clip_max_grad_val']))
        torch.nn.utils.clip_grad_norm_(self.w_out, torch.tensor(par['clip_max_grad_val']))
        torch.nn.utils.clip_grad_norm_(self.w_in, torch.tensor(par['clip_max_grad_val']))

        optimizer.step()

        return loss, perf_loss, spike_loss, weight_loss
    
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

def append_model_performance(model_performance, accuracy, loss, perf_loss, spike_loss, weight_loss, iteration, h, m, gamma, odor_sample, delay):
    model_performance['accuracy'].append(accuracy)
    model_performance['loss'].append(loss)
    model_performance['perf_loss'].append(perf_loss)
    model_performance['spike_loss'].append(spike_loss)
    model_performance['weight_loss'].append(weight_loss)
    model_performance['iteration'].append(iteration)
    model_performance['hidden_act']=h
    model_performance['m']=m
    model_performance['gamma']=gamma
    model_performance['odor_sample']=odor_sample
    model_performance['delay']=delay
    return model_performance

def print_results(iter_num, perf_loss, spike_loss, weight_loss, h, accuracy, class_acc1, decision_acc3):
    logging.info(par['trial_type'] + ' Iter. {:4d}'.format(iter_num) + ' | Accuracy {:0.4f}'.format(accuracy) +
        ' |Class Accuracy {:0.4f}'.format(class_acc1) + ' |Dicision Accuracy {:0.4f}'.format(decision_acc3) +
      ' | Perf loss {:0.4f}'.format(perf_loss) + ' | Spike loss {:0.4f}'.format(spike_loss) +
      ' | Weight loss {:0.4f}'.format(weight_loss) + ' | Mean activity {:0.4f}'.format(torch.mean(h)))

def save_model(model, i):
    logging.info(' '.join(['Model params saved in:\n',file_path + f'/{formatted_time}_model_params_step{i}.pth']))
    torch.save(model.state_dict(), file_path + f'/{formatted_time}_model_params_step{i}.pth')

def save_results(model_performance, weights, i):
    results = {}
    for k,v in model_performance.items():
        results[k] = v
    file_path_model = file_path + f'/{formatted_time}_model_{i}'
    if args.mode == 'test':
        file_path_model += f'_sample_{args.sample}_match_{args.match}_theta_{args.theta}_noiseh_{args.noise_h}'
    file_path_model += '.pt'

    logging.info(' '.join(['Model results saved in:\n',file_path_model]))
    torch.save(results, file_path_model)
    try:
        file_path_model = file_path + f"/{formatted_time}_model_{i-par['iters_between_save_model']}.pt"
        logging.info(' '.join(['Remove model results: ', file_path_model]))
        os.remove(file_path_model)
    except:
        pass

def main(delay_, gpu_id = None):

    device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else "cpu") 
     
    # Create the stimulus class to generate trial paramaters and input activity
    stim = stimulus.Stimulus()
    model = Model(device=device)
    model.to(device)
    if args.mode == 'test':
        model.load_state_dict(torch.load(args.model_path))
        model.eval()

    if args.save_model:
        save_model(model, 0)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) 
       
    model_performance = {'accuracy': [], 'loss': [], 'perf_loss': [], 'spike_loss': [], \
            'weight_loss': [], 'iteration': [], 'hidden_act':[], 'm':[], 'gamma': [], 'odor_sample':[], 'delay': []}

    for i in range(1, par['num_iterations'] + 1):
        duration = (delay_)*1000 #delay_ 秒的时间
        trial_info1 = stim.generate_trial(duration, set_rule = None)
        x = trial_info1['neural_input']
        t = trial_info1['desired_output'] 
        m = trial_info1['train_mask']
        odor_sample = trial_info1['sample']
        match = trial_info1['match']
        test = trial_info1['test']
        delay = trial_info1['delay']

        output, h, theta = model.run(torch.Tensor(x).to(device), torch.Tensor(t).to(device), torch.Tensor(m).to(device))
        
        if args.mode == 'train':
            loss, perf_loss, spike_loss, weight_loss = model.optimize(optimizer)
        else:
            loss, perf_loss, spike_loss, weight_loss = torch.tensor(0.), torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)
        
        target_max = np.argmax(t, axis=2)
        output_max = np.argmax(output.detach().cpu().numpy(), axis=2)
        accuracy = np.sum(np.float32(target_max == output_max))/len(target_max.reshape(-1,1))
        n=(par['dead_time']+par['fix_time'])//par['dt']
        n1=(par['dead_time']+par['fix_time']+par['sample_time'])//par['dt']
        class_acc1 = np.sum(np.float32(target_max[n:n1,:] == output_max[n:n1,:]))/len(target_max[n:n1,:].reshape(-1,1))
        if args.delay == 'random':
            decision_acc3 = 0
            for batch in range(delay.shape[0]):
                n=(par['dead_time']+par['fix_time']+par['sample_time'])//par['dt']
                n1 = (par['dead_time']+par['fix_time']+par['sample_time']+par['test_time'])//par['dt']
                true_sample = target_max[n+delay[batch]:n1+delay[batch],batch] == output_max[n+delay[batch]:n1+delay[batch],batch]
                sum_sample = target_max[n+delay[batch]:n1+delay[batch],batch].reshape(-1,1)
                decision_acc3 += np.sum(np.float32(true_sample))/len(sum_sample)
            decision_acc3 = decision_acc3/delay.shape[0]
        else:
            n=(par['dead_time']+par['fix_time']+par['sample_time'])//par['dt']
            n1 = (par['dead_time']+par['fix_time']+par['sample_time']+par['test_time'])//par['dt']
            decision_acc3 = np.sum(np.float32(target_max[n+delay[0]:n1+delay[0],:] == output_max[n+delay[0]:n1+delay[0],:]))/len(target_max[n+delay[0]:n1+delay[0],:].reshape(-1,1))
        
        # Save the network model and output model performance to screen
        if i % par['iters_between_outputs'] == 0 or i == 1:
            print_results(i, perf_loss, spike_loss, weight_loss, h, accuracy, class_acc1, decision_acc3)
            model_performance = append_model_performance(model_performance, decision_acc3, loss.detach().cpu().numpy(), perf_loss.detach().cpu().numpy(), spike_loss.detach().cpu().numpy(), weight_loss.detach().cpu().numpy(), i, h.detach().cpu().numpy(), model.m, model.gamma, odor_sample, delay)
        if i % par['iters_between_save_model'] == 0 or i == 1:
            try:
                logging.debug(";".join(['m',str(model.m.max().item()), str(model.m.min().item()), str(model.m.mean().item())]))
                logging.debug(";".join(['gamma',str(model.gamma.max().item()), str(model.gamma.min().item()), str(model.gamma.mean().item())]))
                logging.debug(";".join(['h',str(torch.max(h).item()), str(torch.min(h).item()), str(torch.mean(h).item())]))
            except:
                logging.debug(";".join(['h',str(torch.max(h).item()), str(torch.min(h).item()), str(torch.mean(h).item())]))
            save_results(model_performance, model.w_rnn.detach().cpu().numpy(), i)
        if args.mode == 'test':
            break
    if args.save_model:
        save_model(model, i)

def convert_to_serializable(obj):
    try:
        return json.dumps(obj)
    except TypeError:
        return str(obj)

def set_seed(seed):
    torch.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def save_params():
    with open(file_path+f'/{formatted_time}_params.json', 'w') as json_file:
        json.dump([vars(args), par], json_file, default=convert_to_serializable, indent=2)

def check_path():
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def set_logger():
    # 配置日志
    logging.basicConfig(
        level=logging.DEBUG,  # 设置日志级别为INFO
        format='[%(asctime)s][%(levelname)s][%(message)s]',  # 设置日志格式
        filename=file_path+f'/{formatted_time}_log.log',  # 指定日志文件名称
        filemode='w'  # 设置文件写入模式为覆盖写入
    )
    # 输出日志到控制台
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[\033[1;32m%(asctime)s\033[0m][\033[1;34m%(levelname)s\033[0m][%(message)s]')
    console_handler.setFormatter(formatter)
    logging.getLogger('').addHandler(console_handler)

def init_path():
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y%m%d%H%M%S")
    file_path = f"/home/jiashuncheng/code/Trasient/ODPA/new/savedir/exp_{args.exp}_tansient_{args.transient}_connection_{args.connection_prob}_region_{args.regions}_mode_{args.mode}_delay_{args.delay}_hidd_{args.hidden_num}_seed_{args.seed}_lr{args.lr}_vm{args.v_m}_vgamma{args.v_gamma}_alpha{args.alpha}"
    return formatted_time, file_path

if __name__ == '__main__':
    formatted_time, file_path = init_path()
    check_path()
    save_params()

    set_logger()
    set_seed(args.seed)
    main(delay_=args.delay, gpu_id=args.gpu)