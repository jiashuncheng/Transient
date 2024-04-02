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
from parameters import par
import os, sys
from scipy.integrate import odeint
from scipy.stats import entropy
from parameters import args

# Ignore "use compiled version of TensorFlow" errors
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# os.environ
torch.autograd.set_detect_anomaly(True)

class Model:

    def __init__(self, device):

        # Load the input activity, the target data, and the training mask for this batch of trials    
        self.dt_tau_v = torch.tensor(par['dt']/par['tau_v'])
        self.dt_tau_r = torch.tensor(par['dt']/par['tau_r'])
        # self.dt_tau_v = torch.tensor(100/600)
        # self.dt_tau_r = torch.tensor(100/100)
        self.tran = par['transient']  
        self.noise = par['noise']
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
                # print(self.var_dict['m'])


        self.syn_x_init = torch.Tensor(par['syn_x_init']).to(self.device)
        self.syn_u_init = torch.Tensor(par['syn_u_init']).to(self.device)
               
        if self.tran:
            if par['learnable_m']:
                self.m = torch.nn.Parameter(torch.tensor(self.var_dict['m']).to(self.device))
            else:
                self.m = (torch.ones((par['n_hidden'])) * self.var_dict['m']).to(self.device)
        else:
            self.m = torch.tensor(self.var_dict['m']).to(self.device)

        self.w_in  = self.var_dict['w_in']
        self.w_rnn = torch.tensor(self.var_dict['w_rnn']).to(self.device).requires_grad_(True)
        self.b_rnn = self.var_dict['b_rnn']
        self.w_out = self.var_dict['w_out']
        self.b_out = self.var_dict['b_out']

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
        if self.noise:
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
            # with open('/home/jiashuncheng/code/new_ppo/Trajectory_Code/savedir/w_2output.pkl', 'wb') as handle:
            #     pickle.dump(self.w_out, handle)
            # with open('/home/jiashuncheng/code/new_ppo/Trajectory_Code/savedir/w_2rnn.pkl', 'wb') as handle:
            #     pickle.dump(self.w_rnn, handle)
            # with open('/home/jiashuncheng/code/new_ppo/Trajectory_Code/savedir/b_1output.pkl', 'wb') as handle:
            #     pickle.dump(self.b_out, handle)
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
            if self.noise:
                h = F.relu(h * (1-par['alpha_neuron']) + par['alpha_neuron'] * (rnn_input @ F.relu(self.w_in) + h_post @ self.w_rnn + self.b_rnn) - theta \
                + torch.empty(h.shape).normal_(torch.tensor(0.), torch.tensor(par['noise_rnn'])).to(self.device))
                 
                theta = (1-self.dt_tau_v) * theta + F.relu(self.m) * self.dt_tau_v * h
            else:
                h = F.relu(h * (1-par['alpha_neuron']) + par['alpha_neuron'] * (rnn_input @ F.relu(self.w_in) + h_post @ self.w_rnn + self.b_rnn) - theta)
                 
                theta = (1-self.dt_tau_v) * theta + F.relu(self.m) * self.dt_tau_v * h

        else:
            if self.noise:
                F.relu(h + (rnn_input @ F.relu(self.w_in) + h_post @ self.w_rnn + self.b_rnn) \
                + torch.empty(h.shape).normal_(torch.tensor(0.), torch.tensor(par['noise_rnn'])).to(self.device))
            else:
                h = F.relu(h + (rnn_input @ F.relu(self.w_in) + h_post @ self.w_rnn + self.b_rnn))          

        return h, syn_x, syn_u, theta          # 迭代部分因为h theta 没有梯度所以每次只计算一次没有retain graph
    

    def optimize(self, optimizer):

        # Calculate the loss functions and optimize the weights
        if par['random']:
            criterion = torch.nn.CrossEntropyLoss(reduction='none')

            input_data  =  self.y.reshape(-1, par['n_output'])
            target_data =  torch.argmax(self.target_data.reshape(-1, par['n_output']), 1)
            
            perf_loss = criterion(input_data, target_data)  * self.mask.reshape(-1)
            perf_loss = torch.sum(perf_loss)/self.mask.sum()
        else:
            criterion = torch.nn.CrossEntropyLoss()

            input_data  =  self.y.reshape(-1, par['n_output'])
            target_data =  torch.argmax(self.target_data.reshape(-1, par['n_output']), 1)
            
            perf_loss = criterion(input_data, target_data) # * self.mask
            perf_loss = torch.mean(perf_loss)
        # L2 penalty term on hidden state activity to encourage low spike rate solutions
        n = 2 if par['spike_regularization'] == 'L2' else 1
        spike_loss = torch.mean(self.h**n)
        weight_loss = torch.mean(F.relu(self.w_rnn)**n)
        weight_loss2 = torch.mean(F.relu(self.w_out)**n)

        if n==2:
            loss = perf_loss + par['spike_cost'] * spike_loss  + par['l2_cost'] * (weight_loss)
        elif n==1:
            loss = perf_loss + par['spike_cost'] * spike_loss  + par['weight_cost'] *weight_loss
        else:
            loss = perf_loss + par['spike_cost'] * spike_loss
        
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


def main(save_name):

    os.environ["CUDA_VISIBLE_DEVICES"] = str('gpu')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
     
    #Create the stimulus class to generate trial paramaters and input activity
    stim = stimulus.Stimulus()
    model = Model(device=device)

    if par['transient'] :
        optimizer = torch.optim.Adam(params=[model.w_in, model.w_out, model.w_rnn, model.b_rnn, model.b_out, model.m], lr=5e-4)
    else:
        optimizer = torch.optim.Adam(params=[model.w_in, model.w_out, model.w_rnn, model.b_rnn, model.b_out], lr=0.001) 
       
    model_performance = {'accuracy': [], 'loss': [], 'perf_loss': [], 'spike_loss': [], \
            'weight_loss': [], 'iteration': [], 'hidden_act':[], 'm':[], 'odor_sample':[], 'match': [], 'test': [], 'neural_input': [], 'output': []}

    if par['mode'] == 'test':
        model = torch.load(os.path.abspath(os.path.dirname(__file__)) + '/savedir/trian/' +  save_name + '.pth')
        trial_info1 = stim.generate_trial(set_rule = None)
        x = trial_info1['neural_input']
        t = trial_info1['desired_output'] 
        m = trial_info1['train_mask']
        odor_sample = trial_info1['sample']
        match = trial_info1['match']
        test = trial_info1['test']
        delay = trial_info1['delay']
        output, h, theta = model.run(torch.Tensor(x).to(device), torch.Tensor(t).to(device), torch.Tensor(m).to(device))
        # odor_sample = trial_info['sample']
        target_max = np.argmax(t, axis=2)
        output_max = np.argmax(output.detach().cpu().numpy(), axis=2)
        accuracy = np.sum(np.float32(target_max == output_max))/len(target_max.reshape(-1,1))
        n=(par['dead_time']+par['fix_time'])//par['dt']
        n1=(par['dead_time']+par['fix_time']+par['sample_time'])//par['dt']
        class_acc1 = np.sum(np.float32(target_max[n:n1,:] == output_max[n:n1,:]))/len(target_max[n:n1,:].reshape(-1,1))
        decision_acc_test = 0
        for batch in range(delay.shape[0]):
            n=(par['dead_time']+par['fix_time']+par['sample_time'])//par['dt']
            n1 = (par['dead_time']+par['fix_time']+par['sample_time']+par['test_time'])//par['dt']
            decision_acc_test += np.sum(np.float32(target_max[n+delay[batch]:n1+delay[batch],batch] == output_max[n+delay[batch]:n1+delay[batch],batch]))/len(target_max[n+delay[batch]:n1+delay[batch],batch].reshape(-1,1))
            print(delay[batch], np.float32(output_max[:,batch]))
        decision_acc_test = decision_acc_test/delay.shape[0]
        print_results(1, 0, 0, 0, h, accuracy, class_acc1, 0)
        model_performance = append_model_performance(model_performance, decision_acc_test, 0, 0, 0, 0, 1, h.detach().cpu().numpy(), model.m.detach().cpu().numpy(),odor_sample, match, test, x, output.detach().cpu().numpy())
        save_results(model_performance, model.w_rnn.detach().cpu().numpy(), save_fn='test/max3000_3s_odor2.pkl')
        sys.exit()
    elif par['mode'] == "train":
        pass
    else:
        print('Invalid pattern.')
        sys.exit()
    

    
    for i in range(par['num_iterations']):

        trial_info1 = stim.generate_trial(set_rule = None)
        x = trial_info1['neural_input']
        t = trial_info1['desired_output'] 
        m = trial_info1['train_mask']
        odor_sample = trial_info1['sample']
        match = trial_info1['match']
        test = trial_info1['test']
        delay = trial_info1['delay']
        output, h, theta = model.run(torch.Tensor(x).to(device), torch.Tensor(t).to(device), torch.Tensor(m).to(device))
        loss, perf_loss, spike_loss, weight_loss = model.optimize(optimizer)
        target_max = np.argmax(t, axis=2)
        output_max = np.argmax(output.detach().cpu().numpy(), axis=2)
        accuracy = np.sum(np.float32(target_max == output_max))/len(target_max.reshape(-1,1))
        n=(par['dead_time']+par['fix_time'])//par['dt']
        n1=(par['dead_time']+par['fix_time']+par['sample_time'])//par['dt']
        class_acc1 = np.sum(np.float32(target_max[n:n1,:] == output_max[n:n1,:]))/len(target_max[n:n1,:].reshape(-1,1))
        if par['random']:
            decision_acc_test = 0
            for batch in range(delay.shape[0]):
                n=(par['dead_time']+par['fix_time']+par['sample_time'])//par['dt']
                n1 = (par['dead_time']+par['fix_time']+par['sample_time']+par['test_time'])//par['dt']
                true_sample = target_max[n+delay[batch]:n1+delay[batch],batch] == output_max[n+delay[batch]:n1+delay[batch],batch]
                sum_sample = target_max[n+delay[batch]:n1+delay[batch],batch].reshape(-1,1)
                decision_acc_test += np.sum(np.float32(true_sample))/len(sum_sample)
            decision_acc_test = decision_acc_test/delay.shape[0]
        else:
            n=(par['dead_time']+par['fix_time']+par['sample_time'])//par['dt']
            n1 = (par['dead_time']+par['fix_time']+par['sample_time']+par['test_time'])//par['dt']
            decision_acc_test = np.sum(np.float32(target_max[n+delay[0]:n1+delay[0],:] == output_max[n+delay[0]:n1+delay[0],:]))/len(target_max[n+delay[0]:n1+delay[0],:].reshape(-1,1))
        # Save the network model and output model performance to screen
        if (i + 1) % par['iters_between_outputs']==0:
            print_results(i, perf_loss, spike_loss, weight_loss, h, accuracy, class_acc1, decision_acc_test)
            model_performance = append_model_performance(model_performance, decision_acc_test, loss.detach().cpu().numpy(), perf_loss.detach().cpu().numpy(), spike_loss.detach().cpu().numpy(), weight_loss.detach().cpu().numpy(), i, h.detach().cpu().numpy(), model.m.detach().cpu().numpy(),odor_sample, match, test, x, output.detach().cpu().numpy())
        if i == par['num_iterations']-1:
            print(model.m)
            print(torch.mean(h**2))
    #Save model and results
    save_results(model_performance, model.w_rnn.detach().cpu().numpy(), save_fn='train/' + save_name + '.pkl')
    torch.save(model, os.path.abspath(os.path.dirname(__file__)) + '/savedir/train/' +  save_name + '.pth')

def normalization1(data):
    data = data - np.mean(data, axis=1)[:,np.newaxis]
    data = data/(np.max(np.abs(data), axis=1)[:,np.newaxis]+0.00001)
    return data

def relu(inX):
    return np.maximum(0,inX)

def save_results(model_performance, weights, save_fn = None):

    results = {'weights': weights, 'parameters': par}
    for k,v in model_performance.items():
        results[k] = v
    fn = '/savedir/' + save_fn
    fn = os.path.abspath(os.path.dirname(__file__)) + fn
    with open(fn, 'wb') as f:
        pickle.dump(results, f)
    print('Model results saved in ',fn)

def append_model_performance(model_performance, accuracy, loss, perf_loss, spike_loss, weight_loss, iteration, h, m, odor_sample, match,test, neural_input, output):

    model_performance['accuracy'].append(accuracy)
    model_performance['loss'].append(loss)
    model_performance['perf_loss'].append(perf_loss)
    model_performance['spike_loss'].append(spike_loss)
    model_performance['weight_loss'].append(weight_loss)
    model_performance['iteration'].append(iteration)
    model_performance['hidden_act'].append(h)
    model_performance['m'].append(m)
    model_performance['odor_sample'].append(odor_sample)
    model_performance['match'].append(match)
    model_performance['test'].append(test)
    model_performance['neural_input'].append(neural_input)
    model_performance['output'].append(output)
    return model_performance

def print_results(iter_num, perf_loss, spike_loss, weight_loss, h, accuracy, class_acc1, decision_acc3):

    print(par['trial_type'] + ' Iter. {:4d}'.format(iter_num) + ' | Accuracy {:0.4f}'.format(accuracy) +
        ' |Class Accuracy {:0.4f}'.format(class_acc1) + ' |Dicision Accuracy {:0.4f}'.format(decision_acc3) +
      ' | Perf loss {:0.4f}'.format(perf_loss) + ' | Spike loss {:0.4f}'.format(spike_loss) +
      ' | Weight loss {:0.4f}'.format(weight_loss) + ' | Mean activity {:0.4f}'.format(torch.mean(h)))

def make_folders():
    dirs = os.path.abspath(os.path.dirname(__file__)) + '/savedir/train'
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    dirs = os.path.abspath(os.path.dirname(__file__)) + '/savedir/test'
    if not os.path.exists(dirs):
        os.makedirs(dirs)

def set_seed():
    np.random.seed(par['seed'])
    torch.manual_seed(par['seed'])
    torch.cuda.manual_seed(par['seed'])

if __name__ == '__main__':
    time_start = time.time()
    make_folders()
    set_seed()
    name = '{}'.format(args.exp)
    main(name)
    time_end = time.time()
    time_sum = (time_end - time_start)/3600 #hours
    print(time_sum,"h")