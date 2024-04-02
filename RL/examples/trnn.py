import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from typing import List, Tuple, Optional, overload
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class TRNN(nn.Module):
    def __init__(self,input_sz,hidden_sz,config):
        super().__init__()
        self.input_size=input_sz
        self.hidden_size=hidden_sz
        # self.w_in = nn.Parameter(torch.Tensor(input_sz,hidden_sz))
        # self.b_in = nn.Parameter(torch.Tensor(hidden_sz))

        # self.w_rnn = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        # self.b_rnn = nn.Parameter(torch.Tensor(hidden_sz))

        # self.w_out = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        # self.b_out = nn.Parameter(torch.Tensor(hidden_sz))
        self.cfg = config
        self.alpha = config.alpha
        self.gate = config.gate
        self.linear = config.linear
        self.region = config.region
        self.no_history = config.no_history
        stdv = 1.0 / math.sqrt(self.hidden_size)
        if self.region:
            self.w_in = self.initialize([input_sz, hidden_sz], 1, shape=0.2, scale=1.)
            self.b_in = nn.Parameter(torch.Tensor(hidden_sz))
            self.b_in.data.uniform_(-stdv, stdv)

            exc_inh_prop = 0.8
            num_exc_units = int(np.round(hidden_sz*exc_inh_prop))    
            num_inh_units = hidden_sz - num_exc_units
            EI_list = np.ones(hidden_sz)
            inh_index = np.random.choice(hidden_sz, num_inh_units, replace=False)
            EI_list[inh_index] = -1
            EI_matrix = np.diag(EI_list)
            ind_inh = np.where(EI_list==-1)[0]
            w_rnn0 = self.make_conn(np.linspace(-np.pi, np.pi, hidden_sz), 0.8)  # 用于高斯分布距离的初始化
            self.w_rnn = self.regions_conn_2(w_rnn0, hidden_sz//3, hidden_sz//3, hidden_sz//3)
            self.w_rnn[:, ind_inh] = self.initialize([hidden_sz, num_inh_units], 0.8, shape=0.2, scale=1.)
            self.w_rnn[ind_inh, :] = self.initialize([num_inh_units, hidden_sz], 0.8, shape=0.2, scale=1.)
            self.w_rnn = self.w_rnn / 3
            self.w_rnn = self.w_rnn @ EI_matrix
            self.b_rnn = nn.Parameter(torch.Tensor(hidden_sz))
            self.b_rnn.data.uniform_(-stdv, stdv)

            self.w_out = self.initialize([hidden_sz, hidden_sz], 1, shape=0.2)
            self.b_out = nn.Parameter(torch.Tensor(hidden_sz)) 
            self.b_out.data.uniform_(-stdv, stdv)  
            
            self.w_in_mask = np.ones_like(self.w_in)
            self.w_rnn_mask = np.ones_like(self.w_rnn)
            self.w_out_mask = np.ones_like(self.w_out)

            self.w_in_mask[:,hidden_sz//3:] = 0
            self.w_out_mask[:2*hidden_sz//3, :] = 0
            self.w_rnn_mask = self.w_rnn_mask - np.eye(self.w_rnn_mask.shape[0])

            self.w_in = nn.Parameter(torch.from_numpy(self.w_in * self.w_in_mask).float() / 10)
            self.w_rnn = nn.Parameter(torch.from_numpy(self.w_rnn * self.w_rnn_mask).float() / 10)
            self.w_out = nn.Parameter(torch.from_numpy(self.w_out * self.w_out_mask).float() / 10)

            self.w_in_mask = torch.from_numpy(self.w_in_mask)
            self.w_out_mask = torch.from_numpy(self.w_out_mask)
            self.w_rnn_mask = torch.from_numpy(self.w_rnn_mask)
        else:
            self.w_in = nn.Parameter(torch.Tensor(input_sz,hidden_sz))
            self.b_in = nn.Parameter(torch.Tensor(hidden_sz))

            self.w_rnn = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
            self.b_rnn = nn.Parameter(torch.Tensor(hidden_sz))

            self.w_out = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
            self.b_out = nn.Parameter(torch.Tensor(hidden_sz))

            self.init_weights()

        if self.gate:
            self.m = nn.Parameter(torch.ones(hidden_sz) * config.m)

        self.dt = 100
        self.tau_v = 200
        self.dt_tau_v = config.v #self.dt/self.tau_v

    def regions_conn_2(self, w_rnn, sensor, association, motor, sparse=[1.0, 0.9, 0.8]):   # sensor - association - motor

        # in-region （Gaussian）
        all_dims = len(w_rnn)
        assert all_dims == sensor + association + motor
        region1 = np.linspace(-np.pi, np.pi, sensor)
        region2 = np.linspace(-np.pi, np.pi, association)
        region3 = np.linspace(-np.pi, np.pi, motor)
        
        b = sensor 
        c = b + association

        w_rnn[:sensor,:sensor] = self.make_conn(region1, sparse[1], A=0.2) # 0.3
        w_rnn[b:b+association, b:b+association] = self.make_conn(region2, sparse[0], A=0.1)
        w_rnn[c:, c:] = self.make_conn(region3, sparse[0], A=0.1)
        
        # cross-region （Gaussian）
        w_rnn[:sensor, sensor:c] = self.initialize([sensor, association], sparse[2])
        w_rnn[:sensor, c:] = self.initialize([sensor, motor], sparse[2])

        w_rnn[b:c, :sensor] = self.initialize([association, sensor], sparse[2])
        w_rnn[b:c, c:] = self.initialize([association, motor], sparse[2])

        w_rnn[c:, :sensor] = self.initialize([motor, sensor], sparse[2])
        w_rnn[c:, b:c] = self.initialize([motor, association], sparse[2])

        return w_rnn

    def dist(self, d):
        d = np.remainder(d, 2*np.pi)
        d = np.where(d > 0.5 * 2*np.pi, d - 2*np.pi, d)
        return d

    def make_conn(self, x, connection_prob, A=1, std=0.1):
        assert np.ndim(x) == 1
        x_left = np.reshape(x, (-1, 1))
        x_right = np.repeat(x.reshape((1, -1)), len(x), axis=0)
        d = self.dist(x_left - x_right)

        Wxx = A * np.exp(-0.5 * np.square(d /std)) / (np.sqrt(2 * np.pi) * std)
        
        Wxx *= (np.random.rand(Wxx.shape[0], Wxx.shape[1]) < connection_prob)
        return  np.float32(Wxx)

    def initialize(self, dims, connection_prob, shape=0.1, scale=1.0):
        
        w = np.random.gamma(shape, scale, size=dims)
        
        w *= (np.random.rand(*dims) < connection_prob)

        return np.float32(w)

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
            weight.data *= self.cfg.w

    def permute_hidden(self,  # type: ignore[override]
                       hx: Tuple[Tensor, Tensor],
                       permutation: Optional[Tensor]
                       ) -> Tuple[Tensor, Tensor]:
        if permutation is None:
            return hx
        return self._apply_permutation(hx[0], permutation), self._apply_permutation(hx[1], permutation)
    
    def _apply_permutation(self, tensor: Tensor, permutation: Tensor, dim: int = 1) -> Tensor:
        return tensor.index_select(dim, permutation)

    def forward(self,orig_input,recurrent_cell):
        if isinstance(orig_input, nn.utils.rnn.PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = orig_input
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
            input, seq_lengths = pad_packed_sequence(orig_input)
            # recurrent_cell = self.permute_hidden(recurrent_cell, sorted_indices)
        else:
            input = orig_input

        seq_sz,bs,_=input.size()
        output_seq=[]
        self.hidden_state = []
        h = recurrent_cell[0]
        theta = recurrent_cell[1]

        if self.linear:
            for t in range(seq_sz):
                x_t = input[t, :, :]
                h = F.relu((x_t @ self.w_in + self.b_in))
                h = h.unsqueeze(0)
                output_seq.append(h)
            output_seq = torch.cat(output_seq, dim=0)
        else:
            for t in range(seq_sz):
                x_t = input[t, :, :]
                
                if self.no_history:
                    h = F.relu(x_t @ F.relu(self.w_in) + self.b_in)
                    h = h.unsqueeze(0)
                elif self.gate:
                    h = F.relu((1 - self.alpha) * h  + self.alpha * (x_t @ F.relu(self.w_in) + h @ self.w_rnn + self.b_rnn) - theta)
                    theta = (1-self.dt_tau_v) * theta + F.relu(self.m) * self.dt_tau_v * h
                else:
                    h = F.relu((1 - self.alpha) * h  + self.alpha * (x_t @ F.relu(self.w_in) + h @ self.w_rnn + self.b_rnn))
                # h = torch.clamp(h, 0, 100)
                self.hidden_state.append(h)
                y = F.relu(h @ self.w_out + self.b_out)

                output_seq.append(y)
            self.hidden_state = torch.cat(self.hidden_state, dim=0)
            output_seq = torch.cat(output_seq, dim=0)
            recurrent_cell = (h, theta)

        if isinstance(orig_input, nn.utils.rnn.PackedSequence):
            output_seq = pack_padded_sequence(output_seq, seq_lengths, enforce_sorted=False)
            # recurrent_cell = self.permute_hidden(recurrent_cell, unsorted_indices)
        return output_seq, recurrent_cell
    
class GTRNN(nn.Module):
    def __init__(self,input_sz,hidden_sz,config):
        super().__init__()
        self.input_size=input_sz
        self.hidden_size=hidden_sz
        # self.w_in = nn.Parameter(torch.Tensor(input_sz,hidden_sz))
        # self.b_in = nn.Parameter(torch.Tensor(hidden_sz))

        # self.w_rnn = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        # self.b_rnn = nn.Parameter(torch.Tensor(hidden_sz))

        # self.w_out = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
        # self.b_out = nn.Parameter(torch.Tensor(hidden_sz))
        self.cfg = config
        self.alpha = config.alpha
        self.gate = config.gate
        self.linear = config.linear
        self.region = config.region
        self.sig_m = config.sig_m
        self.no_history = config.no_history
        stdv = 1.0 / math.sqrt(self.hidden_size)
        if self.region:
            self.w_in = self.initialize([input_sz, hidden_sz], 1, shape=0.2, scale=1.)
            self.b_in = nn.Parameter(torch.Tensor(hidden_sz))
            self.b_in.data.uniform_(-stdv, stdv)

            exc_inh_prop = 0.8
            num_exc_units = int(np.round(hidden_sz*exc_inh_prop))    
            num_inh_units = hidden_sz - num_exc_units
            EI_list = np.ones(hidden_sz)
            inh_index = np.random.choice(hidden_sz, num_inh_units, replace=False)
            EI_list[inh_index] = -1
            EI_matrix = np.diag(EI_list)
            ind_inh = np.where(EI_list==-1)[0]
            w_rnn0 = self.make_conn(np.linspace(-np.pi, np.pi, hidden_sz), 0.8)  # 用于高斯分布距离的初始化
            self.w_rnn = self.regions_conn_2(w_rnn0, hidden_sz//3, hidden_sz//3, hidden_sz//3)
            self.w_rnn[:, ind_inh] = self.initialize([hidden_sz, num_inh_units], 0.8, shape=0.2, scale=1.)
            self.w_rnn[ind_inh, :] = self.initialize([num_inh_units, hidden_sz], 0.8, shape=0.2, scale=1.)
            self.w_rnn = self.w_rnn / 3
            self.w_rnn = self.w_rnn @ EI_matrix
            self.b_rnn = nn.Parameter(torch.Tensor(hidden_sz))
            self.b_rnn.data.uniform_(-stdv, stdv)

            self.w_out = self.initialize([hidden_sz, hidden_sz], 1, shape=0.2)
            self.b_out = nn.Parameter(torch.Tensor(hidden_sz)) 
            self.b_out.data.uniform_(-stdv, stdv)  
            
            self.w_in_mask = np.ones_like(self.w_in)
            self.w_rnn_mask = np.ones_like(self.w_rnn)
            self.w_out_mask = np.ones_like(self.w_out)

            self.w_in_mask[:,hidden_sz//3:] = 0
            self.w_out_mask[:2*hidden_sz//3, :] = 0
            self.w_out_mask[ind_inh, :] = 0
            self.w_rnn_mask = self.w_rnn_mask - np.eye(self.w_rnn_mask.shape[0])

            self.w_in = nn.Parameter(torch.from_numpy(self.w_in * self.w_in_mask).float() / 10)
            self.w_rnn = nn.Parameter(torch.from_numpy(self.w_rnn * self.w_rnn_mask).float() / 10)
            self.w_out = nn.Parameter(torch.from_numpy(self.w_out * self.w_out_mask).float() / 10)

            self.w_in_mask = torch.from_numpy(self.w_in_mask)
            self.w_out_mask = torch.from_numpy(self.w_out_mask)
            self.w_rnn_mask = torch.from_numpy(self.w_rnn_mask)

            self.w_m = nn.Parameter(torch.Tensor(input_sz,1))
            self.u_m = nn.Parameter(torch.Tensor(hidden_sz,1))
            if "Watermaze2d" in self.cfg.experiment:
                nn.init.xavier_normal_(self.w_m)
                nn.init.xavier_normal_(self.u_m)
            else:
                nn.init.uniform_(self.w_m, a=-stdv, b=stdv)
                nn.init.uniform_(self.u_m, a=-stdv, b=stdv)

            self.w_ga = nn.Parameter(torch.Tensor(input_sz,1))
            self.u_ga = nn.Parameter(torch.Tensor(hidden_sz,1))
            if "Watermaze2d" in self.cfg.experiment:
                nn.init.xavier_normal_(self.w_ga)
                nn.init.xavier_normal_(self.u_ga)
            else:
                nn.init.uniform_(self.w_ga, a=-stdv, b=stdv)
                nn.init.uniform_(self.u_ga, a=-stdv, b=stdv)
            
            self.w_h = nn.Parameter(torch.Tensor(input_sz,hidden_sz))
            if "Watermaze2d" in self.cfg.experiment:
                nn.init.xavier_normal_(self.w_h)
            else:
                nn.init.uniform_(self.w_h, a=-stdv, b=stdv)
            
            self.gamma = config.v_gamma
            self.m = config.v_m
        else:
            self.w_in = nn.Parameter(torch.Tensor(input_sz,hidden_sz))
            self.b_in = nn.Parameter(torch.Tensor(hidden_sz))

            self.w_rnn = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
            self.b_rnn = nn.Parameter(torch.Tensor(hidden_sz))

            self.w_out = nn.Parameter(torch.Tensor(hidden_sz, hidden_sz))
            self.b_out = nn.Parameter(torch.Tensor(hidden_sz))

            self.init_weights()

        self.dt = 100
        self.tau_v = 200
        self.dt_tau_v = config.v #self.dt/self.tau_v

    def regions_conn_2(self, w_rnn, sensor, association, motor, sparse=[1.0, 0.9, 0.8]):   # sensor - association - motor

        # in-region （Gaussian）
        all_dims = len(w_rnn)
        assert all_dims == sensor + association + motor
        region1 = np.linspace(-np.pi, np.pi, sensor)
        region2 = np.linspace(-np.pi, np.pi, association)
        region3 = np.linspace(-np.pi, np.pi, motor)
        
        b = sensor 
        c = b + association

        w_rnn[:sensor,:sensor] = self.make_conn(region1, sparse[1], A=0.2) # 0.3
        w_rnn[b:b+association, b:b+association] = self.make_conn(region2, sparse[0], A=0.1)
        w_rnn[c:, c:] = self.make_conn(region3, sparse[0], A=0.1)
        
        # cross-region （Gaussian）
        w_rnn[:sensor, sensor:c] = self.initialize([sensor, association], sparse[2])
        w_rnn[:sensor, c:] = self.initialize([sensor, motor], sparse[2])

        w_rnn[b:c, :sensor] = self.initialize([association, sensor], sparse[2])
        w_rnn[b:c, c:] = self.initialize([association, motor], sparse[2])

        w_rnn[c:, :sensor] = self.initialize([motor, sensor], sparse[2])
        w_rnn[c:, b:c] = self.initialize([motor, association], sparse[2])

        return w_rnn

    def dist(self, d):
        d = np.remainder(d, 2*np.pi)
        d = np.where(d > 0.5 * 2*np.pi, d - 2*np.pi, d)
        return d

    def make_conn(self, x, connection_prob, A=1, std=0.1):
        assert np.ndim(x) == 1
        x_left = np.reshape(x, (-1, 1))
        x_right = np.repeat(x.reshape((1, -1)), len(x), axis=0)
        d = self.dist(x_left - x_right)

        Wxx = A * np.exp(-0.5 * np.square(d /std)) / (np.sqrt(2 * np.pi) * std)
        
        Wxx *= (np.random.rand(Wxx.shape[0], Wxx.shape[1]) < connection_prob)
        return  np.float32(Wxx)

    def initialize(self, dims, connection_prob, shape=0.1, scale=1.0):
        
        w = np.random.gamma(shape, scale, size=dims)
        
        w *= (np.random.rand(*dims) < connection_prob)

        return np.float32(w)

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)
            weight.data *= self.cfg.w

    def permute_hidden(self,  # type: ignore[override]
                       hx: Tuple[Tensor, Tensor],
                       permutation: Optional[Tensor]
                       ) -> Tuple[Tensor, Tensor]:
        if permutation is None:
            return hx
        return self._apply_permutation(hx[0], permutation), self._apply_permutation(hx[1], permutation)
    
    def _apply_permutation(self, tensor: Tensor, permutation: Tensor, dim: int = 1) -> Tensor:
        return tensor.index_select(dim, permutation)
    
    def sin_pos(self, x):
        y = torch.sin(x)
        y = torch.where(x>=0., x, torch.tensor(0.))
        return y
        
    def forward(self,orig_input,recurrent_cell):
        if isinstance(orig_input, nn.utils.rnn.PackedSequence):
            input, batch_sizes, sorted_indices, unsorted_indices = orig_input
            max_batch_size = batch_sizes[0]
            max_batch_size = int(max_batch_size)
            input, seq_lengths = pad_packed_sequence(orig_input)
            # recurrent_cell = self.permute_hidden(recurrent_cell, sorted_indices)
        else:
            input = orig_input

        seq_sz,bs,_=input.size()
        output_seq=[]
        self.hidden_state = []
        h = recurrent_cell[0]
        theta = recurrent_cell[1]

        if self.linear:
            for t in range(seq_sz):
                x_t = input[t, :, :]
                h = F.relu((x_t @ self.w_in + self.b_in))
                h = h.unsqueeze(0)
                output_seq.append(h)
            output_seq = torch.cat(output_seq, dim=0)
        else:
            for t in range(seq_sz):
                x_t = input[t, :, :]
                
                if self.no_history and not self.region:
                    h = F.relu(x_t @ self.w_in + self.b_in)
                    h = h.unsqueeze(0)
                elif self.gate:
                    #TODO - vision 1
                    # m = F.relu(x_t @ self.w_m + h @ self.u_m + self.b_m)
                    # gamma = F.relu(x_t @ self.w_ga + h @ self.u_ga + self.b_ga)
                    # h = (1 - self.alpha) * h  + self.alpha * F.relu(x_t @ F.relu(self.w_in) + h @ self.w_rnn + self.b_rnn - gamma * theta)
                    # theta = (1-self.dt_tau_v) * theta + self.dt_tau_v * F.relu(m * h + x_t @ self.w_h)
                    #TODO - vision 2
                    # self.m = F.relu(x_t @ self.w_m + h @ self.u_m)
                    # self.gamma = F.relu(x_t @ self.w_ga + h @ self.u_ga)
                    # print(self.m.max().item(), self.gamma.max().item())
                    # h = (1 - self.alpha) * h + self.alpha * F.relu(x_t @ self.w_in + h @ self.w_rnn + self.b_rnn - self.gamma * theta)
                    # theta = (1 - self.dt_tau_v) * theta + self.dt_tau_v * F.relu(self.m * h)
                    #TODO - vision 3
                    # self.m = F.relu(x_t @ self.w_m + h @ self.u_m)
                    # self.gamma = F.relu(x_t @ self.w_ga + h @ self.u_ga)
                    # h = (1 - self.alpha) * h + self.alpha * F.relu(x_t @ self.w_in + h @ self.w_rnn + self.b_rnn - self.gamma * theta)
                    # theta = (1 - self.dt_tau_v) * theta + self.dt_tau_v * F.relu(self.m * h + x_t @ self.w_h)
                    #TODO - vision 4
                    h = (1 - self.alpha) * h + self.alpha * F.relu(x_t @ self.w_in + h @ self.w_rnn + self.b_rnn - self.gamma * theta)
                    theta = (1 - self.dt_tau_v) * theta + self.dt_tau_v * F.relu(self.m * h)
                else:
                    h = (1 - self.alpha) * h + self.alpha * F.relu(x_t @ self.w_in + h @ self.w_rnn + self.b_rnn)
                # h = torch.clamp(h, 0, 100)
                self.hidden_state.append(h)
                y = F.relu(h @ self.w_out + self.b_out)

                output_seq.append(y)
            self.hidden_state = torch.cat(self.hidden_state, dim=0)
            output_seq = torch.cat(output_seq, dim=0)
            recurrent_cell = (h, theta)

        if isinstance(orig_input, nn.utils.rnn.PackedSequence):
            output_seq = pack_padded_sequence(output_seq, seq_lengths, enforce_sorted=False)
            # recurrent_cell = self.permute_hidden(recurrent_cell, unsorted_indices)
        return output_seq, recurrent_cell