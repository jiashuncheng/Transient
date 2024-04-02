import os
import sys
import torch
from torch import nn

from sample_factory.model.encoder import Encoder
from sample_factory.model.core import ModelCore
from sample_factory.model.model_utils import model_device
from sample_factory.utils.typing import Config, ObsSpace
from sample_factory.utils.utils import log
from examples.trnn import TRNN, GTRNN

class GeneralCore(ModelCore):
    def __init__(self, cfg, input_size):
        super().__init__(cfg)

        self.cfg = cfg
        self.is_gru = False
        self.is_rnn = False

        if cfg.rnn_type == "gru":
            self.core = nn.GRU(input_size, cfg.rnn_size, cfg.rnn_num_layers)
            self.is_gru = True
        elif cfg.rnn_type == "lstm":
            self.core = nn.LSTM(input_size, cfg.rnn_size, cfg.rnn_num_layers)
        elif cfg.rnn_type == "trnn":
            self.core = TRNN(input_size, cfg.rnn_size, cfg)
        elif cfg.rnn_type == "gtrnn":
            self.core = GTRNN(input_size, cfg.rnn_size, cfg)
        elif cfg.rnn_type == "rnn":
            self.core = nn.RNN(input_size, cfg.rnn_size, cfg.rnn_num_layers)
            self.is_rnn = True
        else:
            raise RuntimeError(f"Unknown RNN type {cfg.rnn_type}")

        self.core_output_size = cfg.rnn_size
        self.rnn_num_layers = cfg.rnn_num_layers

        self.hidden_output = []

    def forward(self, head_output, rnn_states):
        is_seq = not torch.is_tensor(head_output)
        if not is_seq:
            head_output = head_output.unsqueeze(0)

        if self.rnn_num_layers > 1:
            rnn_states = rnn_states.view(rnn_states.size(0), self.cfg.rnn_num_layers, -1)
            rnn_states = rnn_states.permute(1, 0, 2)
        else:
            rnn_states = rnn_states.unsqueeze(0)

        if self.is_gru or self.is_rnn:
            x, new_rnn_states = self.core(head_output, rnn_states.contiguous())
        else:
            h, c = torch.split(rnn_states, self.cfg.rnn_size, dim=2)
            x, (h, c) = self.core(head_output, (h.contiguous(), c.contiguous()))
            new_rnn_states = torch.cat((h, c), dim=2)

        if not is_seq:
            x = x.squeeze(0)

        if self.rnn_num_layers > 1:
            new_rnn_states = new_rnn_states.permute(1, 0, 2)
            new_rnn_states = new_rnn_states.reshape(new_rnn_states.size(0), -1)
        else:
            new_rnn_states = new_rnn_states.squeeze(0)
        
        if 'enjoy' in os.path.split(sys.argv[0])[1]:
            self.hidden_output.append(h)
        return x, new_rnn_states

def make_core(cfg: Config, obs_space: ObsSpace) -> ModelCore:
    return GeneralCore(cfg, obs_space)