import math
import random
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import numpy as np
import h5py

def set_randomness_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_default_device():
    device = 'cpu'
    if torch.backends.mps.is_available(): device = 'mps'
    if torch.cuda.is_available(): device = 'cuda'
    return device

# copied from huggingface
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_openai_lr(transformer_model): # https://arxiv.org/pdf/2001.08361
    num_params = sum(p.numel() for p in transformer_model.parameters())
    return 0.003239 - 0.0001395 * math.log(num_params)

def torch_nanstd(x, axis=0):
    num = torch.where(torch.isnan(x), torch.full_like(x, 0), torch.full_like(x, 1)).sum(axis=axis)
    value = torch.where(torch.isnan(x), torch.full_like(x, 0), x).sum(axis=axis)
    mean = value / num
    mean_broadcast = torch.repeat_interleave(mean.unsqueeze(axis), x.shape[axis], dim=axis)
    return torch.sqrt(torch.nansum(torch.square(mean_broadcast - x), axis=axis) / (num - 1))

def remove_outliers(X, n_sigma=4, normalize_positions=-1):
    # Expects T, B, H
    assert len(X.shape) == 3, "X must be T,B,H"
    #for b in range(X.shape[1]):
        #for col in range(X.shape[2]):
    data = X if normalize_positions == -1 else X[:normalize_positions]
    data_clean = data[:].clone()
    data_mean, data_std = torch.nanmean(data, axis=0), torch_nanstd(data, axis=0)
    cut_off = data_std * n_sigma
    lower, upper = data_mean - cut_off, data_mean + cut_off

    data_clean[torch.logical_or(data_clean > upper, data_clean < lower)] = np.nan
    data_mean, data_std = torch.nanmean(data_clean, axis=0), torch_nanstd(data_clean, axis=0)
    cut_off = data_std * n_sigma
    lower, upper = data_mean - cut_off, data_mean + cut_off

    X = torch.maximum(-torch.log(1+torch.abs(X)) + lower, X)
    X = torch.minimum(torch.log(1+torch.abs(X)) + upper, X)
    return X

class PriorDumpDataLoader(DataLoader):
    def __init__(self, filename, num_steps, batch_size, device):
        self.filename = filename
        self.num_steps = num_steps
        self.batch_size=batch_size
        with h5py.File(self.filename, "r") as f:
            self.num_datapoints_max = f['inputs'].shape[0]
            self.num_features = f['inputs'].shape[2]
            self.max_num_classes = f['max_num_classes'][0].item()
        self.device=device
        self.pointer=0
    def __iter__(self):
        with h5py.File(self.filename, "r") as f:
            for _ in range(self.num_steps):
                self.data = f
                end = self.pointer + self.batch_size
                x = torch.from_numpy(self.data['inputs'][:,self.pointer:end,:])
                y = torch.from_numpy(self.data['targets'][:,self.pointer:end])
                single_eval_pos = random.choices(range(2, x.shape[0]-1))[0]
                self.pointer += self.batch_size
                if self.pointer >= self.data['inputs'].shape[1]:
                    self.pointer = 0
                yield dict(x=x,y=y,target_y=y, single_eval_pos=single_eval_pos)
    def __len__(self):
        return self.num_steps

