import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import os
import pickle as pkl
import random
from typing import Optional

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, scheduler, epoch, args, printout=True):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.8 ** ((epoch - 3) // 1))}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout: print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def test_params_flop(model,x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    # model_params = 0
    # for parameter in model.parameters():
    #     model_params += parameter.numel()
    #     print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    from ptflops import get_model_complexity_info
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=False)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))



def update_path(cfg):
    data_path = os.environ.get("DATA_PATH")
    mlflow_tag = os.environ.get("MLFLOW_TAG")
    if data_path is not None:
        from omegaconf import open_dict

        with open_dict(cfg):
            cfg.data_root = data_path
            cfg.mlflow.tags.label = mlflow_tag
    return cfg


def seed_everything(seed: int) -> None:
    """
    Seed everything
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def generate_seed(n_seeds: int):
    return list(range(n_seeds))


def load_pkl(path):
    with open(path, "rb") as f:
        return pkl.load(f)


def save_pkl(data, path):
    with open(path, "wb") as f:
        pkl.dump(data, f)


class ResultSaver:
    def __init__(self, path, save=True, logger=None):
        self.path = path
        self.real = []
        self.pred = []
        self.x = []
        self.x_m = []
        self.save = save
        self.log = logger

    def update(self, true, pred, x, x_m):
        if self.save:
            self.real.append(true.detach().cpu().numpy())
            self.pred.append(pred.detach().cpu().numpy())
            self.x.append(x.detach().cpu().numpy())
            self.x_m.append(x_m.detach().cpu().numpy())

    def save_results(self):
        if self.save:
            self.real = np.concatenate(self.real, axis=0)
            self.pred = np.concatenate(self.pred, axis=0)
            self.x = np.concatenate(self.x, axis=0)
            self.x_m = np.concatenate(self.x_m, axis=0)
            np.savez_compressed(
                self.path,
                real=self.real,
                pred=self.pred,
            )


def cal_num_parmas(model):
    return sum(p.numel() for p in model.parameters())


def cal_num_trainable_parmas(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_FLOPs(flops: float, unit: Optional[str] = None):
    if unit is None:
        if flops < 1e3:
            unit = ""
        elif flops < 1e6:
            unit = "K"
        elif flops < 1e9:
            unit = "M"
        elif flops < 1e12:
            unit = "G"
        else:
            unit = "T"
    if unit == "":
        return f"{flops:.2f}"
    else:
        return f"{flops / 10**(3 * ('KMGT'.index(unit) + 1)):.2f}{unit}"


def computational_analytics(model, inputs):
    if type(inputs) is dict:
        new = {}
        for k, v in inputs.items():
            new[k] = v[0:1]
        inputs = new
    else:
        raise TypeError("inputs must be dict")
    from thop import profile

    macs, params = profile(model, inputs=(inputs,))
    return {
        "macs": f"{macs:.0f}",
        "params": f"{params:.0f}",
        "flops": f"{2 * macs:.0f}",
        "F-MACs": format_FLOPs(macs),
        "F-params": format_FLOPs(params),
        "F-flops": format_FLOPs(2 * macs),
    }
