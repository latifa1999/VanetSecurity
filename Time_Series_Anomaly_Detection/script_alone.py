#!/usr/bin/env python3

import os
import torch
import numpy as np
import pandas as pd
from argparse import Namespace
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix

# -----------------------------
# Minimal collate_fn & padding_mask
# -----------------------------
def collate_fn(data, max_len=None):
    features, labels = zip(*data)
    lens = [x.shape[0] for x in features]
    max_len = max_len or max(lens)
    X = torch.zeros(len(features), max_len, features[0].shape[-1])
    for i, f in enumerate(features):
        end = min(lens[i], max_len)
        X[i, :end, :] = f[:end, :]
    targets = torch.stack(labels, dim=0)
    pm = padding_mask(torch.tensor(lens, dtype=torch.int16), max_len)
    return X, targets, pm

def padding_mask(lengths, max_len=None):
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len).unsqueeze(0)
            < lengths.unsqueeze(1)).int()

# -----------------------------
# Minimal Normalizer
# -----------------------------
class Normalizer:
    def __init__(self, norm_type='standardization'):
        self.norm_type = norm_type
        self.mean = None
        self.std = None
        self.min_val = None
        self.max_val = None

    def normalize(self, df):
        eps = np.finfo(float).eps
        if self.norm_type == "standardization":
            if self.mean is None:
                self.mean = df.mean()
                self.std = df.std()
            return (df - self.mean) / (self.std + eps)
        elif self.norm_type == "minmax":

            if self.max_val is None:
                self.max_val = df.max()
                self.min_val = df.min()
            return (df - self.min_val) / (self.max_val - self.min_val + eps)
        return df

# -----------------------------
# Minimal Dataset_Veremi
# -----------------------------
class Dataset_Veremi(Dataset):
    def __init__(self, args, root_path, file_name='veremi_extension_benchmark.csv', flag=None):
        self.args = args
        self.root_path = root_path
        self.flag = flag
        self.file_name = file_name
        
        # Define the features we want to extract in order and make them lowercase
        self.feature_columns = ['posx', 'posy', 'spdx', 'spdy', 'aclx', 'acly', 'hedx', 'hedy']

        # self.feature_columns = ['posx', 'posy', 'spdx', 'spdy', 'aclx', 'acly', 'hedx', 'hedy']

        if self.flag == 'benchmark':
            fdf, ldf = self.load_benchmark_data()
        else:
            fdf, ldf = self.load_data()

        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(fdf).reset_index(drop=True)
        self.labels_df = ldf.reset_index(drop=True)
        self.all_IDs = self.feature_df.index.unique()

    def load_data(self):
        df = pd.read_csv(os.path.join(self.root_path, self.file_name))

        df.columns = df.columns.str.lower()
        
        # Extract the 8 features we need
        features = df[self.feature_columns]
        
        # Get labels
        labels = df['label'].values  # or whatever your label column is named

        X_tr, X_tmp, y_tr, y_tmp = train_test_split(
            features, labels, test_size=0.3, random_state=42, stratify=labels
        )
        X_v, X_te, y_v, y_te = train_test_split(
            X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp
        )
        
        if self.flag == 'train':
            return X_tr, pd.DataFrame(y_tr, columns=['label'])
        elif self.flag == 'val':
            return X_v, pd.DataFrame(y_v, columns=['label'])
        return X_te, pd.DataFrame(y_te, columns=['label'])

    def load_benchmark_data(self):
        bench = pd.read_csv(os.path.join(self.root_path, self.file_name))

        bench.columns = bench.columns.str.lower()
        
        # Extract the 8 features we need
        features = bench[self.feature_columns]
        
        # Get labels
        labels = bench['label'].values  # adjust column name if different
        
        return features, pd.DataFrame(labels, columns=['label'])

    def __getitem__(self, ind):
        bx = self.feature_df.loc[self.all_IDs[ind]].values
        ly = self.labels_df.loc[self.all_IDs[ind]].values
        if len(bx.shape) == 1:
            bx = bx.reshape(-1, self.feature_df.shape[1])
        return torch.from_numpy(bx).float(), torch.tensor(ly).long()

    def __len__(self):
        return len(self.all_IDs)

# -----------------------------
# Minimal data_provider
# -----------------------------
def data_provider(args, flag):
    ds = Dataset_Veremi(args, args.root_path, flag=flag)
    dl = DataLoader(ds, batch_size=1, shuffle=(flag not in ["test","benchmark"]),
                    collate_fn=lambda x: collate_fn(x, max_len=args.seq_len))
    return ds, dl

# -----------------------------
# Model & Utilities
# -----------------------------
def initialize_model(checkpoint_path, args):
    from exp.exp_classification import Exp_Classification
    exp = Exp_Classification(args)
    exp.model = exp._build_model()
    chk = torch.load(checkpoint_path, map_location=args.device)
    sd = chk.get('model_state_dict', chk)
    md = exp.model.state_dict()
    pdict = {k: v for k, v in sd.items() if k in md and md[k].shape == v.shape}
    md.update(pdict)
    exp.model.load_state_dict(md)
    exp.model.eval()
    return exp

def custom_test(exp, flag="benchmark"):
    ds, dl = data_provider(exp.args, flag=flag)
    preds, trues = [], []
    with torch.no_grad():
        for bx, lbl, pm in dl:
            bx, lbl, pm = bx.to(exp.args.device), lbl.to(exp.args.device), pm.to(exp.args.device)
            out = exp.model(bx.float(), pm.float(), None, None)
            preds.append(out.cpu())
            trues.append(lbl.cpu())

    preds = torch.cat(preds, dim=0)
    trues = torch.cat(trues, dim=0)
    probs = torch.softmax(preds, dim=1)
    pred_cls = torch.argmax(probs, dim=1).numpy()
    true_cls = trues.numpy()

    acc = accuracy_score(true_cls, pred_cls)
    f1 = f1_score(true_cls, pred_cls, average='weighted')
    print(f"\nSamples: {len(true_cls)} | Accuracy: {acc:.4f} | F1: {f1:.4f}")
    print(f"True->0:{(true_cls==0).sum()} 1:{(true_cls==1).sum()} | Pred->0:{(pred_cls==0).sum()} 1:{(pred_cls==1).sum()}")
    # print("true:", true_cls)
    print("true:", "[", " ".join(map(str, true_cls.flatten())), "]")
    print("pred:", pred_cls)
    cm = confusion_matrix(true_cls, pred_cls)
    print(cm)
    print(f"\nSamples: {len(true_cls)} | Accuracy: {acc:.4f} | F1: {f1:.4f}")

# -----------------------------
# Main
# -----------------------------
def main():
    args_dict = {
        "device": torch.device("cpu"), "use_multi_gpu": False, "use_gpu": False,
        "model": "Informer", "task_name": "classification", "data": "veremi",
        "model_id": "verClssSingle", "root_path": "/home/latifa.elbouga/lustre/vr_outsec-vh2sz1t4fks/users/latifa.elbouga/Data/",
        "data_path": "veremi_extension_benchmark.csv", "features": "M", "target": "OT", "freq": "h",
        "checkpoints": "./checkpoints/", "seq_len": 1, "label_len": 0, "pred_len": 0,
        "enc_in": 8, "dec_in": 8, "c_out": 2, "d_model": 512, "n_heads": 8,
        "e_layers": 2, "d_layers": 1, "d_ff": 2048, "moving_avg": 25, "factor": 1,
        "distil": True, "dropout": 0.1, "embed": "timeF", "activation": "gelu",
        "decomp_method": "moving_avg", "num_workers": 0, "itr": 1, "train_epochs": 7,
        "batch_size": 256, "patience": 5, "learning_rate": 0.0001, "des": "test",
        "loss": "MSE", "lradj": "type1", "use_amp": False, "p_hidden_dims": [128, 128],
        "p_hidden_layers": 2, "use_dtw": False, "top_k": 5, "num_class": 2
    }
    args = Namespace(**args_dict)
    cp = '/home/latifa.elbouga/AD/Time-Series-Library/checkpoints/Informer/checkpoints_informer_best.pth'
    exp = initialize_model(cp, args)
    exp.args = args
    custom_test(exp, flag="benchmark")

if __name__ == "__main__":
    main()