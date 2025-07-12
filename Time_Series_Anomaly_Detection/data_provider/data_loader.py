import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from data_provider.uea import Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# veremi dataset
class Dataset_Veremi(Dataset):

    # define featutre set PosX,PosY,SpdX,SpdY,AclX,AclY,HedX,HedY
    SELECTED_FEATURES = [
        'PosX', 'PosY', 'SpdX', 'SpdY',
        'AclX', 'AclY', 'HedX', 'HedY'
    ]

    def __init__(self, args, root_path, file_name, flag=None):
        self.args = args
        self.root_path = root_path
        self.flag = flag
        self.file_name = file_name
        self.max_seq_len = args.seq_len

        print(f'Loading {flag} dataset...')
        self.feature_df, self.labels_df = self.load_data()
        print(f'Loaded {len(self.feature_df)} samples with {len(self.feature_df.columns)} features each')
        
        self.feature_df = self.feature_df.reset_index(drop=True)
        self.labels_df = self.labels_df.reset_index(drop=True)
        self.all_IDs = self.feature_df.index.unique()
        
        # Pre-process features
        self.feature_df = Normalizer().normalize(self.feature_df)
        
        # Set class names for classification
        self.class_names = np.unique(self.labels_df)
        print(f"Classes in {flag} set:", self.class_names)
        
        # Generate visualizations if this is the training set
        if flag == 'train':
            self.visualize_data()

    def load_data(self):
        file_path = os.path.join(self.root_path, self.file_name)
        df = pd.read_csv(file_path)
        
        # take sample of the data 1000
        #df = df.sample(n=100, random_state=42)
        
        # Select only the features we want instead of dropping
        features = df[self.SELECTED_FEATURES].copy()
        labels = df['Label'].copy()
        
        # Split data based on flag
        if self.flag == 'benchmark':
            return features, pd.DataFrame(labels, columns=['Label'])
            
        train_features, temp_features, train_labels, temp_labels = train_test_split(
            features, labels,
            test_size=0.3,
            random_state=42,
            stratify=labels
        )
        
        val_features, test_features, val_labels, test_labels = train_test_split(
            temp_features, temp_labels,
            test_size=0.5,
            random_state=42,
            stratify=temp_labels
        )
        
        if self.flag == 'train':
            return train_features, pd.DataFrame(train_labels, columns=['Label'])
        elif self.flag == 'val':
            return val_features, pd.DataFrame(val_labels, columns=['Label'])
        elif self.flag == 'test':
            return test_features, pd.DataFrame(test_labels, columns=['Label'])
        
    def visualize_data(self):
        """Generate various visualizations for the dataset"""
        # 1. Class Distribution
        plt.figure(figsize=(10, 6))
        sns.countplot(data=self.labels_df, x='label')
        plt.title('Class Distribution in Training Set')
        plt.xlabel('Label')
        plt.ylabel('Count')
        plt.savefig('class_distribution.png')
        plt.close()

        # 2. Feature Correlations
        plt.figure(figsize=(12, 10))
        sns.heatmap(self.feature_df.corr(), annot=True, cmap='coolwarm')
        plt.title('Feature Correlations')
        plt.savefig('feature_correlations.png')
        plt.close()

        # 3. Position Distribution
        plt.figure(figsize=(10, 10))
        scatter = plt.scatter(
            self.feature_df['posx'], 
            self.feature_df['posy'],
            c=self.labels_df['label'],
            alpha=0.5,
            cmap='viridis'
        )
        plt.colorbar(scatter)
        plt.title('Vehicle Positions by Class')
        plt.xlabel('Position X')
        plt.ylabel('Position Y')
        plt.savefig('position_distribution.png')
        plt.close()

        # 4. Speed Distribution
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.boxplot(data=self.feature_df[['spdx', 'spdy']])
        plt.title('Speed Distribution')
        plt.subplot(1, 2, 2)
        sns.boxplot(data=self.feature_df[['aclx', 'acly']])
        plt.title('Acceleration Distribution')
        plt.tight_layout()
        plt.savefig('speed_accel_distribution.png')
        plt.close()

    def __getitem__(self, ind):
        batch_x = self.feature_df.loc[self.all_IDs[ind]].values
        labels = self.labels_df.loc[self.all_IDs[ind]].values

        if len(batch_x.shape) == 1:
            batch_x = batch_x.reshape(-1, len(self.SELECTED_FEATURES))

        return torch.from_numpy(batch_x).float(), torch.tensor(labels).long()

    def __len__(self):
        return len(self.all_IDs)
    