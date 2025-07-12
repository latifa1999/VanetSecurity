from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from exp.exp_basic import Exp_Basic, MetricsTracker
from utils.tools import EarlyStopping,cal_accuracy, f1_score
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from datetime import datetime
import time
from tqdm import tqdm
from torch import autocast, GradScaler
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, precision_score, recall_score
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from torch.utils.tensorboard import SummaryWriter
from exp.track import ResourceTracker

warnings.filterwarnings('ignore')

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # Weight for each class
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)
        self.scaler = GradScaler()


    def _build_model(self):
        train_data, _ = self._get_data(flag='train')
        test_data, _ = self._get_data(flag='test')
        
        # Set model parameters based on data
        self.args.seq_len = max(train_data.max_seq_len, test_data.max_seq_len)
        self.args.enc_in = train_data.feature_df.shape[1]
        self.args.num_class = len(train_data.class_names)
        
        # Initialize model
        model = self.model_dict[self.args.model].Model(self.args).float()
        if torch.cuda.is_available() and hasattr(self.args, 'device_ids'):
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model.to(self.device)
    
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    
    def train(self, setting):
        """Enhanced training procedure with comprehensive logging and automatic checkpoint resumption"""
        # Data loading
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        # Calculate class weights for balanced training
        class_counts = np.bincount([y.item() for _, y in train_data])
        total_samples = sum(class_counts)
        class_weights = torch.FloatTensor([
            total_samples / (len(class_counts) * count) for count in class_counts
        ]).to(self.device)
        
        # Setup checkpoint paths
        model_checkpoint_dir = os.path.join('checkpoints', self.args.model)
        os.makedirs(model_checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(model_checkpoint_dir, f'checkpoints_{self.args.model.lower()}.pth')
        best_model_path = os.path.join(model_checkpoint_dir, f'checkpoints_{self.args.model.lower()}_best.pth')

        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        
        # Initialize best validation loss and start epoch
        best_val_loss = float('inf')
        current_epoch = 0
        
        # Check if checkpoint exists and load it
        if os.path.exists(checkpoint_path):
            print(f"Found existing checkpoint: {checkpoint_path}")
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    # Get current model state dict
                    model_dict = self.model.state_dict()
                    
                    # Filter out parameters with mismatched sizes
                    pretrained_dict = {}
                    for k, v in checkpoint["model_state_dict"].items():
                        if k in model_dict:
                            if v.size() == model_dict[k].size():
                                pretrained_dict[k] = v
                            else:
                                print(f"Skipping parameter {k} due to size mismatch: checkpoint size {v.size()} vs model size {model_dict[k].size()}")
                    
                    # Check how many parameters were loaded
                    print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} parameters from checkpoint")
                    
                    # Update model state dict with filtered parameters
                    model_dict.update(pretrained_dict)
                    self.model.load_state_dict(model_dict)
                    
                    best_val_loss = checkpoint.get("best_val_loss", float('inf'))
                    current_epoch = checkpoint.get("epoch", 0) + 1  # Start from the next epoch
                    print(f"Resuming training from epoch {current_epoch}")
                    
                    # Initialize optimizer
                    optimizer = optim.AdamW(self.model.parameters(), 
                                        lr=self.args.learning_rate, 
                                        weight_decay=0.01)
                                        
                    # Load optimizer state if available
                    if "optimizer_state_dict" in checkpoint:
                        try:
                            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                            print("Optimizer state loaded")
                        except Exception as e:
                            print(f"Error loading optimizer state: {e}")
                            print("Initialized fresh optimizer")
                else:
                    # If checkpoint is just the model state dict
                    # Get current model state dict
                    model_dict = self.model.state_dict()
                    
                    # Filter out parameters with mismatched sizes
                    pretrained_dict = {}
                    for k, v in checkpoint.items():
                        if k in model_dict:
                            if v.size() == model_dict[k].size():
                                pretrained_dict[k] = v
                            else:
                                print(f"Skipping parameter {k} due to size mismatch: checkpoint size {v.size()} vs model size {model_dict[k].size()}")
                    
                    # Check how many parameters were loaded
                    print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} parameters from checkpoint")
                    
                    # Update model state dict with filtered parameters
                    model_dict.update(pretrained_dict)
                    self.model.load_state_dict(model_dict)
                    
                    print("Loaded compatible weights from checkpoint")
                    
                    # Initialize optimizer
                    optimizer = optim.AdamW(self.model.parameters(), 
                                        lr=self.args.learning_rate, 
                                        weight_decay=0.01)
            except Exception as e:
                print(f"Error loading checkpoint: {e}")
                print("Starting training from scratch")
                
                # Initialize optimizer
                optimizer = optim.AdamW(self.model.parameters(), 
                                    lr=self.args.learning_rate, 
                                    weight_decay=0.01)
        else:
            print("No checkpoint found. Starting training from scratch")
            
            # Initialize optimizer
            optimizer = optim.AdamW(self.model.parameters(), 
                                lr=self.args.learning_rate, 
                                weight_decay=0.01)
        
        # Configure scheduler based on remaining epochs
        remaining_epochs = max(1, self.args.train_epochs - current_epoch)  # Ensure at least 1 epoch
        print(f"Configuring scheduler for {remaining_epochs} remaining epochs")
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.args.learning_rate,
            epochs=remaining_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3
        )
        
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        print(f"\nTraining {self.args.model} model:")
        print(f"Starting from epoch {current_epoch+1}")
        print(f"Checkpoints will be saved to: {model_checkpoint_dir}")
        
        # Training loop - start from the current epoch
        for epoch in range(current_epoch, self.args.train_epochs):
            self.model.train()
            train_losses = []
            train_preds = []
            train_trues = []
            
            # Training phase
            train_bar = tqdm(train_loader, desc=f'Training Epoch {epoch + 1}/{self.args.train_epochs}')
            for batch_idx, (batch_x, batch_y, padding_mask) in enumerate(train_bar):
                optimizer.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                batch_y = batch_y[:, 0].to(self.device)
                
                with autocast(device_type='cuda', enabled=True):
                    outputs = self.model(batch_x, padding_mask, None, None)
                    loss = criterion(outputs, batch_y)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
                scheduler.step()
                
                train_losses.append(loss.item())
                train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                train_trues.extend(batch_y.cpu().numpy())
                
                # Update progress bar
                train_bar.set_postfix({
                    'loss': f'{np.mean(train_losses):.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.6f}'
                })
                
                # Log batch metrics
                self._log_batch_metrics('train', epoch, batch_idx, loss.item())
            
            # Calculate epoch metrics
            train_accuracy = accuracy_score(train_trues, train_preds)
            train_f1 = f1_score(train_trues, train_preds, average='macro')
            
            # Validation phase
            val_loss, val_accuracy, val_f1, cm = self.validate(vali_loader, criterion)
            
            # Test phase
            test_loss, test_accuracy, test_f1, cm = self.validate(test_loader, criterion)
            
            # Update metrics tracker
            self.metrics.update('train', epoch, np.mean(train_losses), train_accuracy, train_f1)
            self.metrics.update('val', epoch, val_loss, val_accuracy, val_f1)
            self.metrics.update('test', epoch, test_loss, test_accuracy, test_f1)
            
            # Generate and save confusion matrix
            self._plot_confusion_matrix(
                train_trues, 
                train_preds,
                f'Training Confusion Matrix - Epoch {epoch + 1}',
                f'confusion_matrix_epoch_{epoch + 1}.png'
            )
            
            # Print epoch summary
            print(f'\nEpoch {epoch + 1}/{self.args.train_epochs} Summary:')
            print(f'Train - Loss: {np.mean(train_losses):.4f}, Acc: {train_accuracy:.4f}, F1: {train_f1:.4f}')
            print(f'Val   - Loss: {val_loss:.4f}, Acc: {val_accuracy:.4f}, F1: {val_f1:.4f}')
            print(f'Test  - Loss: {test_loss:.4f}, Acc: {test_accuracy:.4f}, F1: {test_f1:.4f}')
            
            # Save current model checkpoint
            self._save_model(
                epoch,
                self.model.state_dict(),
                optimizer.state_dict(),
                val_loss,
                checkpoint_path
            )
            print(f"Saved checkpoint for epoch {epoch + 1} to: {checkpoint_path}")
            
            # Save best model if current validation loss is better
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_model(
                    epoch,
                    self.model.state_dict(),
                    optimizer.state_dict(),
                    val_loss,
                    best_model_path
                )
                print(f"New best model saved to: {best_model_path}")
            
            # Early stopping check
            early_stopping(val_loss, self.model, checkpoint_path)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
        
        # End of training
        self.metrics.plot_metrics(self.plot_dir)
        print(f"\nTraining completed. Final checkpoints saved in: {model_checkpoint_dir}")
        return self.model
        
    def load_trained_model(self, checkpoint_path):


        print(f"Loading model checkpoint from: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint if "model_state_dict" not in checkpoint else checkpoint["model_state_dict"]

        model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)

        self.model.load_state_dict(model_dict)

        # Put model in evaluation mode
        self.model.eval()
        print("Model loaded successfully and set to eval mode.")

    def finetune(self, setting, pretrained_path):
        """Enhanced fine-tuning procedure"""
        print(f'\nFine-tuning with pretrained model from: {pretrained_path}')
        
        # Load pretrained model
        checkpoint = self._load_model(pretrained_path)
        if checkpoint is None:
            return
            
        # Data loading
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        
        # Calculate class weights for balanced training
        class_counts = np.bincount([y.item() for _, y in train_data])
        total_samples = sum(class_counts)
        class_weights = torch.FloatTensor([
            total_samples / (len(class_counts) * count) for count in class_counts
        ]).to(self.device)
        
        # Training components with lower learning rate for fine-tuning
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate * 0.1,  # Lower learning rate for fine-tuning
            weight_decay=0.01
        )
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.args.learning_rate * 0.1,
            epochs=self.args.train_epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3
        )
        
        # Setup directories for fine-tuning
        finetune_log_dir = os.path.join('finetune', 
                                      f'{self.args.model}_finetuning_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        os.makedirs(finetune_log_dir, exist_ok=True)
        
        # Initialize metrics tracker for fine-tuning
        ft_metrics = MetricsTracker(finetune_log_dir)

        # Setup checkpoint paths
        model_checkpoint_dir = os.path.join('checkpoints', self.args.model)
        os.makedirs(model_checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(model_checkpoint_dir, f'checkpoints_finetuning{self.args.model.lower()}.pth')
        best_model_path = os.path.join(model_checkpoint_dir, f'checkpoints_finetuning{self.args.model.lower()}_best.pth')
        
        print(f"\nTraining {self.args.model} model:")
        print(f"Checkpoints will be saved to: {model_checkpoint_dir}")
        
        # Fine-tuning loop
        best_val_f1 = 0
        for epoch in range(self.args.train_epochs):
            self.model.train()
            train_losses = []
            train_preds = []
            train_trues = []
            
            # Training phase
            train_bar = tqdm(train_loader, desc=f'Fine-tuning Epoch {epoch + 1}')
            for batch_idx, (batch_x, batch_y, padding_mask) in enumerate(train_bar):
                optimizer.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                batch_y = batch_y[:, 0].to(self.device)
                
                with autocast(device_type='cuda', enabled=True):
                    outputs = self.model(batch_x, padding_mask, None, None)
                    loss = criterion(outputs, batch_y)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
                scheduler.step()
                
                train_losses.append(loss.item())
                train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                train_trues.extend(batch_y.cpu().numpy())
                
                train_bar.set_postfix({
                    'loss': f'{np.mean(train_losses):.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.6f}'
                })
            
            # Calculate metrics
            train_accuracy = accuracy_score(train_trues, train_preds)
            train_f1 = f1_score(train_trues, train_preds, average='macro')
            
            # Validation phase
            val_loss, val_accuracy, val_f1, cm = self.validate(vali_loader, criterion)
            test_loss, test_accuracy, test_f1, cm = self.validate(test_loader, criterion)
            
            # Update fine-tuning metrics
            ft_metrics.update('train', epoch, np.mean(train_losses), train_accuracy, train_f1)
            ft_metrics.update('val', epoch, val_loss, val_accuracy, val_f1)
            ft_metrics.update('test', epoch, test_loss, test_accuracy, test_f1)

            # Save current model
            self._save_model(
                epoch,
                self.model.state_dict(),
                optimizer.state_dict(),
                val_f1,
                checkpoint_path
            )
            print(f"Saved checkpoint for epoch {epoch + 1} to: {checkpoint_path}")
            
            # Save best model
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                self._save_model(
                    epoch,
                    self.model.state_dict(),
                    optimizer.state_dict(),
                    val_f1,
                    best_model_path
                )
                print(f"New best model saved to: {best_model_path}")
            
            # Print epoch summary
            print(f'\nFine-tuning Epoch {epoch + 1} Summary:')
            print(f'Train - Loss: {np.mean(train_losses):.4f}, Acc: {train_accuracy:.4f}, F1: {train_f1:.4f}')
            print(f'Val   - Loss: {val_loss:.4f}, Acc: {val_accuracy:.4f}, F1: {val_f1:.4f}')
            print(f'Test  - Loss: {test_loss:.4f}, Acc: {test_accuracy:.4f}, F1: {test_f1:.4f}')
        
        # End of fine-tuning
        ft_metrics.plot_metrics('Finetuning')
        print(f"\nFine-tuning completed. Final checkpoints saved in: {model_checkpoint_dir}")
        return self.model

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.CrossEntropyLoss()
        return criterion

    def validate(self, dataloader, criterion):
        """Validation/Test procedure"""
        self.model.eval()
        total_loss = []
        all_preds = []
        all_trues = []
        
        with torch.no_grad():
            for batch_x, batch_y, padding_mask in dataloader:
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                batch_y = batch_y[:, 0].to(self.device)
                
                outputs = self.model(batch_x, padding_mask, None, None)
                loss = criterion(outputs, batch_y)
                
                total_loss.append(loss.item())
                all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                all_trues.extend(batch_y.cpu().numpy())
        
        avg_loss = np.mean(total_loss)
        accuracy = accuracy_score(all_trues, all_preds)
        f1 = f1_score(all_trues, all_preds, average='macro')

        # confusion matrix
        cm = self._plot_confusion_matrix(
            all_trues, 
            all_preds,
            'Confusion Matrix',
            'confusion_matrix.png'
        )

        
        return avg_loss, accuracy, f1, cm 
    
    def _plot_confusion_matrix(self, y_true, y_pred, title, filename):
        """
        Plot and save confusion matrix
        
        Args:
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
            title (str): Title of the plot
            filename (str): Filename to save the plot
        """
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=np.unique(y_true), 
                    yticklabels=np.unique(y_true))
        plt.title(title)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        # Save the plot
        full_path = os.path.join(self.plot_dir, filename)
        plt.savefig(full_path)
        plt.close()
        
        return cm
    

    def test(self, setting, test=0):
        """Enhanced test procedure with explicit checkpoint handling"""
        # Define checkpoint paths
        model_checkpoint_dir = os.path.join('checkpoints', self.args.model)
        regular_checkpoint = os.path.join(model_checkpoint_dir, f'checkpoints_{self.args.model.lower()}.pth')
        best_checkpoint = os.path.join(model_checkpoint_dir, f'checkpoints_{self.args.model.lower()}_best.pth')
        
        if test:
            print('Loading model for testing...')
            # Try to load best model first, fall back to regular checkpoint if not available
            if os.path.exists(best_checkpoint):
                print(f"Using best model checkpoint: {best_checkpoint}")
                self._load_model(best_checkpoint)
            elif os.path.exists(regular_checkpoint):
                print(f"Best model not found, using latest checkpoint: {regular_checkpoint}")
                self._load_model(regular_checkpoint)
            else:
                print("No checkpoint found! Cannot proceed with testing.")
                return
        
        test_data, test_loader = self._get_data(flag='test')
        criterion = nn.CrossEntropyLoss()
        # Initialize resource tracker
        tracker = ResourceTracker()
        tracker.start()
        
        # Start inference with resource tracking
        self.model.eval()
        total_loss = []
        all_preds = []
        all_trues = []
        batch_times = []
        
        print("Starting test with resource tracking...")
        
        with torch.no_grad():
            for batch_idx, (batch_x, batch_y, padding_mask) in enumerate(tqdm(test_loader, desc="Testing")):
                # Record resources before batch
                tracker.update()
                
                # Record batch start time
                batch_start = time.time()
                
                # Process batch
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.float().to(self.device)
                batch_y = batch_y[:, 0].to(self.device)
                
                # Forward pass
                outputs = self.model(batch_x, padding_mask, None, None)
                loss = criterion(outputs, batch_y)
                
                # Record metrics
                total_loss.append(loss.item())
                all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                all_trues.extend(batch_y.cpu().numpy())
                
                # Record batch end time
                batch_end = time.time()
                batch_times.append(batch_end - batch_start)
                
                # Record resources after batch
                tracker.update()
                
                # Optional: Print progress with resource info
                if (batch_idx + 1) % 10 == 0:
                    current_summary = tracker.get_summary()
                    print(f"Batch {batch_idx+1}/{len(test_loader)}: "
                        f"RAM: {current_summary['ram_max_gb']:.2f}GB, "
                        f"GPU: {current_summary['gpu_max_gb']:.2f}GB")
        
        # Calculate test metrics
        test_loss = np.mean(total_loss)
        test_accuracy = accuracy_score(all_trues, all_preds)
        test_f1 = f1_score(all_trues, all_preds, average='weighted')
        precision = precision_score(all_trues, all_preds, average='weighted')
        recall = recall_score(all_trues, all_preds, average='weighted')


        print("shape of all_preds", np.array(all_preds).shape)
        print("shape of all_trues", np.array(all_trues).shape)
        # save all_preds and all_trues in 2 columns in a csv file
        import pandas as pd
        df = pd.DataFrame({'all_preds': all_preds, 'all_trues': all_trues})
        df.to_csv('all_preds_all_trues.csv', index=False)
        
        # Create confusion matrix
        cm = confusion_matrix(all_trues, all_preds)
        
        # Get resource usage summary
        resource_summary = tracker.get_summary()
        
        # Save plots and logs
        plot_path = tracker.plot_usage(self.args.model, "test")
        log_path = tracker.save_log(self.args.model, "test")
        
        # Save detailed test results with resource info
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_f1': test_f1,
            'avg_batch_time': np.mean(batch_times),
            'total_test_time': resource_summary['duration_seconds'],
            'max_ram_gb': resource_summary['ram_max_gb'],
            'mean_ram_gb': resource_summary['ram_mean_gb'],
            'max_gpu_gb': resource_summary['gpu_max_gb'],
            'mean_gpu_gb': resource_summary['gpu_mean_gb'],
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save results in the same directory as the model
        results_path = os.path.join('test_results_NST.txt')
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w') as f:
            f.write(f"Test Results for {self.args.model} on benchmark dataset\n")
            f.write(f"Timestamp: {results['timestamp']}\n\n")
            
            f.write("Performance Metrics:\n")
            f.write(f"Loss: {test_loss:.4f}\n")
            f.write(f"Accuracy: {test_accuracy:.4f}\n")
            f.write(f"F1 Score: {test_f1:.4f}\n\n")
            
            f.write("Resource Usage:\n")
            f.write(f"Average Batch Processing Time: {results['avg_batch_time']:.4f} seconds\n")
            f.write(f"Total Test Time: {results['total_test_time']:.2f} seconds\n")
            f.write(f"Maximum RAM Usage: {results['max_ram_gb']:.2f} GB\n")
            f.write(f"Average RAM Usage: {results['mean_ram_gb']:.2f} GB\n")
            f.write(f"Maximum GPU Memory Usage: {results['max_gpu_gb']:.2f} GB\n")
            f.write(f"Average GPU Memory Usage: {results['mean_gpu_gb']:.2f} GB\n\n")
            
            f.write(f"Resource plots saved to: {plot_path}\n")
            f.write(f"Detailed resource logs saved to: {log_path}\n")

        
        # Print summary to console
        print('\nTest Results:')
        print(f'Loss: {test_loss:.4f}')
        print(f'Accuracy: {test_accuracy:.4f}')
        print(f'F1 Score: {test_f1:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print('\nResource Usage:')
        print(f'Max RAM: {resource_summary["ram_max_gb"]:.2f} GB')
        print(f'Max GPU Memory: {resource_summary["gpu_max_gb"]:.2f} GB')
        print(f'Test Duration: {resource_summary["duration_seconds"]:.2f} seconds')
        print(f'\nResults saved to: {results_path}')
        print(f'Resource plots saved to: {plot_path}')

        # save confusion matrix cm
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('confusion_matrix_testing')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig('confusion_matrix_NST.png')
        plt.close()
        
        return test_loss, test_accuracy, test_f1, resource_summary















