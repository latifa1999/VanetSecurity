import os
import torch
from models import TimesNet, Nonstationary_Transformer,Informer
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from sklearn.metrics import confusion_matrix

class MetricsTracker:
    """Tracks and manages training/validation metrics"""
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)
        self.metrics = {
            'train': {'loss': [], 'accuracy': [], 'f1': []},
            'val': {'loss': [], 'accuracy': [], 'f1': []},
            'test': {'loss': [], 'accuracy': [], 'f1': []}
        }
        
    def update(self, phase, epoch, loss, accuracy=None, f1=None):
        """Update metrics for a given phase"""
        self.metrics[phase]['loss'].append(loss)
        if accuracy is not None:
            self.metrics[phase]['accuracy'].append(accuracy)
        if f1 is not None:
            self.metrics[phase]['f1'].append(f1)
            
        # Log to TensorBoard
        self.writer.add_scalar(f'{phase}/Loss', loss, epoch)
        if accuracy is not None:
            self.writer.add_scalar(f'{phase}/Accuracy', accuracy, epoch)
        if f1 is not None:
            self.writer.add_scalar(f'{phase}/F1', f1, epoch)
    
    def plot_metrics(self, save_dir):
        """Generate and save metric plots"""
        # Learning curves
        plt.figure(figsize=(15, 5))
        
        # Loss plot
        plt.subplot(1, 3, 1)
        for phase in ['train', 'val', 'test']:
            if self.metrics[phase]['loss']:
                plt.plot(self.metrics[phase]['loss'], label=f'{phase}')
        plt.title('Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Accuracy plot
        plt.subplot(1, 3, 2)
        for phase in ['train', 'val', 'test']:
            if self.metrics[phase]['accuracy']:
                plt.plot(self.metrics[phase]['accuracy'], label=f'{phase}')
        plt.title('Accuracy Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        # F1 plot
        plt.subplot(1, 3, 3)
        for phase in ['train', 'val', 'test']:
            if self.metrics[phase]['f1']:
                plt.plot(self.metrics[phase]['f1'], label=f'{phase}')
        plt.title('F1 Score Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
        plt.close()


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'Nonstationary_Transformer': Nonstationary_Transformer,
            'Informer': Informer,
        }

        # Setup logging and visualization
        self.experiment_name = f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.log_dir = os.path.join('Tensorboard/', self.experiment_name)
        self.plot_dir = 'Training'
        os.makedirs(self.plot_dir, exist_ok=True)
        
        # Initialize metrics tracker
        self.metrics = MetricsTracker(self.log_dir)
        
        # Initialize device
        self.device = self._acquire_device()
        self.model = self._build_model()

    def _acquire_device(self):
        """Set up and return the appropriate device"""
        if self.args.use_gpu and torch.cuda.is_available():
            if self.args.use_multi_gpu:
                device_ids = [int(id) for id in self.args.devices.split(',')]
                os.environ["CUDA_VISIBLE_DEVICES"] = self.args.devices
                device = torch.device(f'cuda:{device_ids[0]}')
                print(f'Using GPUs: {device_ids}')
            else:
                device = torch.device(f'cuda:{self.args.gpu}')
                print(f'Using GPU: cuda:{self.args.gpu}')
        else:
            device = torch.device('cpu')
            print('Using CPU')
        return device
    
    def _save_model(self, epoch, model_dict, optimizer_dict, best_val_metric, filename):
        """Save model checkpoint with detailed information"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_dict,
            'optimizer_state_dict': optimizer_dict,
            'best_val_metric': best_val_metric,
            'args': vars(self.args),
            'experiment_name': self.experiment_name
        }
        
        try:
            torch.save(checkpoint, filename)
            print(f'Successfully saved model checkpoint to: {filename}')
        except Exception as e:
            print(f'Error saving model checkpoint: {str(e)}')

    def _load_model(self, checkpoint_path):
        """Load model with proper error handling and device mapping"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            state_dict = checkpoint if "model_state_dict" not in checkpoint else checkpoint["model_state_dict"]
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
            model_dict.update(pretrained_dict)
            
            self.model.load_state_dict(model_dict)
            print(f"Successfully loaded model from {checkpoint_path}")
            return checkpoint
        except Exception as e:
            print(f"Error loading model from {checkpoint_path}: {str(e)}")
            return None

    def _plot_confusion_matrix(self, y_true, y_pred, title, filename):
        """Generate and save confusion matrix plot"""
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join('Training', filename))
        plt.close()
        return cm

    def _log_batch_metrics(self, phase, epoch, batch_idx, loss, accuracy=None, f1=None):
        """Log batch-level metrics to TensorBoard"""
        step = epoch * self.args.batch_size + batch_idx
        self.metrics.writer.add_scalar(f'Batch/{phase}_Loss', loss, step)
        if accuracy is not None:
            self.metrics.writer.add_scalar(f'Batch/{phase}_Accuracy', accuracy, step)
        if f1 is not None:
            self.metrics.writer.add_scalar(f'Batch/{phase}_F1', f1, step)

    def _log_model_params(self):
        """Log model parameters and gradients"""
        for name, param in self.model.named_parameters():
            self.metrics.writer.add_histogram(f'Parameters/{name}', param.data, 0)
            if param.grad is not None:
                self.metrics.writer.add_histogram(f'Gradients/{name}', param.grad, 0)

    def _build_model(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def _get_data(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def vali(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def train(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def test(self):
        raise NotImplementedError("Subclass must implement abstract method")

    def finetune(self):
        raise NotImplementedError("Subclass must implement abstract method")