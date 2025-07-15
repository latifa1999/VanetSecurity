# Time Series Anomaly Detection Module
This module implements state-of-the-art time-series models for detecting anomalies in V2V (Vehicle-to-Vehicle) communication messages. It supports multiple deep learning architectures optimized for time-series classification tasks.

## Overview
The module provides a comprehensive framework for:
- Training time-series models on VANET datasets
- Real-time anomaly detection in V2V messages
- Model evaluation and performance analysis
- Standalone inference capabilities

## Directory Structure

<img width="712" height="532" alt="image" src="https://github.com/user-attachments/assets/5813140e-2a2d-400b-862e-422236f13d73" />



## Installation
- Python 3.8+
- CUDA-capable GPU (recommended)
- PyTorch 1.9+

##Setup Environment
```
# Create virtual environment
python -m venv ts_env
source ts_env/bin/activate  # On Windows: ts_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

#train model
python run.py \
  --task_name classification \
  --is_training 1 \    # for finetuning: --is_training 0 --finetune  # for testing: --is_training 0
  --root_path ./dataset/ \
  --data_path VeReMi \
  --model_id VeReMi_Informer \
  --model Informer \
  --e_layers 3 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 2 \
  --batch_size 16 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10

```

## Citation
If you use this time-series anomaly detection module, please cite the original papers:
```
@inproceedings{informer2021,
  title={Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting},
  author={Zhou, Haoyi and Zhang, Shanghang and Peng, Jieqi and Zhang, Shuai and Li, Jianxin and Xiong, Hui and Zhang, Wancai},
  booktitle={AAAI},
  year={2021}
}

@article{timesnet2023,
  title={TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis},
  author={Wu, Haixu and Hu, Tengge and Liu, Yong and Zhou, Hang and Wang, Jianmin and Long, Mingsheng},
  journal={ICLR},
  year={2023}
}
```
