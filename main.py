import argparse
import os
import torch.nn as nn
import numpy
from solver import Solver
from patched_data_loader import get_my_loader, det_my_dataset
from torch.backends import cudnn
import random
import cv2
import predict_patched
from PIL import Image
import torch
import atexit
import threading
from checkerQt import Checker
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton
from PyQt5.QtCore import QThread
import sys
from main_window import MainWindow
from infer_patches import averag_threshold
from infer_patches import make_prediction, normalize_patches, post3d_process


def main(config):
    cudnn.benchmark = True
    

    # Create directories if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    config.result_path = os.path.join(config.result_path,config.model_type)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    
    lr = random.random()*0.0005 + 0.0000005
    augmentation_prob= random.random()*0.7
    epoch = random.choice([100,150,200,250])
    decay_ratio = random.random()*0.8
    decay_epoch = int(epoch*decay_ratio)

    config.augmentation_prob = augmentation_prob
    #config.lr = lr
    config.num_epochs_decay = decay_epoch

    print(config)
        


    train_loader = get_my_loader(image_path=config.train_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='train',
                            augmentation_prob=config.augmentation_prob)
    valid_loader = get_my_loader(image_path=config.valid_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='valid',
                            augmentation_prob=0.)
    test_loader = get_my_loader(image_path=config.test_path,
                            image_size=config.image_size,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers,
                            mode='test',
                            augmentation_prob=0.)

    solver = Solver(config, train_loader, valid_loader, test_loader)
    # solver.train()     # for debugging 
    # make_prediction("./models/U_Net_plus-172-5.8009-2.5555-final.pkl", config.test_path)   # for debugging 
    app = QApplication(sys.argv)
    window = MainWindow(solver, config)
    window.show()
    window.start()  # Start threads of training or inference
    sys.exit(app.exec_())
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    
    # model hyper-parameters
    parser.add_argument('--image_size', type=int, default=(1024,1024))
    
    # training hyper-parameters
    parser.add_argument('--img_ch', type=int, default=1)
    parser.add_argument('--output_ch', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_epochs_decay', type=int, default=70)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--beta1', type=float, default=0.5)        # momentum1 in Adam
    parser.add_argument('--beta2', type=float, default=0.999)      # momentum2 in Adam    
    parser.add_argument('--augmentation_prob', type=float, default=0.4)

    parser.add_argument('--log_step', type=int, default=2)
    parser.add_argument('--val_step', type=int, default=2)

    # misc
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--model_type', type=str, default='U_Net_plus')
    parser.add_argument('--model_path', type=str, default='./models/')
    parser.add_argument('--train_path', type=str, default='./dataset/train/')
    parser.add_argument('--valid_path', type=str, default='./dataset/valid/')
    parser.add_argument('--test_path', type=str, default='./dataset/test/')
    parser.add_argument('--result_path', type=str, default='./result/')

    parser.add_argument('--cuda_idx', type=int, default=1)

    config = parser.parse_args()

    main(config)
    
    
