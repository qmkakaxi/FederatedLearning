#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--epochs', type=int, default=10, help="rounds of training")
    parser.add_argument('--bs', type=int, default=64, help="test batch size")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--init-method', type=str, default='tcp://192.168.1.2:9113') #选择第五台服务器的ip
    parser.add_argument('--rank', type=int)
    parser.add_argument('--world-size',type=int,default=4)  #后四台服务器
    parser.add_argument('--num-sample-rka', type=int, default=10)  #rka算法采样个数
    parser.add_argument('--id-remove',type=int,default=0) #retrain的remove id
    args = parser.parse_args()
    return args
