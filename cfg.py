import time
import torch
import os
import torch.nn as nn
import torch.optim as optim

from model.ODETAU import ODETAU

import subprocess
import argparse

def find_a_gpu(device='auto',min_gpu=10):
    if device.startswith('auto'):
        while True:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            memory_info = result.stdout.strip().split('\n')
            used_memory = [round(float(line.split(',')[0]) / 1000, 1) for line in memory_info]
            print(f"\rGPU memory usage: {used_memory} GB")
            available_gpus = [(i, mem) for i, mem in enumerate(used_memory) if mem < min_gpu]

            if available_gpus:
                best_gpu = min(available_gpus, key=lambda x: x[1])
                return f'cuda:{best_gpu[0]}'

            time.sleep(10)
    else:
        return device


class Config():
    def __init__(self):
        # self.test = os.getenv('PYCHARM_HOSTED') == '1'
        self.test =0 #Test mode to quickly check code availability
        self.tik = time.time()
        parser = argparse.ArgumentParser(description="Training and evaluation script")
        parser.add_argument('model_type', type=str, default='ODETAU', nargs='?')
        parser.add_argument('device', type=str, default='auto', nargs='?')#cuda:0
        parser.add_argument('--tag', '--t', type=str, default='default', nargs='?')
        parser.add_argument('--load_optimizer', '--lo', type=int, default=0)
        parser.add_argument('--load_best_weights', '--lw', type=int, default=0)
        parser.add_argument('--weight_dir', '--wd', type=str, default='')
        parser.add_argument('--input_length', '--il', type=int, default=12)  # 12
        parser.add_argument('--ouput_length', '--ol', type=int, default=4)  # 4

        parser.add_argument('--lr', type=float, default=1e-4)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--batch_size', type=int, default=2)
        parser.add_argument('--light', '--l', type=int, default=0)
        parser.add_argument('--preload', '--p', type=float, default=1)
        parser.add_argument('--dataset', '--d', type=str, default='USA')
        parser.add_argument('--seed', '--s', type=int, default=42)
        # parser.add_argument('--autodl', '--a', type=int, default=1)


        self.num_workers=4
        args = parser.parse_args()
        self.tag =args.tag

        self.light = args.light
        self.autodl = args.autodl
        self.seed = args.seed
        self.weight_dir = args.weight_dir
        self.lr = args.lr
        self.device = args.device
        self.model_type = args.model_type
        self.preload = args.preload
        self.dataset=args.dataset
        self.input_length = args.input_length
        self.ouput_length = args.ouput_length
        self.load_optimizer = args.load_optimizer
        self.load_best_weights = args.load_best_weights or self.weight_dir
        self.ground_truth = 'groundtruth'

        self.num_epochs = 200
        self.batch_size = args.batch_size
        self.patience = 5

        self.datadir = ''
        self.codedir = ''


        if self.dataset=='CHN':
            self.pollution_list = ['AQI', 'CO', 'NO2', 'O3', 'PM10', 'PM2.5', 'SO2']
            self.map_size = (80, 130)
            self.atmosphere_size = (354, 80, 130)
        elif self.dataset=='USA':
            self.pollution_list = ['CO', 'NO2', 'NOy', 'O3', 'PM10', 'PM2.5', 'SO2']
            self.map_size = (70, 140)
            self.atmosphere_size = (354, 70, 140)
        else:
            print('No such dataset')
        self.pixel_number = self.map_size[0] * self.map_size[1]

        self.which_is_target='PM2.5'
        # self.which_is_target = 'all'

        self.pollution_type = 'PM2.5'
        # self.pollution_type = 'all'

        self.pollution_type_number=1
        if self.pollution_type == 'all':
            self.pollution_type_number=7

        if self.test:
            self.input_length = 12
            self.ouput_length = 4
            self.preload = 0.05
        self.length = self.input_length + self.ouput_length


        if self.model_type == 'ODETAU':
            self.model = ODETAU(self.input_length, self.ouput_length, atmosphere_size=self.atmosphere_size)
        else:
            self.model = None


        self.device=find_a_gpu(self.device)

        self.model=self.model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=args.weight_decay)
        self.message=f'{self.device} {self.model_type} {self.dataset}_dataset {self.length}:{self.input_length}->{self.ouput_length} {self.tag} Preload:{int(self.preload * 100)}%'
        print(self.message)

cfg = Config()
