# coding: utf-8
# @File: dataset.py
# @Author: HE D.H.
# @Email: victor-he@qq.com
# @Time: 2021/12/09 11:01:32
# @Description:

import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BertTokenizer
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import Subset
from sklearn.model_selection import KFold

class TextDataset(Dataset):
    def __init__(self, file_path:str="./dataset/dataset.xlsx", transform=None):
        self.data = pd.read_excel(file_path)  # Load data from the XLSX file
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data.iloc[idx, 0]  #  text in the first column
        label = self.data.iloc[idx, 1]  # labels in the second column

        if self.transform!=None:
            text = self.transform(text)

        return text, label

# You can define a custom transform function for text preprocessing if needed
def text_preprocess(text):
    # Implement your text preprocessing logic here
    # This could involve tokenization, vectorization, padding, etc.
    return text

class CNewsDataset(Dataset):
    def __init__(self, filename):
        # dataset object initialization
        self.labels = ['domestic', 'foriegn',]
        self.labels_id = list(range(len(self.labels)))
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        self.input_ids = []
        self.token_type_ids = []
        self.attention_mask = []
        self.label_id = []
        self.load_data(filename)
    
    def load_data(self, filename:str="./dataset/dataset.xlsx")->None:
        # read in data from xlsx file
        print('loading data from:', filename)
        df=pd.read_excel(filename)
        
        for i in range(1,len(df)+1):
            text,label=df[i,"数据"],df[i,"标签"]
            #label_id = self.labels.index(label)
            label_id = int(label)
            token = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True, max_length=512)
            self.input_ids.append(np.array(token['input_ids']))
            self.token_type_ids.append(np.array(token['token_type_ids']))
            self.attention_mask.append(np.array(token['attention_mask']))
            self.label_id.append(label_id)

        '''    
        with open(filename, 'r', encoding='utf-8') as rf:
            lines = rf.readlines()
        for line in tqdm(lines, ncols=100):
            label, text = line.strip().split('\t')
            label_id = self.labels.index(label)
            token = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True, max_length=512)
            self.input_ids.append(np.array(token['input_ids']))
            self.token_type_ids.append(np.array(token['token_type_ids']))
            self.attention_mask.append(np.array(token['attention_mask']))
            self.label_id.append(label_id)
        '''

    def __getitem__(self, index):
        return self.input_ids[index], self.token_type_ids[index], self.attention_mask[index], self.label_id[index]

    def __len__(self):
        return len(self.input_ids)

