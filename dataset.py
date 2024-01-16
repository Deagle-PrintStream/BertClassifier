r"""
    file: dataset.py
    author: Swayee
    email:  suiyi_liu@mail.ustc.edu.cn
    time:   2023/12/31
    encoding:   utf-8
"""

import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import Dataset
from transformers import BertTokenizer

class TextDataset(Dataset):
    def __init__(self, file_path: str = "./dataset/dataset.xlsx", transform=None):
        df_dataset = pd.read_excel(file_path)  # Load data from the XLSX file
        size_dataset=len(df_dataset)
        self.item_list:list[tuple[np.ndarray, np.ndarray, np.ndarray, int]]=[]
        self.transform = transform
        self.labels = [ #labels in raw dataset didn't provide clear meaning
            "0",
            "1",
        ]
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese",cache_dir=None,local_files_only=False)
        
        for idx in range(0,size_dataset):
            text = str(df_dataset.iloc[idx, 0])  #  text in the first column
            label = int(df_dataset.iloc[idx, 1])  # labels in the second column

            token = self.tokenizer(
                text,
                add_special_tokens=True,
                padding="max_length",
                truncation=True,
                max_length=512,
            )
            input_ids = np.array(token["input_ids"])
            token_type_ids = np.array(token["token_type_ids"])
            attention_mask = np.array(token["attention_mask"])
            self.item_list.append((input_ids,token_type_ids,attention_mask,label))
        
        

    def __len__(self):
        return len(self.item_list)

    def __getitem__(self, idx):
        return self.item_list[idx]

