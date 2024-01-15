# coding: utf-8
# @File: train.py
# @Author: HE D.H.
# @Email: victor-he@qq.com
# @Time: 2020/10/10 17:14:07
# @Description:

import torch
import torch.nn as nn
import logging


from transformers import BertTokenizer, AdamW, BertConfig
from torch.utils.data import DataLoader
from model import BertClassifier
from dataset import TextDataset
from tqdm import tqdm

"""
    bagging may be required since we have too few data
"""


def train(
    train_dataset: TextDataset,
    batch_size: int = 4,
    device: str = "cpu",
    epochs: int = 10,
    learning_rate: float = 1e-5,
) -> nn.Module:
    bert_config = BertConfig.from_pretrained("bert-base-chinese")
    num_labels = 2  #   two labels: 1 and 0

    model = BertClassifier(bert_config, num_labels).to(device)  # init model

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    criterion = nn.CrossEntropyLoss()  # loss function

    best_f1 = 0

    for epoch in range(1, epochs + 1):
        losses = 0  # 损失
        accuracy = 0  # 准确率

        model.train()
        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        train_bar = tqdm(train_dataloader, ncols=100)
        for input_ids, token_type_ids, attention_mask, label_id in train_bar:
            # 梯度清零
            model.zero_grad()
            train_bar.set_description("Epoch %i train" % epoch)

            # 传入数据，调用model.forward()
            output = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                token_type_ids=token_type_ids.to(device),
            )

            # 计算loss
            loss = criterion(output, label_id.to(device))
            losses += loss.item()

            pred_labels = torch.argmax(output, dim=1)  # 预测出的label
            acc = torch.sum(pred_labels == label_id.to(device)).item() / len(
                pred_labels
            )  # acc
            accuracy += acc

            loss.backward()
            optimizer.step()
            train_bar.set_postfix(loss=loss.item(), acc=acc)

        average_loss = losses / len(train_dataloader)
        average_acc = accuracy / len(train_dataloader)
        
        logging.info("epoch %d\tloss %.4f\tacc %.4f"%(epoch,average_loss,average_acc))

    return model
