r"""
    file: dataset.py
    author: Swayee
    email:  suiyi_liu@mail.ustc.edu.cn
    time:   2024/1/15
    encoding:   utf-8
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import logging

from torch.utils.data import DataLoader
from model import BertClassifier
from transformers import BertTokenizer, BertConfig
from sklearn.metrics import f1_score, accuracy_score, classification_report

from dataset import TextDataset


def predict(
    model: torch.nn.Module,
    test_dataset: TextDataset,
    batch_size: int,
    epochs: int,
    device: str = "cpu",
) -> float:
    pred_labels = []
    true_labels = []
    losses = 0

    model.eval()
    criterion = nn.CrossEntropyLoss()  # loss function
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(1, epochs + 1):
        valid_bar = tqdm(test_dataloader, ncols=100)
        for input_ids, token_type_ids, attention_mask, label_id in valid_bar:
            valid_bar.set_description("Epoch %i valid" % epoch)

            output = model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
                token_type_ids=token_type_ids.to(device),
            )

            loss = criterion(output, label_id.to(device))
            losses += loss.item()

            pred_label = torch.argmax(output, dim=1)  # 预测出的label
            acc = torch.sum(pred_label == label_id.to(device)).item() / len(
                pred_label
            )  # acc
            valid_bar.set_postfix(loss=loss.item(), acc=acc)

            pred_labels.extend(pred_label.cpu().numpy().tolist())
            true_labels.extend(label_id.numpy().tolist())

        average_loss = losses / len(test_dataloader)
        logging.debug("Epoch %d loss %.3f acc %.3f "%(epoch+1,average_loss,acc))

        # 分类报告
        
        report = classification_report(
            true_labels,
            pred_labels,
        )
        logging.info("* Classification Report:")
        logging.info(report)


    f1 = f1_score(
        true_labels, pred_labels,  average="macro"
    )
    acc = accuracy_score(
        true_labels, pred_labels, normalize=True
    )

    logging.info(f"acc:{round(acc,4)}\t f1-score:{round(f1,4)}")
    return f1
