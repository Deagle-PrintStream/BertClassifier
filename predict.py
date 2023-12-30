# coding: utf-8
# @File: predict.py
# @Author: HE D.H.
# @Email: victor-he@qq.com
# @Time: 2020/10/10 17:13:57
# @Description:

import torch
import torch.nn as nn
import tqdm
import logging

from torch.utils.data import DataLoader
from model import BertClassifier
from transformers import BertTokenizer, BertConfig
from sklearn.metrics import f1_score, accuracy_score, classification_report

from dataset import CNewsDataset


def predict(
    model: torch.nn.Module,
    test_dataset: CNewsDataset,
    batch_size: int,
    epochs: int,
    device: str = "cpu",
) -> float:
    pred_labels = []
    true_labels = []
    losses = 0

    criterion = nn.CrossEntropyLoss()  # loss function
    valid_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    for epoch in range(1, epochs + 1):
        valid_bar = tqdm(valid_dataloader, ncols=100)
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

        average_loss = losses / len(valid_dataloader)
        logging.info("\tAverage loss:", average_loss)

        # 分类报告
        report = classification_report(
            true_labels,
            pred_labels,
            labels=test_dataset.labels_id,
            target_names=test_dataset.labels,
        )
        logging.info("* Classification Report:")
        logging.info(report)

        # f1 用来判断最优模型
        f1 = f1_score(
            true_labels, pred_labels, labels=test_dataset.labels_id, average="macro"
        )
        acc = accuracy_score(
            true_labels, pred_labels, labels=test_dataset.labels_id, normalize=True
        )

        logging.info(f"acc:{round(acc,4)}\t f1-score:{round(f1,4)}")
        return f1


labels = [
    "domestic",
    "foriegn",
]
bert_config = BertConfig.from_pretrained("bert-base-chinese")

# 定义模型
model = BertClassifier(bert_config, len(labels))

# 加载训练好的模型
model.load_state_dict(
    torch.load("models/best_model.pkl", map_location=torch.device("cpu"))
)
model.eval()

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

print("新闻类别分类")
while True:
    text = input("Input: ")
    token = tokenizer(
        text,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
        max_length=512,
    )
    input_ids = token["input_ids"]
    attention_mask = token["attention_mask"]
    token_type_ids = token["token_type_ids"]

    input_ids = torch.tensor([input_ids], dtype=torch.long)
    attention_mask = torch.tensor([attention_mask], dtype=torch.long)
    token_type_ids = torch.tensor([token_type_ids], dtype=torch.long)

    predicted = model(
        input_ids,
        attention_mask,
        token_type_ids,
    )
    pred_label = torch.argmax(predicted, dim=1)

    print("Label:", labels[pred_label])
