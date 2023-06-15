import os
import wget
import torch
from transformers import BertForSequenceClassification, Trainer, TrainingArguments, BertTokenizer
from data_utils import TextClassificationDataset

if __name__ == "__main__":
    train_dataset = TextClassificationDataset('data/train.txt', tokenizer)
    val_dataset = TextClassificationDataset('data/dev.txt', tokenizer)
    test_dataset = TextClassificationDataset('data/test.txt', tokenizer)

    # 加载预训练模型和词典
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=len(class_labels))

    # 设定训练参数
    training_args = TrainingArguments(
        output_dir='./results',          # 输出目录
        num_train_epochs=2,              # 总的训练轮数
        per_device_train_batch_size=64,  # 每个GPU的训练batch大小
        per_device_eval_batch_size=64,   # 每个GPU的评估batch大小
        warmup_steps=500,                # 预热步数
        weight_decay=0.01,               # 权重衰减
        logging_dir='./logs',            # 日志目录
        logging_steps=20,
    )

    # 设定训练器
    trainer = Trainer(
        model=model,                     # 待训练的模型
        args=training_args,              # 训练参数
        train_dataset=train_dataset,     # 训练数据集
        eval_dataset=val_dataset         # 验证数据集
    )

    # 开始训练和验证
    trainer.train()

    trainer.evaluate()