import os
from torch.utils.data import Dataset
import torch

def accuracy(outputs, labels, k=1):
    batch_size = labels.size(0)
    
    if k == 1:
        _, preds = torch.max(outputs, dim=1)
        acc = torch.sum(preds == labels).item() / batch_size
    else:
        top_probs, top_indices = outputs.topk(k=k, dim=1)
        correct = top_indices.eq(labels.view(-1, 1).expand_as(top_indices))
        correct_k = correct.view(-1).float().sum(0, keepdim=True).item()
        acc = correct_k / batch_size

    return acc

class TextClassificationDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        # file_path: (str) 数据文件的路径
        # tokenizer: (BertTokenizer) BERT的词典
        self.tokenizer = tokenizer
        self.texts = []
        self.inputs = []
        self.targets = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 每一行都是"类别\t文本"的格式，我们将其分割为类别和文本
                category, text = line.strip().split('\t')
                
                # 使用词典对文本进行编码
                # 这将文本转换为模型可以接受的输入格式
                inputs = tokenizer(text, truncation=True, padding='max_length', max_length=128)
                self.texts.append(text)
                self.inputs.append(inputs)          # 将输入添加到输入列表中
                self.targets.append(int(category))  # 将类别转换为整数并添加到目标列表中

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # 根据索引获取数据，返回包含输入和目标的 dict
        # item = {key: torch.tensor(val[idx], dtype=torch.long) for key, val in self.inputs.items()}
        item = {}
        for key, val in self.inputs[idx].items():
            item[key] =  torch.tensor(val, dtype=torch.long)  # 将输入转换为张量
        item['labels'] = torch.tensor(self.targets[idx], dtype=torch.long)  # 将目标转换为张量
        return item


if __name__ == "__main__":
    if not os.path.exists("data/test.txt"):
        # 下载数据集
        train_url = "https://raw.githubusercontent.com/xubuvd/Short-Text-Classification/main/VSQ/train.txt"
        dev_url = "https://raw.githubusercontent.com/xubuvd/Short-Text-Classification/main/VSQ/dev.txt"
        test_url = "https://raw.githubusercontent.com/xubuvd/Short-Text-Classification/main/VSQ/test.txt"
        class_url = "https://raw.githubusercontent.com/xubuvd/Short-Text-Classification/main/VSQ/class.txt"

        print("未检测到完整数据集，正在下载...")
        wget.download(train_url, 'train.txt')
        wget.download(dev_url, 'dev.txt')
        wget.download(test_url, 'test.txt')
        wget.download(class_url, 'class.txt')
        print("下载完毕。")