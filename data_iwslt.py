import os
import re
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset

def read_xml(filepath):
    """解析 XML 文件，提取 <seg> 段落文本"""
    tree = ET.parse(filepath)
    root = tree.getroot()
    texts = [seg.text.strip() for seg in root.iter("seg") if seg.text]
    return texts

def read_txt(filepath):
    """读取 IWSLT2017 训练集纯文本文件，跳过注释与标签"""
    texts = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # 跳过 XML / HTML / 注释 / 空行
            if not line or line.startswith("<") or line.startswith("&"):
                continue
            texts.append(line)
    return texts

def load_iwslt_dataset(data_dir):
    """加载 IWSLT2017 英德翻译数据集"""
    train_de = read_txt(os.path.join(data_dir, "train.tags.en-de.de"))
    train_en = read_txt(os.path.join(data_dir, "train.tags.en-de.en"))

    # 🔍 自动对齐句子数（有些版本略有差异）
    min_len = min(len(train_de), len(train_en))
    train_de, train_en = train_de[:min_len], train_en[:min_len]

    dev_de = read_xml(os.path.join(data_dir, "IWSLT17.TED.dev2010.en-de.de.xml"))
    dev_en = read_xml(os.path.join(data_dir, "IWSLT17.TED.dev2010.en-de.en.xml"))

    test_de = read_xml(os.path.join(data_dir, "IWSLT17.TED.tst2010.en-de.de.xml"))
    test_en = read_xml(os.path.join(data_dir, "IWSLT17.TED.tst2010.en-de.en.xml"))

    assert len(dev_de) == len(dev_en)
    assert len(test_de) == len(test_en)

    dataset = {
        "train": [{"de": d, "en": e} for d, e in zip(train_de, train_en)],
        "validation": [{"de": d, "en": e} for d, e in zip(dev_de, dev_en)],
        "test": [{"de": d, "en": e} for d, e in zip(test_de, test_en)],
    }

    print("✅ 成功加载 IWSLT2017 数据集！")
    print(f"训练样本数: {len(train_de)}")
    print(f"验证样本数: {len(dev_de)}")
    print(f"测试样本数: {len(test_de)}")
    return dataset

class TranslationDataset(Dataset):
    def __init__(self, data, tokenizer=None, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if self.tokenizer:
            src = self.tokenizer(
                item["de"],
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                return_tensors="pt",
            )
            tgt = self.tokenizer(
                item["en"],
                truncation=True,
                padding="max_length",
                max_length=self.max_len,
                return_tensors="pt",
            )
            return {"src": src, "tgt": tgt}
        return item


if __name__ == "__main__":
    dataset = load_iwslt_dataset("data")
    print(dataset["train"][0])
