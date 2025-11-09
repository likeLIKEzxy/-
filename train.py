import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data_iwslt import load_iwslt_dataset, TranslationDataset
from src.model import TransformerModel
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import yaml
import argparse


# ===========================
# 🔹 解析命令行参数
# ===========================
def parse_args():
    parser = argparse.ArgumentParser(description="Train Transformer Model")
    parser.add_argument("--config", type=str, default="src/configs/base.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--tag", type=str, default=None,
                        help="Experiment tag for naming output (e.g., no_posenc)")
    return parser.parse_args()


# ===========================
# 🔹 加载配置 + 类型转换
# ===========================
def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 自动类型修正
    float_keys = ["lr", "weight_decay"]
    int_keys = ["batch_size", "num_epochs", "d_model", "nhead",
                "num_encoder_layers", "num_decoder_layers", "dim_feedforward"]

    for k in float_keys:
        if k in cfg:
            cfg[k] = float(cfg[k])
    for k in int_keys:
        if k in cfg:
            cfg[k] = int(cfg[k])

    return cfg


# ===========================
# 🔹 训练与评估函数
# ===========================
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="训练中"):
        src = batch["src"]["input_ids"].squeeze(1).to(device)
        tgt = batch["tgt"]["input_ids"].squeeze(1).to(device)
        optimizer.zero_grad()
        output = model(src, tgt[:, :-1])  # decoder输入不包含最后一个token
        loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            src = batch["src"]["input_ids"].squeeze(1).to(device)
            tgt = batch["tgt"]["input_ids"].squeeze(1).to(device)
            output = model(src, tgt[:, :-1])
            loss = criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)


# ===========================
# 🔹 绘图
# ===========================
def plot_metrics(train_losses, val_losses, save_dir="results", tag="base"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{tag}_loss_curve.png")

    epochs = np.arange(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, "o-", label="Train Loss")
    plt.plot(epochs, val_losses, "o-", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training & Validation Loss ({tag})")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📈 曲线已保存至 {save_path}")


# ===========================
# 🔹 主程序
# ===========================
def main():
    args = parse_args()
    cfg = load_config(args.config)

    # 获取实验标签（优先命令行参数，否则从 config 文件名）
    tag = args.tag or os.path.splitext(os.path.basename(args.config))[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✅ 使用配置文件: {args.config}")
    print(f"✅ 实验标签: {tag}")
    print(f"✅ 使用设备: {device}\n")

    # ✅ 加载数据
    dataset = load_iwslt_dataset(cfg["data_dir"])
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")

    train_data = TranslationDataset(dataset["train"], tokenizer)
    val_data = TranslationDataset(dataset["validation"], tokenizer)
    train_loader = DataLoader(train_data, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=cfg["batch_size"])

    # ✅ 定义模型
    model = TransformerModel(
        src_vocab_size=tokenizer.vocab_size,
        tgt_vocab_size=tokenizer.vocab_size,
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_encoder_layers=cfg["num_encoder_layers"],
        num_decoder_layers=cfg["num_decoder_layers"],
        dim_feedforward=cfg["dim_feedforward"],
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(1, cfg["num_epochs"] + 1):
        print(f"\n🚀 Epoch {epoch}/{cfg['num_epochs']}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch} | 训练Loss: {train_loss:.4f} | 验证Loss: {val_loss:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # ✅ 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = os.path.join(cfg["save_dir"], f"{tag}_best_model.pt")
            os.makedirs(cfg["save_dir"], exist_ok=True)
            torch.save(model.state_dict(), model_path)
            print(f"✅ 已保存最佳模型至 {model_path}")

    # ✅ 绘图保存
    plot_metrics(train_losses, val_losses, save_dir=cfg["save_dir"], tag=tag)


if __name__ == "__main__":
    main()
