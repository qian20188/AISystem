import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
import time
# ================== 模型定义 ==================
class MLP32(nn.Module):
    def __init__(self, input_dim=3*32*32, hidden_dims=[512, 256, 128], num_classes=2, dropout_prob=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.bn3 = nn.BatchNorm1d(hidden_dims[2])
        self.fc4 = nn.Linear(hidden_dims[2], num_classes)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.fc4(x)
        return x
def load_dataset(base_path, batch_size=64):
    """
    使用：
    - base_path/train → train_loader
    - base_path/val   → val_loader
    没有 test
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    # 直接加载
    train_data = datasets.ImageFolder(os.path.join(base_path, 'train'), transform=transform)
    val_data   = datasets.ImageFolder(os.path.join(base_path, 'val'),   transform=transform)

    print("Train:", train_data.class_to_idx)
    print("Val:  ", val_data.class_to_idx)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_data,   batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"Train samples: {len(train_data)}")
    print(f"Val samples:   {len(val_data)}")

    return train_loader, val_loader


# ================== 训练与验证 ==================
def train(model, train_loader, val_loader, device, epochs=20, lr=1e-3, weight_decay=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # ======= 验证评估阶段 =======
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        avg_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f}, Val Acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_sd1.4_max_mlp32.pth")
            print("✅ 模型更新，保存中...")

    print(f"🎯 最佳验证准确率: {best_acc:.4f}")


# ================== 主程序 =============================================
# ----------------直接从已经处理好32*32图像的数据集进行训练-----------------------
if __name__ == "__main__":
    base_path = "/home2/yaojiayi/datasets/SD1_4_max_32"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader= load_dataset(base_path, batch_size=64)
    model = MLP32().to(device)
    
    start_time = time.time()  # ⏱ 开始计时
    train(model, train_loader, val_loader, device, epochs=20, lr=1e-3, weight_decay=1e-4)
    end_time = time.time()    # ⏱ 结束计时
    total = end_time - start_time

    print("==============训练结束==========")
    print(f"⏱ 总训练时间: {total/60:.2f} 分钟 ({total:.2f} 秒)")
