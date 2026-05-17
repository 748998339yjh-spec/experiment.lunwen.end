import os, sys, tempfile, time, warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

# ================= 全局配置 =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = os.path.join(tempfile.gettempdir(), "mnist_data")
os.makedirs(DATA_DIR, exist_ok=True)

# 严格对齐论文参数！
EPOCHS = 50
BATCH_SIZE = 64  # 已对齐论文表1配置
LR = 0.01
DP_CLIP = 1.0
TEST_SIZE = 1000
EPSILONS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

ENS_CONFIGS = [
    {"name": "s=0.1_A", "sigma": 0.1, "seed": 42},
    {"name": "s=0.1_B", "sigma": 0.1, "seed": 123},
    {"name": "s=0.3_A", "sigma": 0.3, "seed": 456},
    {"name": "s=0.3_B", "sigma": 0.3, "seed": 789},
    {"name": "s=0.5",   "sigma": 0.5, "seed": 101},
]
BASELINE_C_SIGMA = 0.3

# ================= 1. 数据切分 =================
def load_and_split_data():
    tf = transforms.Compose([transforms.ToTensor()])
    train_all = torchvision.datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=tf)
    test_all = torchvision.datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=tf)
    
    chunk_size = 5000
    train_loaders = []
    for i in range(5):
        indices = range(i * chunk_size, (i + 1) * chunk_size)
        subset = torch.utils.data.Subset(train_all, indices)
        train_loaders.append(torch.utils.data.DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True))
        
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(test_all, range(TEST_SIZE)), batch_size=BATCH_SIZE, shuffle=False)
    return train_loaders, test_loader

# ================= 2. 网络架构 =================
class ResNet18(nn.Module):
    def __init__(self, nc=10):
        super().__init__()
        base = torchvision.models.resnet18(weights=None)
        base.conv1 = nn.Conv2d(1, 64, 3, 1, 1, bias=False)
        base.maxpool = nn.Identity()
        base.avgpool = nn.AdaptiveAvgPool2d(1)
        base.fc = nn.Linear(512, nc)
        def rb(m):
            for n, c in m.named_children():
                if isinstance(c, nn.BatchNorm2d):
                    setattr(m, n, nn.GroupNorm(32, c.num_features))
                else: rb(c)
        rb(base)
        self.net = base
    def forward(self, x): return self.net(x)

# ================= 3. 训练引擎 =================
def standard_train(model, loader, model_name, add_input_noise=False):
    model.to(DEVICE).train()
    opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9) # 已对齐论文SGD和动量0.9
    crit = nn.CrossEntropyLoss()
    print(f"  > 正在训练 {model_name}...")
    for ep in range(EPOCHS):
        t0 = time.time()
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            if add_input_noise: imgs = torch.clamp(imgs + torch.randn_like(imgs) * 0.1, 0, 1)
            opt.zero_grad()
            crit(model(imgs), labels).backward()
            opt.step()
        print(f"    [{model_name}] Epoch {ep+1}/{EPOCHS} 完成 | 耗时: {time.time()-t0:.1f}秒")
    return model

def dp_sgd_train(model, loader, model_name, sigma):
    model.to(DEVICE).train()
    opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    print(f"  > 正在训练 {model_name}...")
    for ep in range(EPOCHS):
        t0 = time.time()
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            B = imgs.size(0)
            opt.zero_grad()
            crit(model(imgs), labels).backward()
            if sigma > 0:
                n2 = sum(p.grad.detach().pow(2).sum().item() for p in model.parameters() if p.grad is not None)
                sc = min(1.0, DP_CLIP / (n2**0.5 + 1e-8))
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(sc)
                        p.grad.add_(torch.randn_like(p.grad) * (sigma / B))
            opt.step()
        print(f"    [{model_name}] Epoch {ep+1}/{EPOCHS} 完成 | 耗时: {time.time()-t0:.1f}秒")
    return model

def fgsm_attack(model, images, labels, eps):
    images = images.clone().detach().requires_grad_(True)
    loss = nn.CrossEntropyLoss()(model(images), labels)
    loss.backward()
    return torch.clamp(images + eps * images.grad.sign(), 0, 1).detach()

def fgsm_ensemble_attack(models, images, labels, eps):
    images = images.clone().detach().requires_grad_(True)
    total_loss = sum(nn.CrossEntropyLoss()(m(images), labels) for m in models)
    total_loss.backward()
    return torch.clamp(images + eps * images.grad.sign(), 0, 1).detach()

# ================= 5. 主执行逻辑 =================
if __name__ == "__main__":
    train_loaders, test_loader = load_and_split_data()
    
    # ---------------- 核心逻辑 ----------------
    print("\n--- 阶段 1：训练对比组 (所有人，包括同源集成，全部严格限制在第1份 5000 张数据！) ---")
    model_A = standard_train(ResNet18(), train_loaders[0], "基线A(5000张)")
    model_B = standard_train(ResNet18(), train_loaders[0], "基线B(5000张)", add_input_noise=True)
    model_C = dp_sgd_train(ResNet18(), train_loaders[0], "基线C(5000张)", sigma=BASELINE_C_SIGMA)
    
    # 为了表1和表2的公平对比，训练同源集成模型（全都只吃第1份 5000 张数据）
    ensemble_same_5k = []
    for i, cfg in enumerate(ENS_CONFIGS):
        torch.manual_seed(cfg["seed"])
        m = ResNet18()
        dp_sgd_train(m, train_loaders[0], f"公平局集成子模型-{cfg['name']}(同为5000张)", sigma=cfg["sigma"])
        ensemble_same_5k.append(m)
        m.eval()

    print("\n--- 阶段 2：训练秀肌肉的非交组 (用于表3，吃满 5份不相交数据，共 25000 张) ---")
    ensemble_non_overlap_25k = []
    for i, cfg in enumerate(ENS_CONFIGS):
        torch.manual_seed(cfg["seed"])
        m = ResNet18()
        dp_sgd_train(m, train_loaders[i], f"非交局集成子模型-{cfg['name']}(第{i+1}份数据)", sigma=cfg["sigma"])
        ensemble_non_overlap_25k.append(m)
        m.eval()
        
    model_A.eval(); model_B.eval(); model_C.eval()

    print("\n正在进行全量 1000 张测试集攻击测试，生成论文表格...")

    # 表 1 和 表 2 的数据 (公平局 5000对5000)
    wbox = {n: [] for n in ['A', 'B', 'C', 'Ens_Same']}
    bbox = {n: [] for n in ['A', 'B', 'C', 'Ens_Same']}
    
    # 表 3 的数据 (非交局 25000)
    tab3 = {n: [] for n in ['White', 'Black']}

    for eps in EPSILONS:
        w_A = w_B = w_C = w_EnsS = b_A = b_B = b_C = b_EnsS = w_EnsN = b_EnsN = total = 0
        for bx, by in test_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            total += by.size(0)
            
            if eps == 0:
                # 干净准确率
                w_A += model_A(bx).argmax(1).eq(by).sum().item()
                w_B += model_B(bx).argmax(1).eq(by).sum().item()
                w_C += model_C(bx).argmax(1).eq(by).sum().item()
                w_EnsS += torch.mode(torch.stack([m(bx).argmax(1) for m in ensemble_same_5k]), dim=0)[0].eq(by).sum().item()
                w_EnsN += torch.mode(torch.stack([m(bx).argmax(1) for m in ensemble_non_overlap_25k]), dim=0)[0].eq(by).sum().item()
                # 干净无攻击，黑白盒一样
                b_A, b_B, b_C, b_EnsS, b_EnsN = w_A, w_B, w_C, w_EnsS, w_EnsN
            else:
                # ====== 白盒攻击 ======
                w_A += model_A(fgsm_attack(model_A, bx, by, eps)).argmax(1).eq(by).sum().item()
                w_B += model_B(fgsm_attack(model_B, bx, by, eps)).argmax(1).eq(by).sum().item()
                w_C += model_C(fgsm_attack(model_C, bx, by, eps)).argmax(1).eq(by).sum().item()
                
                # 公平局集成白盒
                adv_w_EnsS = fgsm_ensemble_attack(ensemble_same_5k, bx, by, eps)
                w_EnsS += torch.mode(torch.stack([m(adv_w_EnsS).argmax(1) for m in ensemble_same_5k]), dim=0)[0].eq(by).sum().item()
                
                # 非交局集成白盒 (表3用)
                adv_w_EnsN = fgsm_ensemble_attack(ensemble_non_overlap_25k, bx, by, eps)
                w_EnsN += torch.mode(torch.stack([m(adv_w_EnsN).argmax(1) for m in ensemble_non_overlap_25k]), dim=0)[0].eq(by).sum().item()
                
                # ====== 黑盒迁移攻击 (源：基线A) ======
                adv_b_A = fgsm_attack(model_A, bx, by, eps)
                b_A = w_A # 基线A打自己算白盒
                b_B += model_B(adv_b_A).argmax(1).eq(by).sum().item()
                b_C += model_C(adv_b_A).argmax(1).eq(by).sum().item()
                
                # 公平局集成黑盒
                b_EnsS += torch.mode(torch.stack([m(adv_b_A).argmax(1) for m in ensemble_same_5k]), dim=0)[0].eq(by).sum().item()
                
                # 非交局集成黑盒 (表3用)
                b_EnsN += torch.mode(torch.stack([m(adv_b_A).argmax(1) for m in ensemble_non_overlap_25k]), dim=0)[0].eq(by).sum().item()

        # 存入百分比
        wbox['A'].append(100.*w_A/total); wbox['B'].append(100.*w_B/total)
        wbox['C'].append(100.*w_C/total); wbox['Ens_Same'].append(100.*w_EnsS/total)
        
        bbox['A'].append(100.*b_A/total); bbox['B'].append(100.*b_B/total)
        bbox['C'].append(100.*b_C/total); bbox['Ens_Same'].append(100.*b_EnsS/total)
        
        tab3['White'].append(100.*w_EnsN/total)
        tab3['Black'].append(100.*b_EnsN/total)

    # ================= 严格对齐论文输出 =================

    print("\n\n" + "="*80)
    print("【表 1：不同模型在白盒FGSM攻击下的准确率】 (声明：所有模型均只使用 5000 张同源数据，绝对公平对比)")
    print(f"{'模型':<15} | {'0.00':<8} | {'0.05':<8} | {'0.10':<8} | {'0.15':<8} | {'0.20':<8} | {'0.25':<8} | {'0.30':<8}")
    print("-" * 80)
    print(f"{'基线A':<17} | " + " | ".join([f"{v:>6.2f}" for v in wbox['A']]))
    print(f"{'基线B':<17} | " + " | ".join([f"{v:>6.2f}" for v in wbox['B']]))
    print(f"{'基线C':<17} | " + " | ".join([f"{v:>6.2f}" for v in wbox['C']]))
    print(f"{'本文机制(同5k)':<13} | " + " | ".join([f"{v:>6.2f}" for v in wbox['Ens_Same']]))
    print("=" * 80)

    print("\n【表 2：不同模型在黑盒迁移攻击下的准确率】 (声明：所有模型均只使用 5000 张同源数据，绝对公平对比)")
    print(f"{'模型':<15} | {'0.00':<8} | {'0.05':<8} | {'0.10':<8} | {'0.15':<8} | {'0.20':<8} | {'0.25':<8} | {'0.30':<8}")
    print("-" * 80)
    print(f"{'基线A':<17} | " + " | ".join([f"{v:>6.2f}" for v in bbox['A']]))
    print(f"{'基线B':<17} | " + " | ".join([f"{v:>6.2f}" for v in bbox['B']]))
    print(f"{'基线C':<17} | " + " | ".join([f"{v:>6.2f}" for v in bbox['C']]))
    print(f"{'本文机制(同5k)':<13} | " + " | ".join([f"{v:>6.2f}" for v in bbox['Ens_Same']]))
    print("=" * 80)

    print("\n【表 3 (你论文中的横排表)：集成模型在白盒及黑盒攻击下的准确率对比】 (声明：此表专秀非交数据，吃满 25000 张)")
    print(f"{'测试环境':<15} | {'0.00':<8} | {'0.05':<8} | {'0.10':<8} | {'0.15':<8} | {'0.20':<8} | {'0.25':<8} | {'0.30':<8}")
    print("-" * 80)
    print(f"{'白盒攻击':<15} | " + " | ".join([f"{v:>6.2f}" for v in tab3['White']]))
    print(f"{'黑盒攻击':<15} | " + " | ".join([f"{v:>6.2f}" for v in tab3['Black']]))
    print("=" * 80)
    print("\n评测彻底结束！现在逻辑完全无懈可击，表1表2是同量级公平打架，表3是降维打击。快去跑吧！")