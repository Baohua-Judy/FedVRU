import torch
import torch.nn as nn
import torch.nn.functional as F


# 全局特征提取器 (Global Feature Extractor)
class GFE(nn.Module):
    def __init__(self, in_channels=1, hidden_size=1024, embed_dim=512):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, embed_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        z = self.convs(x)
        z = z.view(x.size(0), -1)
        return self.fc(z)


# 客户端特定特征提取器 (Client-Specific Feature Extractor)
class CSFE(nn.Module):
    def __init__(self, in_channels=1, hidden_size=1024, embed_dim=512):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, embed_dim * 2),  # 输出 mean 和 logvar
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        z = self.convs(x)
        z = z.view(x.size(0), -1)
        return self.fc(z)


# 联邦学习核心模型
class FedVRUModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, hidden_size=1024, embed_dim=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.gfe = GFE(in_channels, hidden_size, embed_dim)
        self.csfe = CSFE(in_channels, hidden_size, embed_dim)

        # 创新点：持久化的投影头（用于对比学习空间映射）
        self.g_proj = nn.Sequential(nn.Linear(embed_dim, 256), nn.ReLU(), nn.Linear(256, 128))
        self.c_proj = nn.Sequential(nn.Linear(embed_dim, 256), nn.ReLU(), nn.Linear(256, 128))

        self.phead = nn.Linear(embed_dim * 2, num_classes)

    def reparameterize(self, mean, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mean + eps * std
        return mean

    def classification(self, x):
        # 1. 获取全局特征及其投影
        gf = self.gfe(x)
        gf_proj = self.g_proj(gf)

        # 2. 获取客户端特征及其投影 (VAE 采样)
        z_c = self.csfe(x)
        mean, logvar = torch.split(z_c, self.embed_dim, dim=1)
        cf = self.reparameterize(mean, logvar)
        cf_proj = self.c_proj(cf)

        # 3. 不确定性感知门控 (Uncertainty-Aware Gating)
        uncertainty = logvar.mean(dim=1, keepdim=True)
        gate = torch.sigmoid(-uncertainty)
        weighted_cf = cf * gate

        # 4. 拼接并分类
        combined_feat = torch.cat([gf, weighted_cf], dim=1)
        logits = self.phead(combined_feat)

        return logits, gf_proj, cf_proj, mean, logvar


# 工厂函数
def create_model(dataset):
    configs = {
        "MNIST": (1, 10, 1024),
        "FashionMNIST": (1, 10, 1024),
        "Cifar10": (3, 10, 1600),
        "Cifar100": (3, 100, 1600),
        "OfficeCaltech10": (3, 10, 10816),
        "DomainNet": (3, 10, 10816)
    }
    if dataset not in configs:
        raise ValueError(f"Unsupported dataset: {dataset}")
    in_c, num_c, hid_s = configs[dataset]
    return FedVRUModel(in_c, num_c, hid_s, 512)