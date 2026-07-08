import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset.utils import read_client_data
from model import create_model


class FedVRUClient:
    def __init__(self, args, client_id):
        # 必须先初始化 device 属性
        self.device = args.device
        self.experiment_name = args.experiment_name
        self.client_id = client_id
        self.base_data_dir = args.base_data_dir
        self.dataset = args.dataset
        self.train_samples = 0
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size

        # 创建本地模型并移至设备
        self.model = create_model(args.dataset).to(self.device)
        self.global_model = copy.deepcopy(self.model)
        for p in self.global_model.parameters():
            p.requires_grad = False

        self.cls_loss_func = nn.CrossEntropyLoss().to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.train_loader = self.load_train_data()
        self.test_loader = self.load_test_data()

    def set_model(self, gfe):
        """同步全局 GFE 参数"""
        self.model.gfe.load_state_dict(gfe.state_dict())
        self.global_model.gfe.load_state_dict(gfe.state_dict())

    def calculate_kl_loss(self, mean, logvar):
        # 学术修正：对维度取平均，防止数值过大
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)
        return kl_loss.mean()

    def calculate_contrastive_loss(self, gf_s, gf_t, cf_s, temperature=0.2):
        """
        gf_s: 学生全局投影, gf_t: 教师全局投影, cf_s: 学生本地投影
        """
        batch_size = gf_s.shape[0]
        gf_s = F.normalize(gf_s, dim=1)
        gf_t = F.normalize(gf_t, dim=1)
        cf_s = F.normalize(cf_s, dim=1)

        # 1. 正样本：学生GF vs 教师GF (一致性)
        pos_sim = torch.exp(torch.sum(gf_s * gf_t, dim=1) / temperature)

        # 2. 负样本 A：学生GF vs 学生CF (解耦)
        neg_sim_internal = torch.exp(torch.sum(gf_s * cf_s, dim=1) / temperature)

        # 3. 负样本 B：Batch 内其他样本 (增强鲁棒性)
        all_sim = torch.mm(gf_s, gf_t.t()) / temperature
        mask = torch.eye(batch_size).to(self.device)
        neg_sim_external = (torch.exp(all_sim) * (1 - mask)).sum(dim=1)

        loss = -torch.log(pos_sim / (pos_sim + neg_sim_internal + neg_sim_external))
        return loss.mean()

    def train_classification_with_distillation(self):
        self.model.train()
        # 论文建议权重：对比损失 lambda=0.5, KL lambda=0.01
        l_con, l_kl = 0.5, 0.1

        for epoch in range(self.local_epochs):
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.opt.zero_grad()

                # 前向传播
                logits, gf_s_p, cf_s_p, mean, logvar = self.model.classification(x)

                # 教师特征提取
                with torch.no_grad():
                    gf_t_raw = self.global_model.gfe(x)
                    # 教师特征通过当前学生的投影头映射到同一空间
                    gf_t_p = self.model.g_proj(gf_t_raw)

                # 损失函数
                loss_cls = self.cls_loss_func(logits, y)
                loss_con = self.calculate_contrastive_loss(gf_s_p, gf_t_p, cf_s_p)
                loss_kl = self.calculate_kl_loss(mean, logvar)

                total_loss = loss_cls + l_con * loss_con + l_kl * loss_kl
                total_loss.backward()
                self.opt.step()

    def train(self):
        self.train_classification_with_distillation()

    def test_metrics(self):
        self.model.eval()
        test_corrects, test_loss, samples = 0, 0, 0
        with torch.no_grad():
            for x, y in self.test_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits, _, _, _, _ = self.model.classification(x)
                test_corrects += (torch.argmax(logits, dim=1) == y).sum().item()
                test_loss += self.cls_loss_func(logits, y).item() * y.size(0)
                samples += y.size(0)
        return {"test_num_samples": samples, "test_corrects": test_corrects, "test_cls_loss": test_loss}

    def train_metrics(self):
        self.model.eval()
        train_corrects, train_loss, samples = 0, 0, 0
        with torch.no_grad():
            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits, _, _, _, _ = self.model.classification(x)
                train_corrects += (torch.argmax(logits, dim=1) == y).sum().item()
                train_loss += self.cls_loss_func(logits, y).item() * y.size(0)
                samples += y.size(0)
        return {"train_num_samples": samples, "train_corrects": train_corrects, "train_cls_loss": train_loss}

    def load_train_data(self):
        train_data = read_client_data(self.base_data_dir, self.dataset, self.experiment_name, self.client_id,
                                      is_train=True)
        self.train_samples = len(train_data)
        return DataLoader(train_data, self.batch_size, drop_last=True, shuffle=True)

    def load_test_data(self):
        test_data = read_client_data(self.base_data_dir, self.dataset, self.experiment_name, self.client_id,
                                     is_train=False)
        return DataLoader(test_data, self.batch_size, drop_last=False, shuffle=False)




