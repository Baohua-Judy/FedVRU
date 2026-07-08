import argparse
import copy
import torch
import time
import numpy as np
import os
import csv 
from client import FedVRUClient


class FedVRUServer:
    def __init__(self, args):
        self.args = args 
        self.global_epochs = args.global_epochs
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.eval_interval = args.eval_interval
        self.args.partition = args.partition

        # 初始化客户端
        self.clients = []
        for i in range(self.num_clients):
            client = FedVRUClient(args, client_id=i)
            self.clients.append(client)

        # 全局特征提取器
        self.gfe = None
        self.uploaded_weights = []
        self.uploaded_models = []
        self.selected_clients = []
        self.best_test_acc = 0
        self.Budget = []
        self.results = []

    def send_models(self):
        """分发全局模型到所有客户端"""
        assert len(self.clients) > 0
        for client in self.clients:
            client.set_model(copy.deepcopy(self.gfe))

    def add_parameters(self, w, client_model):
        """加权聚合客户端模型参数"""
        for server_param, client_param in zip(self.gfe.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def aggregate_parameters(self):
        """聚合客户端上传的模型参数"""
        assert len(self.uploaded_models) > 0

        # 初始化全局模型
        self.gfe = copy.deepcopy(self.uploaded_models[0])
        for param in self.gfe.parameters():
            param.data.zero_()

        # 加权平均聚合
        total_samples = sum(self.uploaded_weights)
        for w, model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w / total_samples, model)

    def run(self):
        """运行联邦学习主流程"""
        print(f"\n▶ Starting Federal Training with {self.num_clients} clients...")

        for epoch in range(self.global_epochs):
            start_time = time.time()
            self.selected_clients = self.select_clients()

            # 定期评估模型性能
            if epoch % self.eval_interval == 0:
                print(f"\n🔁 Round {epoch + 1}/{self.global_epochs} {'-' * 30}")
                self.evaluate(epoch)

            # 客户端本地训练
            for client in self.selected_clients:
                client.train()

            # 模型聚合与分发
            self.receive_models()
            self.aggregate_parameters()
            self.send_models()

            # 记录时间消耗
            round_time = time.time() - start_time
            self.Budget.append(round_time)
            print(f'⏱️ Round time cost: {round_time:.2f}s')

        # 最终报告
        print("\n✅ Training completed!")
        print(f"🏆 Best test accuracy: {self.best_test_acc:.2%}")
        print(f"🕰️ Average time per round: {np.mean(self.Budget):.2f}s")

        # 保存结果到 CSV
        self.save_results_to_csv()

    def receive_models(self):
        """接收客户端上传的模型"""
        assert len(self.selected_clients) > 0

        total_samples = sum([c.train_samples for c in self.selected_clients])
        self.uploaded_weights = []
        self.uploaded_models = []

        for client in self.selected_clients:
            self.uploaded_weights.append(client.train_samples)
            # 仅上传 GFE 部分
            self.uploaded_models.append(copy.deepcopy(client.model.gfe))

    def test_metrics(self):
        """收集所有客户端的测试指标"""
        num_samples = []
        corrects = []
        losses = []

        for client in self.clients:
            stats = client.test_metrics()
            num_samples.append(stats["test_num_samples"])
            corrects.append(stats["test_corrects"])
            losses.append(stats["test_cls_loss"])
        return num_samples, corrects, losses

    def train_metrics(self):
        """收集训练指标"""
        num_samples = []
        corrects = []
        losses = []

        for client in self.clients:
            stats = client.train_metrics()
            num_samples.append(stats["train_num_samples"])
            corrects.append(stats["train_corrects"])
            losses.append(stats["train_cls_loss"])
        return num_samples, corrects, losses

    def evaluate(self, epoch):
        """评估模型性能并记录结果"""
        test_samples, test_corrects, test_losses = self.test_metrics()
        train_samples, train_corrects, train_losses = self.train_metrics()

        test_acc = sum(test_corrects) / sum(test_samples)
        test_loss = sum(test_losses) / sum(test_samples)
        train_acc = sum(train_corrects) / sum(train_samples)
        train_loss = sum(train_losses) / sum(train_samples)

        if test_acc > self.best_test_acc:
            self.best_test_acc = test_acc

        client_accs = [c / s for c, s in zip(test_corrects, test_samples)]
        acc_std = np.std(client_accs)

        print(f"\n📊 Evaluation results:")
        print(f"Train › Loss: {train_loss:.4f}  Acc: {train_acc:.2%}")
        print(f"Test  › Loss: {test_loss:.4f}  Acc: {test_acc:.2%} ±{acc_std:.4f}")

        formatted_train_acc = f"{train_acc * 100:.2f}%"
        formatted_test_acc_with_std = f"{test_acc * 100:.2f}% ±{acc_std:.4f}"
        self.results.append({
            "epoch": epoch,
            "train_acc": formatted_train_acc,
            "train_loss": f"{train_loss:.4f}",
            "test_acc": formatted_test_acc_with_std,
            "test_loss": f"{test_loss:.4f}"
        })

    def save_results_to_csv(self):
        filename = (
            f"{self.args.dataset}_"
            f"clients{self.args.num_clients}_"
            f"ratio{self.args.join_ratio}_"
            f"alpha{self.args.alpha}_"
            f"partition{self.args.partition}_"
            f"test{self.args.test}.csv"
        )
        save_dir = r"D:\A简化代码\FedVRU-xiugai\result"
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, filename)
        with open(filepath, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.results[0].keys())
            writer.writeheader()
            writer.writerows(self.results)
        print(f"💾 Saved results to: {filepath}")

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.randint(
                self.num_join_clients, self.num_clients + 1
            )
        else:
            self.current_num_join_clients = self.num_join_clients
        return np.random.choice(self.clients, self.current_num_join_clients, replace=False).tolist()