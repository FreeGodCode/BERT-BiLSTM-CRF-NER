# # -*- coding: utf-8 -*-
# """
# @file: text_similar.py
# @author： ty
# @time: 2024/5/14 18:34
# """
# import numpy as np
#
# from bert_base.client import BertClient
#
#
# def cos_similar(sen_a_vec, sen_b_vec):
#     '''
#     计算两个句子的余弦相似度
#     '''
#     vector_a = np.mat(sen_a_vec)
#     vector_b = np.mat(sen_b_vec)
#     num = float(vector_a * vector_b.T)
#     denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
#     cos = num / denom
#     return cos
#
#
# def main():
#     bc = BertClient()
#     doc_vecs = bc.encode(['今天天空很蓝，阳光明媚', '今天天气好晴朗', '现在天气如何', '自然语言处理', '机器学习任务'])
#     # print(doc_vecs)
#     similarity = cos_similar(doc_vecs[0], doc_vecs[4])
#     print(similarity)
#
#
# if __name__ == '__main__':
#     main()
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

# train_data = torchvision.datasets.MNIST("data", train=True, transform=torchvision.transforms.ToTensor(), download=True)
# dataset = DataLoader(train_data, batch_size=4, shuffle=True, drop_last=True)
# for data in dataset:
#     image, target = data
#     print(image.shape)
#     print(image.target)
# vgg16 = torchvision.models.vgg16()
from torchvision.models import VGG16_Weights
# weights=VGG16_Weights.IMAGENET1K_V1
vgg16 = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
print(vgg16)
vgg16.add_module("add_linear", nn.Linear(1000, 10))  # 增加层
vgg16.classifier[6] = nn.Linear(4096, 10)  # 修改网络模型的层


# 模型保存结构+参数
torch.save(vgg16, "vgg16.pth")
# 加载
model1 = torch.load("vgg16.pth")

# 保存模型的参数，以字典的格式保存
torch.save(vgg16.state_dict(), "vgg16.pth")
# 加载
model2 = torch.load("vgg16.pth")
vgg16.load_state_dict(model2)
