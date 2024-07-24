# import numpy as np
# def lerp_np(x,y,w):
#     fin_out = (y-x)*w + x
#     return fin_out
# import math
# def rand_perlin_2d_np(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):  #shape (256 256) res (16,2))
#     # 计算网格步长 delta 和每个网格块的大小 d
#     delta = (res[0] / shape[0], res[1] / shape[1]) #(1/16,1/128)
#     d = (shape[0] // res[0], shape[1] // res[1])  #(16,128)
#     # 生成网格坐标 grid，其中每个点的坐标值在0到1之间
#     grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1   #delta 为间隔 0:res[0]为上下界。 (256,256,2)
#
#     #  生成(17, 3)个随机角度
#     angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)    #(17,3)
#     # 据角度计算出对应的二维梯度向量
#     gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)  #(17,3,2)
#
#     # 重复梯度向量以匹配网格大小
#     tt = np.repeat(np.repeat(gradients,d[0],axis=0),d[1],axis=1) # (272,384,2)
#
#     # tile_grads是一个lambda函数
#     # slice1和slice2: 两个元组,表示要从梯度矩阵gradients中截取的行列范围
#     tile_grads = lambda slice1, slice2: np.repeat(np.repeat(gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]],d[0],axis=0),d[1],axis=1)
#     # 计算网格坐标与梯度向量的点积
#     # grad: 一个2D梯度矩阵,其中每个元素是一个2D梯度向量
#     # shift: 一个长度为2的NumPy数组,表示要对网格坐标进行的位移
#     dot = lambda grad, shift: (
#                 np.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
#                             axis=-1) * grad[:shape[0], :shape[1]]).sum(axis=-1)
#
#     n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0]) #(256,256)
#     n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
#     n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
#     n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
#     t = fade(grid[:shape[0], :shape[1]]) #(256,256,2)
#     return math.sqrt(2) * lerp_np(lerp_np(n00, n10, t[..., 0]), lerp_np(n01, n11, t[..., 0]), t[..., 1]) #(256,256)
#
# print(1==0)
#
# dict = {"PG 58-22":10,
#         "PG 72-50":101
#         }
#
#
# fr = open('train.csv', 'r')
# frw = open('train_1.csv','w')
#
# line = fr.readline()
# inputs = []
# for L in line:
#     strs = L.strip("\n").split(",")
#     input = []
#     for str in strs:
#         if(isinstance(str, (float))):
#             input.append(float(str))
#         else:
#             input.append(dict[str])
#     inputs.append(input)


import torch
import torch.nn as nn
import numpy as np

x = np.arange(8).reshape(2, 4)
np.expand_dims(x,0)

print(x.shape)
# 假设输入 tensor 的形状为 (batch, channels, height, width)
input_tensor = torch.tensor([[[[0,0],[0,0]]],
                             [[[1,1],[1,1]]],
                             [[[2,2],[2,2]]],
                             [[[3,3],[3,3]]],
                             [[[4,4],[4,4]]],
                             [[[5,5],[5,5]]],
                             [[[6,6],[6,6]]],
                             [[[7,7],[7,7]]],
                             [[[8,8],[8,8]]],
                             [[[9,9],[9,9]]],
                             [[[10,10],[10,10]]],
                             [[[11,11],[11,11]]],
                             [[[12,12],[12,12]]],
                             [[[13,13],[13,13]]],
                             [[[14,14],[14,14]]],
                             [[[15,15],[15,15]]]])
input_tensor = input_tensor.permute(1,0,2,3).squeeze()
print(input_tensor.shape)

# 使用 pixel_shuffle 进行上采样，unpixel_shuffle就是通道变多，尺寸变小
upscale_factor = 2
output_channels = 16 // (upscale_factor ** 2)
output_tensor = nn.functional.pixel_shuffle(input_tensor, upscale_factor)

# output_tensor 的形状为 (batch, output_channels, height*upscale_factor, width*upscale_factor)
print(output_tensor.shape)  # 输出: torch.Size([1, 16, 64, 64])

print(input_tensor)
print(output_tensor)