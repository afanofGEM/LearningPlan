import torch
import torch.nn.functional as F

# 创建两张三通道的28*28的彩色图像
image = torch.randn(2,3,28,28)

'''定义卷积核，输出通道决定卷积核的组数，每组对应一张输出特征图
   输入通道决定每组卷积核的个数，
   每个卷积核与每个输入通道的输入图卷积运算，之后求和得到一张输出特征图'''
convolution_kernel = torch.nn.Conv2d(in_channels=3,out_channels=8,kernel_size=3,
                                     stride=1,padding=1)
# 卷积核组数：8 每组个数：3 每个尺寸：3*3

# 每张特征图尺寸:向上取整[(28+1*2-3) / 1] + 1 = 28
convolution_output = convolution_kernel(image)
print(convolution_output.shape)

'''ReLU核'''
relu = torch.nn.ReLU()
relu_output = relu(convolution_output)

'''定义池化层'''
pool_kernel = torch.nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
# 池化层并不会改变输入和输出通道数
# 输出尺寸 [(28+0*2-2) / 2] + 1 = 14
pool_output = pool_kernel(relu_output)
print(pool_output.shape)