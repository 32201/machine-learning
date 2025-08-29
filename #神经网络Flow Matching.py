# Neural Network Flow Matching

# 搭建神经网络
import torch
import torch.nn as nn
import torch.optim as optim

# 绘图
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import griddata  # 插值

import random

mpl.rcParams.update({
    "pdf.fonttype": 42,     # 字体类型
    "ps.fonttype": 42,      # 字体类型
    "font.size": 24,        # 字体大小
    "axes.titlesize": 42,    # 子图标题大小
    "axes.labelsize": 42,    # x轴标签大小
    "xtick.labelsize": 12,   # x轴刻度大小
    "ytick.labelsize": 12,  
    "legend.fontsize": 12,  # 图例中字体大小
    "figure.titlesize": 20  # 整图标题大小
})

# 分配device
device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# 设置随机数种子
def set_seed(seed = 0):
    random.seed(seed)   # python内置random库随机数种子
    np.random.seed(seed)    # numpy库随机数种子
    torch.manual_seed(seed) # torch库cpu随机数种子
    torch.cuda.manual_seed(seed)    # torch库当前gpu随机数种子
    torch.cuda.manual_seed_all(seed)    # torch库所有gpu随机数种子
    torch.backends.cudnn.deterministic = True   # 确保cudnn使用确定性算法
    torch.backends.cudnn.benchmark = False  # 关闭cudnn的基准优化模式
    print(f'set seed to {seed}')

# 精确解函数
def u_exact_solution(X):    # X: (N, 2) tensor
    x = X[: ,0]
    y = X[: ,1]
    u = 30 * x * (1 - x) * y * (1 - y)  # u: (N, ) tensor
    return u.unsqueeze(1)   # u.unsqueeze(1): (N, 1) tensor

# 生成网格点
def generate_grid_point(num = 200, start = 0.0, end = 1.0, device = device):
    x = np.linspace(start, end, num)
    y = np.linspace(start, end, num)
    xx, yy = np.meshgrid(x, y, indexing = 'ij')
    xx.shape = (num * num, 1)
    yy.shape = (num * num, 1)
    X = np.concatenate((xx, yy), axis = 1)  # X: (num * num, 2)
    return torch.tensor(X, dtype = torch.float32, device = device)  # (num * num, 2) tensor

# 噪声函数
def add_noise(u, noisy_noise = 0.5, clean_noise = 0.01, noisy_ratio = 0.2, device = device):   # u: (N, 1) tensor
    N = u.shape[0]  #数据个数
    random_index = torch.randperm(N, device = device)   # 随机索引
    noisy_N = int(noisy_ratio * N)
    noisy_index = random_index[: noisy_N]
    clean_index = random_index[noisy_N :]
    
    noise = torch.randn((N, 1), device = device)    # 噪声
    u[noisy_index] += noise[noisy_index] * noisy_noise
    u[clean_index] += noise[clean_index] * clean_noise
    return u

set_seed(0) # 设置随机数种子

# 训练集
X_train_data = generate_grid_point(15, 0.125, 0.875)    # [0.125, 0.875]^2上均匀225个网格点
u_train_data = add_noise(u_exact_solution(X_train_data), noisy_noise = 0., clean_noise = 0., noisy_ratio = 0.2)    # 带噪声的u

# 神经网络
class ConditionalVectorField(nn.Module):
    def __init__(self, input_dim = 1 + 1 + 2, hidden_layers = 5, hidden_dim = 100, output_dim = 1):   # 输入(u, t, X)，输出v_pred
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]  # 输入层到第一隐藏层
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]    # 隐藏层
        layers += [nn.Linear(hidden_dim, output_dim)]   # 隐藏层到输出层
        self.net = nn.Sequential(*layers)

    def forward(self, u, t, X): #前向传播
        return self.net(torch.cat([u, t, X], dim = 1))

model = ConditionalVectorField().to(device) # 生成神经网络实例
optimizer = optim.Adam(model.parameters(), lr = 1e-3)   # 优化器，优化全体参数，设置误差

epochs = 10000  # 训练轮数
batch_size = X_train_data.shape[0]  # 批量训练样本，这里为全训练
print_every = 500

# 训练模型
def fm_train_function(model, optimizer, X_batch, u_batch, device = device):
    model.train()   # 训练模式

    N = X_batch.shape[0]    # 批量训练样本个数
    u0 = torch.randn((N, 1), device = device)   # 初始正态分布采样
    t = torch.rand((N, 1), device = device) # 随机时间张量

    ut = (1 - t) * u0 + t * u_batch   # 线性路径插值

    v_target = u_batch - u0
    v_pred = model(ut, t, X_batch)

    loss = torch.mean((v_pred - v_target) ** 2) # 均方误差

    optimizer.zero_grad()   # 清空梯度
    loss.backward() # 反向传播
    optimizer.step()    # 更新参数
    return loss.item()

print("Start FM training (data only, no PDE residual) ...")
for epoch in range(1, epochs + 1):
    loss = fm_train_function(model, optimizer, X_train_data, u_train_data, device = device)
    if epoch % print_every == 0 or epoch == 1:
        print(f"[Epoch {epoch:5d}] FM loss: {loss:.6e}")
print("Training completed.")

# 预测函数
def pred_funtion(model, X_to_predict, steps = 50, u0_value = 0.0, device = device):
    model.eval()    # 评估模式

    N = X_to_predict.shape[0]   # 预测数据个数
    dt = 1.0 / steps    # 步长

    with torch.no_grad():

        u = torch.full((N, 1), float(u0_value), device = device)
        t = torch.zeros((N, 1), device = device)

        for _ in range(steps):
            v = model(u, t, X_to_predict)
            u += dt * v
            t += dt

    return u

# 模型评估
def evaluate_model(model, N_test = 10000, device = device):
    X_test = torch.rand((N_test, 2), device = device)    # [0, 1]^2上随机采样10000个点
    u_pred = pred_funtion(model, X_test, steps = 50, u0_value = 0.0)
    u_exact = u_exact_solution(X_test)
    rel_l2_err = torch.norm(u_pred - u_exact, 2) / torch.norm(u_exact, 2)
    print(f'Relative L2 error on random test set: {rel_l2_err.item():.6f}')

# 绘图
def plot_solution(model, X_train_data, u_train_data, resolution = 200, device = device, save_pdf=True, pdf_path="no_noise.pdf"):

    # 绘图数据
    X_draw_data = generate_grid_point(num = resolution, start = 0, end = 1, device = device)   # 网格点
    X = X_draw_data[: ,0].view(resolution, resolution).detach().cpu().numpy()  #X: (resolution, resolution) numpy
    Y = X_draw_data[: ,1].view(resolution, resolution).detach().cpu().numpy()  #Y: (resolution, resolution) numpy
    u_exact_data = u_exact_solution(X_draw_data).view(resolution, resolution).detach().cpu().numpy()    # 真值: (resolution,resolution) numpy
    with torch.no_grad():
        u_pred_data = pred_funtion(model, X_draw_data, steps = 200, device = device).view(resolution, resolution).detach().cpu().numpy()    # 预测值: (resolution, resolution) numpy
    error = np.abs(u_exact_data - u_pred_data)

    # 插值
    points = X_train_data.detach().cpu().numpy()    # (X_train_data.shape[0], 2)
    values = u_train_data.detach().cpu().numpy().reshape(-1)    # (u_train_data.shape[0], )
    u_linear = griddata(points, values, (X, Y), method = 'linear')  #线性插值
    u_near = griddata(points, values, (X, Y), method = 'nearest')   #最近邻插值
    u_train = np.where(np.isnan(u_linear), u_near, u_linear)    #补洞

    # 2×2子图
    fig, ax = plt.subplots(2, 2, figsize = (18, 10))

    # 图1: 训练点集
    im00 = ax[0, 0].contourf(X, Y, u_train, 50, cmap = 'viridis')
    ax[0, 0].set_title('Training Points')
    fig.colorbar(im00, ax = ax[0, 0])

    # 图2: 真解
    im01 = ax[0, 1].contourf(X, Y, u_exact_data, 50, cmap = 'viridis')
    ax[0,1].set_title("Exact Solution")
    fig.colorbar(im01, ax=ax[0,1])

    # 图3: 预测解
    im10 = ax[1,0].contourf(X, Y, u_pred_data, 50, cmap='viridis')
    ax[1,0].set_title("FM Predicted Solution")
    fig.colorbar(im10, ax=ax[1,0])

    # 图4: 绝对误差
    im11 = ax[1,1].contourf(X, Y, error, 50, cmap='viridis')
    ax[1,1].set_title("Absolute Error")
    fig.colorbar(im11, ax=ax[1,1])

    for a in ax.ravel():
        a.set_xlim(0,1); a.set_ylim(0,1)

    plt.tight_layout()

    if save_pdf:
        plt.savefig(pdf_path, format="pdf", bbox_inches="tight")
        print(f"[PDF] Saved figure to: {pdf_path}")
        
    plt.show()

evaluate_model(model, N_test = 10000, device = device)
plot_solution(model, X_train_data, u_train_data, resolution = 200)