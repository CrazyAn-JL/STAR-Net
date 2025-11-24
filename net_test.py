from thop import profile
import torch
import time
from net.Net import Net


import torch
import time
from thop import profile

device = "cuda"

# =====================
# 1. 初始化模型
# =====================
model = Net().to(device)
model.eval()

# =====================
# 2. 构造固定输入 (600×400)
# =====================
inp = torch.rand(1, 3, 400, 600).to(device)

# =====================
# 3. Warm-up 预热
# =====================
with torch.no_grad():
    for _ in range(3):
        _ = model(inp)
torch.cuda.synchronize()

# =====================
# 4. 测试 15 次推理时间
# =====================
times = []

with torch.no_grad():
    for _ in range(15):
        torch.cuda.synchronize()
        t0 = time.time()

        _ = model(inp)

        torch.cuda.synchronize()
        t1 = time.time()
        times.append(t1 - t0)

avg_time = sum(times) / len(times)
print(f"Avg inference time (15 runs): {avg_time:.6f} sec")
print(f"FPS: {1/avg_time:.2f}")

# =====================
# 5. 参数量和 FLOPs
# =====================
macs, params = profile(model, inputs=(inp,))
print(f"Params: {params / 1e6:.2f}M")
print(f"FLOPs: {macs / 1e9:.2f}G")



# import torch
# import time
# from net.Net import Net
#
# # 确保模型和输入都在同一个设备上
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = Net().to(device)
# input = torch.rand(1, 3, 256, 256).to(device)
# model.eval()
#
# # --- 预热阶段 ---
# # 运行一次模型以确保GPU和CUDA核都已加载
# print("Starting warm-up...")
# with torch.no_grad():
#     for _ in range(5): # 运行几次预热，确保更充分
#         _ = model(input)
#
# # --- 计时阶段 ---
# num_runs = 100 # 设置循环次数
# total_time = 0
#
# # 确保所有之前的操作都已完成
# if device == 'cuda':
#     torch.cuda.synchronize()
#
# print(f"Starting timing for {num_runs} runs...")
# with torch.no_grad():
#     for _ in range(num_runs):
#         time_start = time.time()
#         _ = model(input)
#         if device == 'cuda':
#             torch.cuda.synchronize()
#         time_end = time.time()
#         total_time += (time_end - time_start)
#
# # 计算平均时间
# average_time = total_time / num_runs
#
# print(f"Average time over {num_runs} runs: {average_time:.6f} seconds")