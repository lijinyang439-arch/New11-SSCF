import torch

# 替换成您的实际文件路径
path = '/data/lijinyang/1_sleep/DG1/results/sleepdg_original_run_2025-11-07_21-41-16/fold0/best_test_ep62_tacc_0.82745_tf1_0.76698.pth'

# 注意：如果上次的 'weights_only' 错误没有解决，您可能需要先加 weights_only=False
checkpoint = torch.load(path, map_location='cpu', weights_only=False) # 加载到CPU以防显存不足

print(f"加载的类型: {type(checkpoint)}")
if isinstance(checkpoint, dict):
    print(f"字典中的所有键: {checkpoint.keys()}")
elif isinstance(checkpoint, torch.nn.Module):
    print("内容是一个直接的模型对象，这很少见。")
else:
    # 如果不是字典，它很可能就是直接的 state_dict
    print(f"假设它是直接的 state_dict。前5个键: {list(checkpoint.keys())[:5]}")