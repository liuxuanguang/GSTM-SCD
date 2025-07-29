# test_multiutae.py
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from multiutae import MultiUTAE
import torch

def test_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建模型实例
    model = MultiUTAE(
        input_dim=4,
        num_classes=8,
        in_features=512,
        T=2  # 时间周期
    ).to(device)

    # 减小输入分辨率，节省显存
    h, w = 512, 512
    
    # 输入数据
    input_data = torch.randn(1, 2, 4, h, w).to(device)  # [batch, time, channels, height, width]

    # 位置信息
    time_positions = torch.arange(1, 3).to(device)
    batch_positions = time_positions[None, :, None, None]

    batch = {
        "data": input_data,
        "positions": batch_positions
    }

    # 前向传播
    try:
        output = model(batch)
        print("输出形状:", output["logits"].shape)
        print("测试成功!")
    except Exception as e:
        print(f"测试失败: {str(e)}")
        raise e

    # 计算FLOPs
    from thop import profile

    FLOPs, Params = profile(model, inputs=(batch,))
    print('Params = %.2f M, FLOPs = %.2f G' % (Params / 1e6, FLOPs / 1e9))

if __name__ == "__main__":
    test_model()