import h5py
import numpy as np
from pathlib import Path
import sys


def load_h5_file(file_path):
    """
    读取指定的 .h5 文件，返回包含所有数据（如输入特征、预测值、潜变量等）的字典。

    参数:
        file_path (str or Path): .h5 文件的完整路径

    返回:
        dict: 包含所有读取数据的字典
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件未找到: {path}")

    data_dict = {}

    print(f"正在读取文件: {path.name}")
    with h5py.File(path, "r") as f:
        # 1. 读取属性 (Metadata)
        if hasattr(f, "attrs"):
            data_dict["attrs"] = dict(f.attrs)

        # 2. 读取所有数据集 (Datasets)
        keys = list(f.keys())
        print(f"包含数据集: {keys}")

        for key in keys:
            # 将数据加载到内存中 (numpy array)
            vals = f[key][()]

            # 处理字符串类型的标签 (解码 bytes 为 str)
            if key == "label":
                if vals.size > 0 and isinstance(vals.flatten()[0], (bytes, np.bytes_)):
                    vals = np.array([v.decode("utf-8") for v in vals])

            data_dict[key] = vals

            # 打印简要信息
            shape_info = vals.shape if hasattr(vals, "shape") else "scalar"
            dtype_info = vals.dtype if hasattr(vals, "dtype") else type(vals)
            print(f"  - Loaded '{key}': type={dtype_info}, shape={shape_info}")

    return data_dict


if __name__ == "__main__":
    # ================= 配置区域 =================
    # 请在这里修改为您想要读取的 H5 文件的实际路径
    # 例如: r"D:\path\to\your\Exp2_test_cached.h5"
    target_h5_path = (
        r"D:\PersonFiles\Codes\Project\MACNet v1\Results\Exp2_ViT20260130_223957\prediction_data\Exp2_test_cached.h5"
    )
    # ===========================================

    # 允许通过命令行参数覆盖路径
    if len(sys.argv) > 1:
        target_h5_path = sys.argv[1]

    try:
        # 1. 加载数据
        data = load_h5_file(target_h5_path)

        print("\n" + "=" * 30)
        print("读取完成，数据概览:")
        print("=" * 30)

        # 2. 演示：获取潜变量 (Latent Representation)
        if "latent" in data:
            latent_vector = data["latent"]
            print(f"\n[潜变量] 'latent' 提取成功!")
            print(f"  > 形状: {latent_vector.shape}")
            print(f"  > 数据类型: {latent_vector.dtype}")
            print(f"  > 前2行示例:\n{latent_vector[:2]}")
        else:
            print("\n[潜变量] 未找到 'latent' 数据集。")

        # 3. 演示：获取预测值与真实值
        if "y_pred" in data and "y_true" in data:
            y_pred = data["y_pred"]
            y_true = data["y_true"]
            print(f"\n[预测结果] 提取成功!")
            print(f"  > 真实值形状: {y_true.shape}")
            print(f"  > 预测值形状: {y_pred.shape}")

        # 4. 演示：获取原始输入
        if "x" in data:
            x_data = data["x"]
            print(f"\n[输入特征] 提取成功!")
            print(f"  > 形状: {x_data.shape} (N, C, H, W)")

    except Exception as e:
        print(f"\n[错误] 读取文件失败: {e}")
        print(f"请检查路径: {target_h5_path}")
