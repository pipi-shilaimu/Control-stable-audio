import torch
import sys

print("=" * 60)
print("PyTorch GPU 支持检查")
print("=" * 60)

# 1. 基本信息
print("1. PyTorch 版本信息:")
print(f"   PyTorch 版本: {torch.__version__}")
print(f"   Torchvision 版本: {torch.__version__}")

# 2. CUDA 支持
print("\n2. CUDA 支持情况:")
print(f"   CUDA 是否可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"   CUDA 版本: {torch.version.cuda}")
    print(f"   支持的CUDA架构: {torch._C._cuda_getArchFlags()}")
else:
    print("   ✗ CUDA 不可用")
    sys.exit(1)

# 3. GPU 信息
print("\n3. GPU 详细信息:")
gpu_count = torch.cuda.device_count()
print(f"   检测到的GPU数量: {gpu_count}")

if gpu_count > 0:
    for i in range(gpu_count):
        print(f"\n   GPU {i}:")
        print(f"     名称: {torch.cuda.get_device_name(i)}")
        prop = torch.cuda.get_device_properties(i)
        print(f"     显存总量: {prop.total_memory / 1e9:.2f} GB")
        print(f"     已用显存: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
        print(f"     预留显存: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
        print(f"     计算能力: {prop.major}.{prop.minor}")
        print(f"     Multiprocessors: {prop.multi_processor_count}")
        
        # 当前设备
        if i == torch.cuda.current_device():
            print(f"     ✓ 这是当前设备")

# 4. 计算测试
print("\n4. GPU 计算测试:")
if torch.cuda.is_available():
    try:
        # 创建张量
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')
        
        # 矩阵乘法
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        z = torch.matmul(x, y)
        end_time.record()
        
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time)
        
        print(f"   ✓ GPU计算测试通过")
        print(f"   计算时间: {elapsed_time:.2f} ms")
        print(f"   结果验证: 矩阵形状 {z.shape}, 数据类型 {z.dtype}")
        
        # 内存测试
        del x, y, z
        torch.cuda.empty_cache()
        print(f"   ✓ 内存清理完成")
        
    except Exception as e:
        print(f"   ✗ GPU计算测试失败: {e}")
else:
    print("   ⚠ 跳过GPU计算测试（CUDA不可用）")

# 5. CUDA 功能检查
print("\n5. CUDA 功能支持:")
if torch.cuda.is_available():
    print(f"   cuDNN 可用: {torch.backends.cudnn.is_available()}")
    print(f"   cuDNN 版本: {torch.backends.cudnn.version() if hasattr(torch.backends.cudnn, 'version') else 'N/A'}")
    print(f"   cuDNN 启用: {torch.backends.cudnn.enabled}")
    print(f"   AMP 支持: {hasattr(torch, 'amp')}")
    print(f"   Tensor cores 支持: {torch.cuda.get_device_properties(0).major >= 7}")

# 6. 环境变量
print("\n6. 相关环境变量:")
import os
env_vars = ['CUDA_VISIBLE_DEVICES', 'CUDA_HOME', 'LD_LIBRARY_PATH', 'NVIDIA_VISIBLE_DEVICES']
for var in env_vars:
    value = os.environ.get(var, '未设置')
    print(f"   {var}: {value}")

print("\n" + "=" * 60)
print("总结:")
if torch.cuda.is_available() and gpu_count > 0:
    print(f"✓ PyTorch GPU支持正常！")
    print(f"  可用GPU: {gpu_count}个")
    for i in range(gpu_count):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("✗ PyTorch GPU支持有问题")
    print("  可能原因:")
    print("  1. PyTorch未安装CUDA版本")
    print("  2. NVIDIA驱动未安装")
    print("  3. CUDA工具包未安装")
    print("  4. Docker未正确配置GPU支持")
print("=" * 60)