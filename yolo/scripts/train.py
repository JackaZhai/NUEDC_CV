#!/usr/bin/env python3
"""
YOLOv11 训练脚本

功能: 使用YOLO模型进行目标检测训练
使用方法: python scripts/train.py --data data/dataset.yaml --model yolo11n.pt --epochs 100

Author: JackZhai
Date: 2025-07-24
"""

import argparse
import os
import sys
import gc
from pathlib import Path

# 添加项目根目录到Python路径
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from ultralytics import YOLO
import torch

def parse_args():
    """
    解析命令行参数
    
    Returns:
        argparse.Namespace: 包含所有命令行参数的命名空间对象
    """
    parser = argparse.ArgumentParser(description='YOLOv11 训练脚本')
    
    # 数据和模型参数
    parser.add_argument('--data', type=str, default='data/dataset.yaml', 
                       help='数据集配置文件路径')
    parser.add_argument('--model', type=str, default='yolo11n.pt', 
                       help='预训练模型路径')
    
    # 训练超参数
    parser.add_argument('--epochs', type=int, default=100, 
                       help='训练轮数')
    parser.add_argument('--imgsz', type=int, default=640, 
                       help='输入图像尺寸')
    parser.add_argument('--batch-size', type=int, default=4, 
                       help='批次大小', dest='batch_size')
    parser.add_argument('--lr0', type=float, default=0.01, 
                       help='初始学习率')
    parser.add_argument('--weight-decay', type=float, default=0.0005, 
                       help='权重衰减', dest='weight_decay')
    
    # 硬件和性能参数
    parser.add_argument('--device', type=str, default='', 
                       help='训练设备 (cuda/cpu/mps)')
    parser.add_argument('--workers', type=int, default=0, 
                       help='数据加载线程数')
    
    # 输出和恢复参数
    parser.add_argument('--project', type=str, default='runs/train', 
                       help='保存结果的项目目录')
    parser.add_argument('--name', type=str, default='exp', 
                       help='实验名称')
    parser.add_argument('--resume', action='store_true', 
                       help='恢复训练')
    parser.add_argument('--amp', action='store_true', 
                       help='使用混合精度训练')
    
    return parser.parse_args()

def check_gpu_info(device):
    """
    检查并显示GPU信息，提供硬件相关建议
    
    Args:
        device (str): 指定的设备类型
        
    Returns:
        str: 实际使用的设备类型
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}")
    
    if torch.cuda.is_available():
        print(f"⚡ CUDA 版本: {torch.version.cuda}")
        print(f"🎮 GPU 数量: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            
            # 根据显存大小给出建议
            if gpu_memory < 6:
                print(f"   💡 显存较小，建议使用较小的批次大小 (batch-size=4)")
            elif gpu_memory < 8:
                print(f"   💡 显存中等，建议批次大小 (batch-size=8)")
            else:
                print(f"   💡 显存充足，可使用较大批次大小 (batch-size=16)")
    else:
        print("⚠️  CUDA不可用，将使用CPU训练")
    
    return device


def validate_dataset(data_path):
    """
    验证数据集配置文件是否存在
    
    Args:
        data_path (str): 数据集配置文件路径
        
    Returns:
        bool: 文件是否存在
    """
    if not os.path.exists(data_path):
        print(f"❌ 错误: 数据集配置文件 {data_path} 不存在")
        return False
    return True


def print_training_config(args, device):
    """
    打印训练配置信息
    
    Args:
        args: 命令行参数
        device (str): 训练设备
    """
    print("\n🔧 训练配置:")
    print(f"   📂 数据集: {args.data}")
    print(f"   🔄 训练轮数: {args.epochs}")
    print(f"   📦 批次大小: {args.batch_size}")
    print(f"   📏 图像尺寸: {args.imgsz}")
    print(f"   📈 学习率: {args.lr0}")
    print(f"   ⚖️  权重衰减: {args.weight_decay}")
    print(f"   🖥️  设备: {device}")
    print(f"   👷 工作线程: {args.workers}")
    print(f"   📁 输出目录: {args.project}/{args.name}")
    print(f"   ⚡ 混合精度: {args.amp}")


def clean_memory():
    """清理内存和GPU缓存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("   🧹 GPU内存清理完成")


def main():
    """
    主函数：执行模型训练的完整流程
    
    Returns:
        int: 程序退出码 (0: 成功, 1: 失败)
    """
    # 解析命令行参数
    args = parse_args()
    
    # 检查GPU信息并确定使用设备
    device = check_gpu_info(args.device)
    
    # 验证数据集配置文件
    if not validate_dataset(args.data):
        return 1
    
    # 创建输出目录
    os.makedirs(args.project, exist_ok=True)
    
    try:
        # 初始化模型
        print(f"\n🔄 加载模型: {args.model}")
        model = YOLO(args.model)
        
        # 显示训练配置
        print_training_config(args, device)
        
        # 预训练提示
        print(f"\n🚀 开始训练...")
        print("📝 注意: 首次训练时数据预处理可能需要几分钟时间，请耐心等待...")
        print("💡 使用保守配置避免训练中断...")
        print("   - 工作线程: 0 (避免多进程问题)")
        print("   - 批次大小: 4 (更稳定)")
        print("   - 关闭复杂数据增强")
        
        # 清理内存
        clean_memory()
        
        # 开始训练 - 使用稳定的配置参数
        print("⚡ 开始模型训练...")
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch_size,
            lr0=args.lr0,
            weight_decay=args.weight_decay,
            device=device,
            workers=0,           # 单线程数据加载，避免多进程问题
            project=args.project,
            name=args.name,
            resume=args.resume,
            amp=args.amp,
            save=True,
            save_period=5,       # 每5轮保存一次检查点
            cache=False,         # 关闭缓存以节省内存
            plots=False,         # 关闭图表生成以节省时间
            verbose=True,
            patience=20,         # 增加早停耐心值
            exist_ok=True,
        )
        
        # 训练完成信息
        print("\n🎉 训练完成!")
        print(f"📁 最佳模型保存在: {results.save_dir}/weights/best.pt")
        print(f"📁 最后模型保存在: {results.save_dir}/weights/last.pt")
        print(f"📊 训练结果保存在: {results.save_dir}")
        
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
        return 1
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("\n❌ GPU显存不足！")
            print("💡 建议:")
            print("   - 减小批次大小: --batch-size 4")
            print("   - 减小图像尺寸: --imgsz 416")
            print("   - 使用CPU训练: --device cpu")
        else:
            print(f"\n❌ 运行时错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
