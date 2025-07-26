#!/usr/bin/env python3
"""
数据集处理工具
包含数据集转换、预处理、增强等功能
"""

import os
import shutil
import random
from pathlib import Path
import xml.etree.ElementTree as ET
import json
import cv2
import numpy as np
from PIL import Image

def convert_voc_to_yolo(voc_dir, yolo_dir, class_names):
    """
    将VOC格式数据集转换为YOLO格式
    
    Args:
        voc_dir: VOC数据集目录
        yolo_dir: YOLO格式输出目录
        class_names: 类别名称列表
    """
    # 创建输出目录
    os.makedirs(f"{yolo_dir}/images/train", exist_ok=True)
    os.makedirs(f"{yolo_dir}/images/val", exist_ok=True)
    os.makedirs(f"{yolo_dir}/labels/train", exist_ok=True)
    os.makedirs(f"{yolo_dir}/labels/val", exist_ok=True)
    
    # 获取所有XML文件
    xml_files = list(Path(f"{voc_dir}/Annotations").glob("*.xml"))
    
    # 随机分割训练和验证集
    random.shuffle(xml_files)
    split_idx = int(len(xml_files) * 0.8)
    train_files = xml_files[:split_idx]
    val_files = xml_files[split_idx:]
    
    def convert_bbox(size, box):
        """转换边界框格式"""
        dw = 1.0 / size[0]
        dh = 1.0 / size[1]
        x = (box[0] + box[2]) / 2.0
        y = (box[1] + box[3]) / 2.0
        w = box[2] - box[0]
        h = box[3] - box[1]
        x = x * dw
        w = w * dw
        y = y * dh
        h = h * dh
        return (x, y, w, h)
    
    def process_files(files, split):
        """处理文件列表"""
        for xml_file in files:
            # 解析XML
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # 获取图像信息
            filename = root.find('filename').text
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            
            # 复制图像文件
            img_src = f"{voc_dir}/JPEGImages/{filename}"
            img_dst = f"{yolo_dir}/images/{split}/{filename}"
            if os.path.exists(img_src):
                shutil.copy2(img_src, img_dst)
            
            # 转换标注
            label_file = f"{yolo_dir}/labels/{split}/{xml_file.stem}.txt"
            with open(label_file, 'w') as f:
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    if class_name in class_names:
                        class_id = class_names.index(class_name)
                        bbox = obj.find('bndbox')
                        xmin = float(bbox.find('xmin').text)
                        ymin = float(bbox.find('ymin').text)
                        xmax = float(bbox.find('xmax').text)
                        ymax = float(bbox.find('ymax').text)
                        
                        bbox_yolo = convert_bbox((width, height), (xmin, ymin, xmax, ymax))
                        f.write(f"{class_id} {' '.join(map(str, bbox_yolo))}\n")
    
    process_files(train_files, "train")
    process_files(val_files, "val")
    
    print(f"转换完成! 训练集: {len(train_files)}, 验证集: {len(val_files)}")

def split_dataset(data_dir, train_ratio=0.8):
    """
    分割数据集为训练集和验证集
    
    Args:
        data_dir: 数据集目录
        train_ratio: 训练集比例
    """
    images_dir = Path(data_dir) / "images"
    labels_dir = Path(data_dir) / "labels"
    
    # 获取所有图像文件
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    
    # 随机分割
    random.shuffle(image_files)
    split_idx = int(len(image_files) * train_ratio)
    
    # 创建目录
    for split in ["train", "val"]:
        os.makedirs(images_dir / split, exist_ok=True)
        os.makedirs(labels_dir / split, exist_ok=True)
    
    # 移动文件
    for i, img_file in enumerate(image_files):
        split = "train" if i < split_idx else "val"
        
        # 移动图像文件
        shutil.move(str(img_file), str(images_dir / split / img_file.name))
        
        # 移动标签文件
        label_file = labels_dir / f"{img_file.stem}.txt"
        if label_file.exists():
            shutil.move(str(label_file), str(labels_dir / split / label_file.name))
    
    print(f"数据集分割完成! 训练集: {split_idx}, 验证集: {len(image_files) - split_idx}")

def check_dataset(data_dir):
    """
    检查数据集完整性
    
    Args:
        data_dir: 数据集目录
    """
    print("检查数据集...")
    
    for split in ["train", "val"]:
        images_dir = Path(data_dir) / "images" / split
        labels_dir = Path(data_dir) / "labels" / split
        
        if not images_dir.exists():
            print(f"警告: {images_dir} 不存在")
            continue
            
        if not labels_dir.exists():
            print(f"警告: {labels_dir} 不存在")
            continue
        
        # 统计文件数量
        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        label_files = list(labels_dir.glob("*.txt"))
        
        print(f"{split} 集:")
        print(f"  图像文件: {len(image_files)}")
        print(f"  标签文件: {len(label_files)}")
        
        # 检查匹配
        missing_labels = []
        for img_file in image_files:
            label_file = labels_dir / f"{img_file.stem}.txt"
            if not label_file.exists():
                missing_labels.append(img_file.name)
        
        if missing_labels:
            print(f"  缺少标签的图像: {len(missing_labels)}")
            if len(missing_labels) <= 5:
                print(f"    {missing_labels}")
            else:
                print(f"    {missing_labels[:5]} ...")

def visualize_dataset(data_dir, num_samples=5):
    """
    可视化数据集样本
    
    Args:
        data_dir: 数据集目录
        num_samples: 可视化样本数量
    """
    import matplotlib.pyplot as plt
    
    images_dir = Path(data_dir) / "images" / "train"
    labels_dir = Path(data_dir) / "labels" / "train"
    
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    random.shuffle(image_files)
    
    fig, axes = plt.subplots(1, min(num_samples, len(image_files)), figsize=(15, 3))
    if num_samples == 1:
        axes = [axes]
    
    for i, img_file in enumerate(image_files[:num_samples]):
        # 读取图像
        img = cv2.imread(str(img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        
        # 读取标签
        label_file = labels_dir / f"{img_file.stem}.txt"
        if label_file.exists():
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            # 绘制边界框
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x, y, width, height = map(float, parts[1:5])
                    
                    # 转换为像素坐标
                    x1 = int((x - width/2) * w)
                    y1 = int((y - height/2) * h)
                    x2 = int((x + width/2) * w)
                    y2 = int((y + height/2) * h)
                    
                    # 绘制矩形
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(img, str(class_id), (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        
        axes[i].imshow(img)
        axes[i].set_title(f"{img_file.name}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{data_dir}/dataset_samples.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"可视化结果保存到: {data_dir}/dataset_samples.png")

if __name__ == "__main__":
    data_dir = "data"
    
    # 检查数据集
    check_dataset(data_dir)
    
    # 可视化数据集
    if os.path.exists(f"{data_dir}/images/train"):
        visualize_dataset(data_dir)
