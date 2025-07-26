#!/usr/bin/env python3
"""
VOC数据集转YOLO格式脚本
使用方法: python utils/voc_to_yolo.py --voc_dir /path/to/voc --classes person car bike
"""

import argparse
import os
import sys
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import random
from tqdm import tqdm

# 添加项目根目录到Python路径
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

def convert_bbox(size, box):
    """
    转换边界框格式：从VOC的绝对坐标转换为YOLO的相对坐标
    
    Args:
        size: (width, height) 图像尺寸
        box: (xmin, ymin, xmax, ymax) VOC格式边界框
    
    Returns:
        (x, y, w, h) YOLO格式边界框（相对坐标）
    """
    dw = 1.0 / size[0]  # 宽度归一化因子
    dh = 1.0 / size[1]  # 高度归一化因子
    x = (box[0] + box[2]) / 2.0  # 中心点x
    y = (box[1] + box[3]) / 2.0  # 中心点y
    w = box[2] - box[0]  # 宽度
    h = box[3] - box[1]  # 高度
    
    # 归一化到[0,1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    
    return (x, y, w, h)

def parse_xml_annotation(xml_file, class_names):
    """
    解析XML标注文件
    
    Args:
        xml_file: XML文件路径
        class_names: 类别名称列表
    
    Returns:
        annotations: 标注列表
        image_info: 图像信息 (filename, width, height)
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # 获取图像信息
    filename = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    annotations = []
    
    # 解析所有目标
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        
        # 跳过不在指定类别中的目标
        if class_name not in class_names:
            continue
        
        class_id = class_names.index(class_name)
        
        # 获取边界框
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        # 转换为YOLO格式
        bbox_yolo = convert_bbox((width, height), (xmin, ymin, xmax, ymax))
        
        annotations.append({
            'class_id': class_id,
            'bbox': bbox_yolo
        })
    
    return annotations, (filename, width, height)

def convert_voc_to_yolo(voc_dir, output_dir, class_names, train_ratio=0.8):
    """
    将VOC格式数据集转换为YOLO格式
    
    Args:
        voc_dir: VOC数据集根目录
        output_dir: 输出目录
        class_names: 类别名称列表
        train_ratio: 训练集比例
    """
    voc_path = Path(voc_dir)
    output_path = Path(output_dir)
    
    # 检查VOC目录结构
    annotations_dir = voc_path / "Annotations"
    
    # 尝试多种可能的图像目录名称
    possible_image_dirs = ["Images", "JPEGImages", "images", "jpegimages"]
    images_dir = None
    
    for dir_name in possible_image_dirs:
        potential_dir = voc_path / dir_name
        if potential_dir.exists():
            images_dir = potential_dir
            print(f"✅ 找到图像目录: {images_dir}")
            break
    
    if not annotations_dir.exists():
        print(f"❌ 错误: {annotations_dir} 不存在")
        return False
    
    if images_dir is None:
        print(f"❌ 错误: 未找到图像目录，检查了以下位置:")
        for dir_name in possible_image_dirs:
            print(f"   - {voc_path / dir_name}")
        return False
    
    # 创建输出目录结构
    for split in ['train', 'val']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # 获取所有XML文件
    xml_files = list(annotations_dir.glob("*.xml"))
    if not xml_files:
        print(f"错误: 在 {annotations_dir} 中没有找到XML文件")
        return False
    
    print(f"找到 {len(xml_files)} 个标注文件")
    
    # 随机分割训练和验证集
    random.shuffle(xml_files)
    split_idx = int(len(xml_files) * train_ratio)
    train_files = xml_files[:split_idx]
    val_files = xml_files[split_idx:]
    
    print(f"训练集: {len(train_files)} 个文件")
    print(f"验证集: {len(val_files)} 个文件")
    
    # 统计信息
    total_converted = 0
    total_objects = 0
    missing_images = []
    
    def process_files(files, split):
        """处理文件列表"""
        nonlocal total_converted, total_objects
        
        split_converted = 0
        split_objects = 0
        
        print(f"\n处理 {split} 集...")
        
        for xml_file in tqdm(files, desc=f"转换{split}集"):
            try:
                # 解析XML标注
                annotations, (filename, width, height) = parse_xml_annotation(xml_file, class_names)
                
                # 检查对应的图像文件是否存在
                img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
                img_file = None
                
                for ext in img_extensions:
                    potential_img = images_dir / filename.replace('.jpg', ext).replace('.jpeg', ext).replace('.png', ext).replace('.bmp', ext)
                    if potential_img.exists():
                        img_file = potential_img
                        break
                
                if img_file is None:
                    # 尝试不带扩展名的文件名
                    base_name = Path(filename).stem
                    for ext in img_extensions:
                        potential_img = images_dir / f"{base_name}{ext}"
                        if potential_img.exists():
                            img_file = potential_img
                            break
                
                if img_file is None:
                    missing_images.append(filename)
                    continue
                
                # 复制图像文件到输出目录
                dst_img = output_path / 'images' / split / img_file.name
                shutil.copy2(str(img_file), str(dst_img))
                
                # 创建YOLO格式标签文件
                label_file = output_path / 'labels' / split / f"{xml_file.stem}.txt"
                
                with open(label_file, 'w') as f:
                    for ann in annotations:
                        bbox = ann['bbox']
                        f.write(f"{ann['class_id']} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
                
                split_converted += 1
                split_objects += len(annotations)
                
            except Exception as e:
                print(f"\n处理文件 {xml_file} 时出错: {e}")
                continue
        
        print(f"{split}集转换完成: {split_converted} 个文件, {split_objects} 个目标")
        return split_converted, split_objects
    
    # 处理训练集和验证集
    train_converted, train_objects = process_files(train_files, 'train')
    val_converted, val_objects = process_files(val_files, 'val')
    
    total_converted = train_converted + val_converted
    total_objects = train_objects + val_objects
    
    # 创建数据集配置文件
    dataset_config = f"""# 从VOC转换的数据集配置
# 原始VOC目录: {voc_dir}
# 转换时间: {Path(__file__).stat().st_mtime}

# 数据路径
train: {output_path}/images/train
val: {output_path}/images/val

# 类别数量
nc: {len(class_names)}

# 类别名称
names:
"""
    
    for i, class_name in enumerate(class_names):
        dataset_config += f"  {i}: '{class_name}'\n"
    
    config_file = output_path / 'dataset.yaml'
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(dataset_config)
    
    # 打印转换统计
    print("\n" + "="*50)
    print("转换完成统计:")
    print(f"总文件数: {len(xml_files)}")
    print(f"成功转换: {total_converted}")
    print(f"总目标数: {total_objects}")
    print(f"类别数量: {len(class_names)}")
    print(f"类别列表: {class_names}")
    
    if missing_images:
        print(f"\n缺失图像文件: {len(missing_images)}")
        if len(missing_images) <= 10:
            for img in missing_images:
                print(f"  - {img}")
        else:
            for img in missing_images[:10]:
                print(f"  - {img}")
            print(f"  ... 还有 {len(missing_images)-10} 个")
    
    print(f"\n输出目录: {output_path}")
    print(f"配置文件: {config_file}")
    print("="*50)
    
    return True

def scan_voc_classes(voc_dir):
    """
    扫描VOC数据集中的所有类别
    
    Args:
        voc_dir: VOC数据集目录
    
    Returns:
        set: 类别名称集合
    """
    annotations_dir = Path(voc_dir) / "Annotations"
    
    if not annotations_dir.exists():
        print(f"错误: {annotations_dir} 不存在")
        return set()
    
    classes = set()
    xml_files = list(annotations_dir.glob("*.xml"))
    
    print(f"扫描 {len(xml_files)} 个标注文件...")
    
    for xml_file in tqdm(xml_files, desc="扫描类别"):
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                if class_name:
                    classes.add(class_name)
        except Exception as e:
            print(f"解析文件 {xml_file} 时出错: {e}")
            continue
    
    return classes

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='VOC数据集转YOLO格式工具')
    parser.add_argument('voc_dir', nargs='?', help='VOC数据集目录路径 (可直接粘贴路径)')
    parser.add_argument('--output', '-o', type=str, default='data', help='输出目录 (默认: data)')
    parser.add_argument('--classes', '-c', nargs='+', help='类别名称列表，例如: person car bike')
    parser.add_argument('--train_ratio', '-r', type=float, default=0.8, help='训练集比例 (默认: 0.8)')
    parser.add_argument('--scan', '-s', action='store_true', help='扫描数据集中的所有类别')
    parser.add_argument('--auto', '-a', action='store_true', help='自动检测类别并转换')
    
    args = parser.parse_args()
    
    # 如果没有提供VOC目录，提示用户输入
    if not args.voc_dir:
        print("=== VOC到YOLO数据集转换工具 ===")
        print()
        voc_dir = input("请输入或粘贴VOC数据集目录路径: ").strip()
        if not voc_dir:
            print("❌ 未输入目录路径")
            return 1
        # 移除可能的引号
        voc_dir = voc_dir.strip('"').strip("'")
    else:
        voc_dir = args.voc_dir.strip('"').strip("'")
    
    # 检查VOC目录是否存在
    if not os.path.exists(voc_dir):
        print(f"❌ 错误: VOC目录 {voc_dir} 不存在")
        return 1
    
    print(f"📁 VOC数据集目录: {voc_dir}")
    print(f"📁 输出目录: {args.output}")
    
    # 如果需要扫描类别
    if args.scan:
        print("\n🔍 扫描数据集中的类别...")
        classes = scan_voc_classes(voc_dir)
        
        if classes:
            print(f"\n✅ 找到的类别 ({len(classes)} 个):")
            for i, cls in enumerate(sorted(classes)):
                print(f"  {i}: {cls}")
            
            print(f"\n💡 使用方法示例:")
            classes_str = ' '.join(sorted(classes))
            print(f"python utils/voc_to_yolo.py \"{voc_dir}\" --classes {classes_str}")
            print(f"或者使用自动模式:")
            print(f"python utils/voc_to_yolo.py \"{voc_dir}\" --auto")
        else:
            print("❌ 未找到任何类别")
        
        return 0
    
    # 自动模式：自动检测所有类别
    if args.auto:
        print("\n🔍 自动检测类别...")
        classes = scan_voc_classes(voc_dir)
        
        if not classes:
            print("❌ 未找到任何类别")
            return 1
        
        class_names = sorted(list(classes))
        print(f"✅ 自动检测到 {len(class_names)} 个类别: {class_names}")
    else:
        # 检查类别参数
        if not args.classes:
            print("❌ 错误: 请指定类别名称")
            print("💡 使用以下选项之一:")
            print("   --scan  或 -s     : 扫描数据集中的所有类别")
            print("   --auto  或 -a     : 自动检测所有类别并转换")
            print("   --classes 或 -c   : 手动指定类别，例如: --classes person car bike")
            print(f"\n示例:")
            print(f"  python utils/voc_to_yolo.py \"{voc_dir}\" --scan")
            print(f"  python utils/voc_to_yolo.py \"{voc_dir}\" --auto")
            print(f"  python utils/voc_to_yolo.py \"{voc_dir}\" --classes person car bike")
            return 1
        
        class_names = args.classes
    
    print(f"🏷️  指定类别: {class_names}")
    print(f"📊 训练集比例: {args.train_ratio}")
    
    # 开始转换
    print(f"\n🔄 开始转换数据集...")
    success = convert_voc_to_yolo(
        voc_dir=voc_dir,
        output_dir=args.output,
        class_names=class_names,
        train_ratio=args.train_ratio
    )
    
    if success:
        print("\n🎉 转换成功!")
        print(f"💡 现在可以使用以下命令开始训练:")
        print(f"   python scripts/train.py --data {args.output}/dataset.yaml")
        return 0
    else:
        print("\n❌ 转换失败")
        return 1

if __name__ == '__main__':
    exit(main())
