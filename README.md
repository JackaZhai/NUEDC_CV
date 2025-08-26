# NUEDC CV

开源的计算机视觉工具集，用于于全国大学生电子设计竞赛（NUEDC）。
项目主要面向 **MaixCam** 平台，包含阈值调试、YOLOv11 训练与权重转换、
激光点识别等功能模块。

## Directory Structure
```
NUEDC_CV/
├── .gitignore                          # Git 忽略规则
├── BeiSai/                             # 比赛相关示例脚本
│   ├── pic_capture.py                  # 图像采集示例
│   ├── 五区域巡线法.py                 # 五区域巡线算法
│   ├── 本地调LAB阈值.py                # 本地 LAB 阈值调试
│   ├── 激光识别_LAB色块.py             # 基于 LAB 色块识别激光
│   ├── 激光识别_帧差法.py              # 帧差法识别激光
│   ├── 激光识别_综合.py                # 激光识别综合示例
│   └── 在线调LAB阈值/                  # 浏览器在线 LAB 阈值调节工具
│       └── thresholding-filter-browser-html.html  # 阈值调节网页
├── yolo/                               # YOLOv11 训练与工具
│   ├── configs/
│   │   └── train_config.yaml           # 训练参数示例
│   ├── models/
│   │   ├── convert_yolo11_to_cvimodel.sh   # 模型转换脚本
│   │   ├── export_model.py             # 导出模型权重
│   │   └── text.png                    # 模型结构示意图
│   ├── scripts/
│   │   └── train.py                    # YOLOv11 训练脚本
│   ├── utils/
│   │   ├── data_utils.py               # 数据处理工具函数
│   │   └── voc_to_yolo.py              # VOC 数据集转 YOLO 格式
│   └── yolo11n.pt                      # YOLOv11n 预训练权重
├── LICENSE                             # Apache 2.0 许可证
└── README.md                           # 项目说明文档
```


## Contributing
欢迎通过 Issue 或 Pull Request 贡献代码、文档或功能建议。

## License
本项目采用 [Apache License 2.0](./LICENSE) 许可协议。
