'''
LAB色彩阈值配置工具
该工具用于配置LAB色彩空间的阈值参数，支持实时预览与参数调整。
推荐调试顺序：
1. 调整L通道的最小值和最大值
    # 点击 <L_min 和 <L_max 按钮
    # L通道通常设置较大范围，如 [0, 80] 或 [20, 100]
    # 激光通常亮度较高，建议 L_min=20, L_max=100
2. 调整A通道的最小值和最大值（红绿轴）
    # 对于红色：A值为正值，通常 [15, 127]
    # 对于绿色：A值为负值，通常 [-80, -5]
    # 精确调整这个范围最为关键
3. 调整B通道的最小值和最大值（黄蓝轴）
    # 对于黄色：B值为正值，通常 [15, 127]
    # 对于蓝色：B值为负值，通常 [-80, -5]
    # 精确调整这个范围最为关键

'''

# 导入必要的模块
from maix import image, camera, display, time, touchscreen, app
import math

# ------------------ 配置与常量定义（集中管理可配置项） ------------------

# 屏幕与摄像头参数

SCREEN_WIDTH, SCREEN_HEIGHT = 320, 240
CAMERA_RESOLUTION = (SCREEN_WIDTH, SCREEN_HEIGHT)

# LAB 参数范围常量（避免硬编码，统一维护）

PARAM_RANGES = {
"L_min": (0, 100), # L 通道范围
"L_max": (0, 100),
"A_min": (-128, 127), # A/B 通道范围
"A_max": (-128, 127),
"B_min": (-128, 127),
"B_max": (-128, 127)
}

# 其他常量

MIN_AREA = 10 # 色块最小有效面积
ADJUST_STEP = 2 # 参数调整步长
BUTTON_MARGIN_X = 4 # 按钮水平边距
BUTTON_MARGIN_Y = 4 # 按钮垂直边距

# ------------------ 核心类定义（封装状态与方法） ------------------

class ColorThresholdConfig:
    def __init__(self):  # 初始化硬件
        self.cam = camera.Camera(*CAMERA_RESOLUTION)
        self.disp = display.Display()
        self.ts = touchscreen.TouchScreen()

        # 状态变量 - 通用默认值，便于各种颜色调试
        self.lab_params = {
            "L_min": 30,   # 亮度最小值 - 过滤较暗区域
            "L_max": 100,  # 亮度最大值 - 保持高亮度范围
            "A_min": -64,  # A通道最小值 - 中性偏绿
            "A_max": 64,   # A通道最大值 - 中性偏红
            "B_min": -64,  # B通道最小值 - 中性偏蓝
            "B_max": 64    # B通道最大值 - 中性偏黄
        }
        self.in_binary_mode = False  # 二值化显示模式
        self.exit_flag = False       # 退出标志
        self.selected_param = None   # 当前选中参数
        self.ui_img = image.Image(SCREEN_WIDTH, SCREEN_HEIGHT)  # UI背景

        # 按钮位置存储（key: 按钮标签, value: [x, y, w, h]）
        self.buttons = {
            "exit": None,
            "binary": None,
            "params": {}  # 存储参数按钮: {label: pos}
        }

        # 初始化UI
        self._init_ui()

    def _init_ui(self):
        """初始化UI界面（绘制背景与按钮）"""
        # 绘制白色背景
        self.ui_img.draw_rect(0, 0, self.ui_img.width(), self.ui_img.height(), image.COLOR_WHITE)

        # 绘制退出按钮（右上角）
        exit_label = "< Exit"
        self.buttons["exit"] = self._draw_button(exit_label,
                                                x=SCREEN_WIDTH - self._get_button_width(exit_label),
                                                y=0)

        # 绘制二值化切换按钮（顶部中间）
        binary_label = "< Binary"
        btn_width = self._get_button_width(binary_label)
        self.buttons["binary"] = self._draw_button(binary_label,
                                                  x=(SCREEN_WIDTH - btn_width) // 2,
                                                  y=0)

        # 绘制参数调整按钮（左侧垂直排列）
        param_labels = ["<L_min", "<L_max", "<A_min", "<A_max", "<B_min", "<B_max"]
        y_offset = 0  # 垂直偏移量
        for label in param_labels:
            self.buttons["params"][label] = self._draw_button(label, x=0, y=y_offset)
            # 更新偏移量（下一个按钮在当前按钮下方）
            y_offset += self.buttons["params"][label][3]

    def _get_text_size(self, text):
        """获取文本尺寸（封装重复调用）"""
        return image.string_size(text)

    def _get_button_width(self, text):
        """计算按钮宽度（文本宽 + 2倍水平边距）"""
        text_w, _ = self._get_text_size(text)
        return 2 * BUTTON_MARGIN_X + text_w

    def _get_button_height(self, text):
        """计算按钮高度（文本高 + 2倍垂直边距）"""
        _, text_h = self._get_text_size(text)
        return 2 * BUTTON_MARGIN_Y + text_h

    def _draw_button(self, text, x, y):
        """绘制按钮并返回位置信息 [x, y, w, h]"""
        text_w, text_h = self._get_text_size(text)
        btn_w = self._get_button_width(text)
        btn_h = self._get_button_height(text)

        # 绘制按钮边框
        self.ui_img.draw_rect(x, y, btn_w, btn_h, image.COLOR_WHITE, 2)
        # 绘制按钮文本
        self.ui_img.draw_string(x + BUTTON_MARGIN_X, y + BUTTON_MARGIN_Y,
                               text, image.COLOR_WHITE)
        return [x, y, btn_w, btn_h]

    def _is_touch_in_button(self, x, y, btn_pos):
        """判断触摸点是否在按钮区域内"""
        btn_x, btn_y, btn_w, btn_h = btn_pos
        return (btn_x < x < btn_x + btn_w) and (btn_y < y < btn_y + btn_h)

    def _update_ui_status(self, img):
        """更新UI状态显示（参数值、模式状态）"""
        # 状态显示区域起始Y坐标（参数按钮下方）
        status_y = sum(btn[3] for btn in self.buttons["params"].values()) + 10

        # 绘制状态区域背景（白色覆盖）
        img.draw_rect(0, status_y, SCREEN_WIDTH, SCREEN_HEIGHT - status_y, image.COLOR_WHITE)

        # 显示LAB参数值（选中参数标红）
        for i, (param, value) in enumerate(self.lab_params.items()):
            color = image.COLOR_RED if param == self.selected_param else image.COLOR_BLACK
            img.draw_string(10, status_y + i * 15, f"{param}: {value}", color)

        # 显示二值化模式状态
        mode_text = "Binary: ON" if self.in_binary_mode else "Binary: OFF"
        img.draw_string(SCREEN_WIDTH - 100, status_y, mode_text, image.COLOR_BLACK)

    def _handle_touch(self, x, y):
        """处理触摸事件（拆分逻辑，减少主循环复杂度）"""
        # 1. 检查是否点击退出按钮
        if self._is_touch_in_button(x, y, self.buttons["exit"]):
            self.exit_flag = True
            return

        # 2. 检查是否点击二值化切换按钮
        if self._is_touch_in_button(x, y, self.buttons["binary"]):
            self.in_binary_mode = not self.in_binary_mode
            print(f"二值化模式: {'开启' if self.in_binary_mode else '关闭'}")
            return

        # 3. 检查是否点击参数按钮（选中参数）
        for label, btn_pos in self.buttons["params"].items():
            if self._is_touch_in_button(x, y, btn_pos):
                self.selected_param = label[1:]  # 去掉前缀"<"
                print(f"选中参数: {self.selected_param}")
                return

        # 4. 右侧区域触摸（调整选中的参数）
        if x > SCREEN_WIDTH / 2 and self.selected_param:
            self._adjust_param(y)

    def _adjust_param(self, touch_y):
        """根据触摸Y坐标调整参数（上半部分增大，下半部分减小）"""
        param = self.selected_param
        min_val, max_val = PARAM_RANGES[param]

        # 上半屏触摸：增大参数；下半屏触摸：减小参数
        if touch_y < SCREEN_HEIGHT / 2:
            new_val = self.lab_params[param] + ADJUST_STEP
        else:
            new_val = self.lab_params[param] - ADJUST_STEP

        # 限制在有效范围内
        self.lab_params[param] = max(min_val, min(new_val, max_val))
        print(f"{param} = {self.lab_params[param]}")

    def run_threshold_adjust(self):
        """运行阈值调整模式（主交互逻辑）"""
        print("阈值调整模式：")
        print(" - 点击< Binary切换二值化显示")
        print(" - 点击< Exit退出调整")
        print(" - 点击参数按钮选择参数，右侧触摸调整值")

        # 预计算二值化显示区域（固定不变，无需每次循环计算）
        max_btn_width = max(btn[2] for btn in self.buttons["params"].values())
        top_btn_height = max(self.buttons["exit"][3], self.buttons["binary"][3])
        binary_area = {
            "x": max_btn_width + 5,
            "y": top_btn_height + 5,
            "w": SCREEN_WIDTH - max_btn_width - 10,  # 左右各留5px边距
            "h": SCREEN_HEIGHT - top_btn_height - 10
        }

        while not self.exit_flag:
            # 读取摄像头图像
            img = self.cam.read()
            original_img = img.copy()  # 保存原始图像
            # 绘制UI背景
            img.draw_image(0, 0, self.ui_img)

            # 二值化模式处理
            current_threshold = (
                self.lab_params["L_min"], self.lab_params["L_max"],
                self.lab_params["A_min"], self.lab_params["A_max"],
                self.lab_params["B_min"], self.lab_params["B_max"]
            )
            if self.in_binary_mode:
                # 裁剪显示区域并二值化
                crop = original_img.crop(
                    binary_area["x"], binary_area["y"],
                    binary_area["w"], binary_area["h"]
                )
                crop.binary([current_threshold])  # 应用阈值
                img.draw_image(binary_area["x"], binary_area["y"], crop)  # 绘制二值化图像
                # 绘制红色边框标记二值化区域
                img.draw_rect(binary_area["x"], binary_area["y"],
                             binary_area["w"], binary_area["h"],
                             image.COLOR_RED, 1)

            # 处理触摸事件
            touch_x, touch_y, pressed = self.ts.read()
            if pressed:
                # 转换触摸坐标到屏幕坐标系
                adjusted_x, adjusted_y = image.resize_map_pos_reverse(
                    SCREEN_WIDTH, SCREEN_HEIGHT,
                    self.disp.width(), self.disp.height(),
                    image.Fit.FIT_CONTAIN, touch_x, touch_y
                )
                self._handle_touch(adjusted_x, adjusted_y)

            # 更新UI状态显示
            self._update_ui_status(img)
            # 显示图像
            self.disp.show(img)

        return current_threshold

    def run_blob_detection(self, threshold):
        """运行色块检测模式（调整完成后）"""
        while not app.need_exit():
            img = self.cam.read()
            # 查找色块
            blobs = img.find_blobs([threshold], merge=True)

            # 绘制并打印有效色块（过滤小面积）
            for blob in blobs:
                if blob.area() < MIN_AREA:
                    continue
                # 绘制矩形和中心点
                img.draw_rect(blob[0], blob[1], blob[2], blob[3], image.COLOR_GREEN)
                img.draw_cross(blob[5], blob[6], image.COLOR_GREEN, size=3)
                print(f"色块中心: ({blob.cx()}, {blob.cy()}) 面积: {blob.area()}")

            self.disp.show(img)

# ------------------ 主程序入口 ------------------

if __name__ == "__main__":  # 初始化配置对象
    config = ColorThresholdConfig()
    # 运行阈值调整
    final_threshold = config.run_threshold_adjust()

    # 打印最终阈值
    print("\n调整完成！最终阈值：")
    print(f"L: {final_threshold[0]}~{final_threshold[1]}")
    print(f"A: {final_threshold[2]}~{final_threshold[3]}")
    print(f"B: {final_threshold[4]}~{final_threshold[5]}")

    # 运行色块检测
    config.run_blob_detection(final_threshold)