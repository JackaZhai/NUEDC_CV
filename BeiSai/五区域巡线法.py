#该方案未验证
from maix import camera, display, image, app
import time
import math

# ================================ 配置与常量定义 ================================
class FiveRegionLineConfig:
    """五区域巡线配置类 - 集中管理所有可配置项"""
    
    # ==================== 摄像头配置 ====================
    CAMERA_WIDTH = 320      # 摄像头宽度(像素)
    CAMERA_HEIGHT = 240     # 摄像头高度(像素)
    CAMERA_FORMAT = image.Format.FMT_RGB888  # 图像格式
    
    # ==================== 区域划分参数 ====================
    # 5个感兴趣区域(ROI)的Y坐标位置（从上到下）
    ROI_1_Y = 40           # 最远区域（前瞻）
    ROI_2_Y = 80           # 远区域
    ROI_3_Y = 120          # 中间区域
    ROI_4_Y = 160          # 近区域
    ROI_5_Y = 200          # 最近区域（车身前方）
    
    ROI_HEIGHT = 20        # 每个ROI的高度
    ROI_WIDTH_RATIO = 0.8  # ROI宽度占图像宽度的比例
    
    # ==================== 线条检测参数 ====================
    # 二值化阈值 - 用于黑线检测
    BINARY_THRESHOLD = 80   # 低于此值认为是黑线
    
    # LAB颜色阈值 - 用于特定颜色线条检测
    # 黑线的LAB阈值
    BLACK_LINE_LAB = [0, 30, -128, 127, -128, 127]
    # 白线的LAB阈值（可选）
    WHITE_LINE_LAB = [70, 100, -128, 127, -128, 127]
    
    # ==================== 巡线控制参数 ====================
    # 各区域权重 - 用于计算综合偏差
    WEIGHT_1 = 0.15        # 最远区域权重（前瞻性）
    WEIGHT_2 = 0.20        # 远区域权重
    WEIGHT_3 = 0.30        # 中间区域权重（主要参考）
    WEIGHT_4 = 0.20        # 近区域权重
    WEIGHT_5 = 0.15        # 最近区域权重（即时反应）
    
    # PID控制参数
    KP = 0.5               # 比例系数
    KI = 0.1               # 积分系数
    KD = 0.2               # 微分系数
    
    # 偏差范围
    MAX_ERROR = 160        # 最大偏差（半个图像宽度）
    DEAD_ZONE = 10         # 死区范围（小于此值不调整）
    
    # ==================== 显示参数 ====================
    # 绘制参数
    LINE_THICKNESS = 2      # 线条粗细
    RECT_THICKNESS = 2      # 矩形框粗细
    CIRCLE_RADIUS = 5       # 圆点半径
    
    # 文本缩放比例
    TEXT_SCALE_LARGE = 1.2
    TEXT_SCALE_MEDIUM = 1.0
    TEXT_SCALE_SMALL = 0.8
    TEXT_SCALE_TINY = 0.7
    
    # 状态信息显示位置
    STATUS_X = 10
    STATUS_Y_FPS = 10
    STATUS_Y_ERROR = 30
    STATUS_Y_CONTROL = 50
    
    # ==================== 颜色常量 ====================
    COLOR_ROI = image.COLOR_YELLOW      # ROI区域颜色
    COLOR_LINE = image.COLOR_GREEN      # 检测到的线条颜色
    COLOR_CENTER = image.COLOR_RED      # 中心点颜色
    COLOR_TARGET = image.COLOR_BLUE     # 目标点颜色
    COLOR_TEXT = image.COLOR_WHITE      # 文本颜色
    
    # ==================== 调试参数 ====================
    ENABLE_CONSOLE_OUTPUT = True        # 是否启用控制台输出
    SHOW_ALL_REGIONS = True            # 是否显示所有区域
    SHOW_DETECTION_POINTS = True       # 是否显示检测点
    
    @classmethod
    def get_camera_config(cls):
        """获取摄像头配置"""
        return {
            'width': cls.CAMERA_WIDTH,
            'height': cls.CAMERA_HEIGHT,
            'format': cls.CAMERA_FORMAT
        }
    
    @classmethod
    def get_roi_list(cls):
        """获取所有ROI区域"""
        roi_x = int(cls.CAMERA_WIDTH * (1 - cls.ROI_WIDTH_RATIO) / 2)
        roi_width = int(cls.CAMERA_WIDTH * cls.ROI_WIDTH_RATIO)
        
        return [
            (roi_x, cls.ROI_1_Y, roi_width, cls.ROI_HEIGHT),
            (roi_x, cls.ROI_2_Y, roi_width, cls.ROI_HEIGHT),
            (roi_x, cls.ROI_3_Y, roi_width, cls.ROI_HEIGHT),
            (roi_x, cls.ROI_4_Y, roi_width, cls.ROI_HEIGHT),
            (roi_x, cls.ROI_5_Y, roi_width, cls.ROI_HEIGHT)
        ]
    
    @classmethod
    def get_weights(cls):
        """获取权重列表"""
        return [cls.WEIGHT_1, cls.WEIGHT_2, cls.WEIGHT_3, cls.WEIGHT_4, cls.WEIGHT_5]
    
    @classmethod
    def print_config(cls):
        """打印当前配置"""
        print("=== 五区域巡线配置 ===")
        print(f"摄像头: {cls.CAMERA_WIDTH}x{cls.CAMERA_HEIGHT}")
        print(f"ROI位置: Y={cls.ROI_1_Y}, {cls.ROI_2_Y}, {cls.ROI_3_Y}, {cls.ROI_4_Y}, {cls.ROI_5_Y}")
        print(f"ROI尺寸: 宽度比例={cls.ROI_WIDTH_RATIO}, 高度={cls.ROI_HEIGHT}")
        print(f"权重: {cls.WEIGHT_1}, {cls.WEIGHT_2}, {cls.WEIGHT_3}, {cls.WEIGHT_4}, {cls.WEIGHT_5}")
        print(f"PID参数: KP={cls.KP}, KI={cls.KI}, KD={cls.KD}")
        print("========================")

# ================================ 五区域巡线类 ================================
class FiveRegionLineFollower:
    def __init__(self, config=FiveRegionLineConfig):
        """初始化五区域巡线器"""
        self.config = config
        
        # PID控制变量
        self.last_error = 0
        self.integral = 0
        
        # 历史记录
        self.error_history = []
        self.max_history = 10
        
        # 显示相关
        self.frame_count = 0
        self.start_time = time.time()
        self.current_fps = 0
        
        # 获取ROI列表
        self.roi_list = config.get_roi_list()
        self.weights = config.get_weights()
        
    def detect_line_in_roi(self, img, roi):
        """在指定ROI中检测线条"""
        x, y, w, h = roi
        
        # 提取ROI区域
        roi_img = img.crop(x, y, w, h)
        
        # 方法1：使用二值化检测黑线
        # 转换为灰度图
        gray_roi = roi_img.to_grayscale()
        
        # 查找黑色区域
        blobs = gray_roi.find_blobs([(0, self.config.BINARY_THRESHOLD)],
                                   pixels_threshold=10,
                                   area_threshold=10,
                                   merge=True)
        
        if not blobs:
            return None
        
        # 找到最大的blob（主线条）
        largest_blob = max(blobs, key=lambda b: b.area())
        
        # 计算线条中心在原图中的位置
        line_center_x = x + largest_blob.cx()
        line_center_y = y + largest_blob.cy()
        
        return {
            'center': (line_center_x, line_center_y),
            'blob': largest_blob,
            'roi_offset': (x, y)
        }
    
    def calculate_weighted_error(self, detections):
        """计算加权偏差"""
        total_error = 0
        total_weight = 0
        image_center_x = self.config.CAMERA_WIDTH / 2
        
        for i, detection in enumerate(detections):
            if detection is not None:
                # 计算该区域的偏差
                error = detection['center'][0] - image_center_x
                # 应用权重
                total_error += error * self.weights[i]
                total_weight += self.weights[i]
        
        # 计算加权平均偏差
        if total_weight > 0:
            weighted_error = total_error / total_weight
        else:
            # 没有检测到线条，使用上一次的误差
            weighted_error = self.last_error
        
        return weighted_error
    
    def pid_control(self, error):
        """PID控制计算"""
        # 死区处理
        if abs(error) < self.config.DEAD_ZONE:
            error = 0
        
        # 比例项
        p_term = self.config.KP * error
        
        # 积分项（限幅防止积分饱和）
        self.integral += error
        self.integral = max(-100, min(100, self.integral))
        i_term = self.config.KI * self.integral
        
        # 微分项
        d_term = self.config.KD * (error - self.last_error)
        
        # 更新上一次误差
        self.last_error = error
        
        # 计算控制输出
        control_output = p_term + i_term + d_term
        
        # 限幅
        control_output = max(-100, min(100, control_output))
        
        return control_output
    
    def process_frame(self, img):
        """处理一帧图像"""
        detections = []
        
        # 在每个ROI中检测线条
        for roi in self.roi_list:
            detection = self.detect_line_in_roi(img, roi)
            detections.append(detection)
        
        # 计算加权偏差
        error = self.calculate_weighted_error(detections)
        
        # PID控制
        control_output = self.pid_control(error)
        
        # 记录历史
        self.error_history.append(error)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
        
        return {
            'detections': detections,
            'error': error,
            'control': control_output,
            'direction': 'LEFT' if control_output < 0 else 'RIGHT' if control_output > 0 else 'CENTER'
        }
    
    def draw_results(self, img, result):
        """绘制检测结果"""
        detections = result['detections']
        error = result['error']
        control = result['control']
        
        # 绘制所有ROI区域
        if self.config.SHOW_ALL_REGIONS:
            for i, roi in enumerate(self.roi_list):
                x, y, w, h = roi
                # 根据是否检测到线条选择颜色
                color = self.config.COLOR_LINE if detections[i] else self.config.COLOR_ROI
                img.draw_rect(x, y, w, h, color, self.config.RECT_THICKNESS)
                
                # 显示区域编号
                img.draw_string(x + 2, y + 2, f"R{i+1}", self.config.COLOR_TEXT, 
                              scale=self.config.TEXT_SCALE_TINY)
        
        # 绘制检测到的线条中心点
        if self.config.SHOW_DETECTION_POINTS:
            for i, detection in enumerate(detections):
                if detection:
                    center = detection['center']
                    # 绘制检测点
                    img.draw_circle(center[0], center[1], 
                                  self.config.CIRCLE_RADIUS, 
                                  self.config.COLOR_LINE, -1)
                    
                    # 绘制到图像中心的连线
                    img_center_x = self.config.CAMERA_WIDTH // 2
                    img.draw_line(center[0], center[1], 
                                img_center_x, center[1], 
                                self.config.COLOR_CENTER, 1)
        
        # 绘制中心参考线
        center_x = self.config.CAMERA_WIDTH // 2
        img.draw_line(center_x, 0, center_x, self.config.CAMERA_HEIGHT, 
                     self.config.COLOR_CENTER, 1)
        
        # 绘制目标位置（根据误差）
        target_x = int(center_x + error)
        target_x = max(0, min(self.config.CAMERA_WIDTH - 1, target_x))
        img.draw_line(target_x, 0, target_x, self.config.CAMERA_HEIGHT, 
                     self.config.COLOR_TARGET, 2)
        
        # 绘制控制方向箭头
        arrow_y = 20
        arrow_length = int(abs(control) / 2)
        if control < -self.config.DEAD_ZONE:
            # 左转箭头
            img.draw_line(center_x, arrow_y, center_x - arrow_length, arrow_y, 
                         self.config.COLOR_TARGET, 3)
            img.draw_line(center_x - arrow_length, arrow_y, 
                         center_x - arrow_length + 10, arrow_y - 5, 
                         self.config.COLOR_TARGET, 3)
            img.draw_line(center_x - arrow_length, arrow_y, 
                         center_x - arrow_length + 10, arrow_y + 5, 
                         self.config.COLOR_TARGET, 3)
        elif control > self.config.DEAD_ZONE:
            # 右转箭头
            img.draw_line(center_x, arrow_y, center_x + arrow_length, arrow_y, 
                         self.config.COLOR_TARGET, 3)
            img.draw_line(center_x + arrow_length, arrow_y, 
                         center_x + arrow_length - 10, arrow_y - 5, 
                         self.config.COLOR_TARGET, 3)
            img.draw_line(center_x + arrow_length, arrow_y, 
                         center_x + arrow_length - 10, arrow_y + 5, 
                         self.config.COLOR_TARGET, 3)
    
    def draw_status(self, img, result):
        """绘制状态信息"""
        # 计算FPS
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        
        if elapsed_time > 1.0:
            self.current_fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
        
        # 显示FPS
        img.draw_string(self.config.STATUS_X, self.config.STATUS_Y_FPS, 
                       f"FPS: {self.current_fps:.1f}", 
                       self.config.COLOR_TEXT, 
                       scale=self.config.TEXT_SCALE_MEDIUM)
        
        # 显示误差
        error = result['error']
        img.draw_string(self.config.STATUS_X, self.config.STATUS_Y_ERROR, 
                       f"Error: {error:.1f}", 
                       self.config.COLOR_TEXT, 
                       scale=self.config.TEXT_SCALE_MEDIUM)
        
        # 显示控制输出
        control = result['control']
        direction = result['direction']
        img.draw_string(self.config.STATUS_X, self.config.STATUS_Y_CONTROL, 
                       f"Control: {control:.1f} {direction}", 
                       self.config.COLOR_TEXT, 
                       scale=self.config.TEXT_SCALE_MEDIUM)
        
        # 显示检测状态
        detections = result['detections']
        detected_regions = sum(1 for d in detections if d is not None)
        img.draw_string(self.config.CAMERA_WIDTH - 100, self.config.STATUS_Y_FPS, 
                       f"Detected: {detected_regions}/5", 
                       self.config.COLOR_TEXT, 
                       scale=self.config.TEXT_SCALE_MEDIUM)

def main():
    """主函数"""
    print("MaixPy 五区域巡线系统启动中...")
    
    # 打印配置信息
    FiveRegionLineConfig.print_config()
    
    # 初始化巡线器
    line_follower = FiveRegionLineFollower()
    
    # 初始化摄像头
    cam_config = FiveRegionLineConfig.get_camera_config()
    cam = camera.Camera(cam_config['width'], cam_config['height'], cam_config['format'])
    
    # 初始化显示器
    disp = display.Display()
    
    print("系统已启动，开始巡线...")
    print("- 黄色框：ROI区域")
    print("- 绿色框：检测到线条的区域")
    print("- 红线：图像中心线")
    print("- 蓝线：目标位置")
    
    # 主循环
    while not app.need_exit():
        # 读取图像
        img = cam.read()
        
        # 处理图像
        result = line_follower.process_frame(img)
        
        # 绘制结果
        line_follower.draw_results(img, result)
        line_follower.draw_status(img, result)
        
        # 控制台输出
        if FiveRegionLineConfig.ENABLE_CONSOLE_OUTPUT:
            error = result['error']
            control = result['control']
            direction = result['direction']
            if abs(error) > FiveRegionLineConfig.DEAD_ZONE:
                print(f"偏差: {error:.1f}, 控制: {control:.1f}, 方向: {direction}")
        
        # 显示图像
        disp.show(img)
    
    print("程序退出")

if __name__ == "__main__":
    main()