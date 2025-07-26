# 注意！！！该方法未测试，大概思路是当帧差法识别不到激光时采用LAB色块的方法
from maix import camera, display, image, app
import time

# ================================ 配置与常量定义 ================================
class ComprehensiveLaserConfig:
    """综合激光检测器配置类 - 集中管理所有可配置项"""
    
    # ==================== 摄像头配置 ====================
    CAMERA_WIDTH = 320      # 摄像头宽度(像素)
    CAMERA_HEIGHT = 240     # 摄像头高度(像素)
    CAMERA_FORMAT = image.Format.FMT_RGB888  # RGB格式，兼容两种检测方法
    
    # ==================== 帧差法参数 ====================
    BINARY_THRESHOLD = 200  # 亮度二值化阈值 - 激光通常很亮
    MOVEMENT_THRESHOLD = 5  # 移动检测阈值(像素) - 小于此值认为是静止
    
    # ==================== LAB颜色检测参数 ====================
    # LAB颜色空间阈值 - 用于识别红色激光点
    LAB_L_MIN = 20          # 最小亮度阈值
    LAB_L_MAX = 100         # 最大亮度阈值
    LAB_A_MIN = 15          # 最小A值 - 红色偏向
    LAB_A_MAX = 127         # 最大A值
    LAB_B_MIN = -128        # 最小B值
    LAB_B_MAX = 127         # 最大B值
    
    # ==================== 形态学检测参数 ====================
    MIN_AREA = 5            # 最小激光点面积(像素)
    MAX_AREA = 500          # 最大激光点面积(像素)
    PIXELS_THRESHOLD = 5    # 最小像素数阈值
    AREA_THRESHOLD = 5      # 最小面积阈值
    MERGE_BLOBS = True      # 是否合并相邻blob
    
    # ==================== 检测模式参数 ====================
    FRAME_DIFF_PRIORITY = True  # 优先使用帧差法
    SWITCH_DELAY = 10           # 切换到LAB检测前的等待帧数
    
    # ==================== 显示参数 ====================
    # 绘制参数
    CIRCLE_RADIUS_INNER = 5
    CIRCLE_RADIUS_OUTER = 15
    CROSSHAIR_LENGTH = 20
    RECT_THICKNESS = 2
    CIRCLE_THICKNESS = 2
    TEXT_OFFSET_X = 25
    TEXT_OFFSET_Y_1 = -20
    TEXT_OFFSET_Y_2 = 0
    TEXT_OFFSET_Y_3 = 15
    
    # 文本缩放比例
    TEXT_SCALE_LARGE = 1.2
    TEXT_SCALE_MEDIUM = 1.0
    TEXT_SCALE_SMALL = 0.8
    TEXT_SCALE_TINY = 0.7
    
    # 状态信息显示位置
    STATUS_X = 10
    STATUS_Y_FPS = 10
    STATUS_Y_MODE = 30
    STATUS_Y_TITLE_OFFSET = 60
    STATUS_Y_CONTROL1_OFFSET = 40
    
    # ==================== 性能监控参数 ====================
    FPS_UPDATE_INTERVAL = 1.0
    
    # ==================== 颜色常量 ====================
    COLOR_FRAME_DIFF = image.COLOR_GREEN    # 帧差法检测颜色
    COLOR_LAB = image.COLOR_YELLOW          # LAB检测颜色
    COLOR_TEXT = image.COLOR_WHITE
    
    # ==================== 调试参数 ====================
    ENABLE_CONSOLE_OUTPUT = True
    
    @classmethod
    def get_lab_threshold(cls):
        """获取LAB颜色阈值列表"""
        return [cls.LAB_L_MIN, cls.LAB_L_MAX, 
                cls.LAB_A_MIN, cls.LAB_A_MAX, 
                cls.LAB_B_MIN, cls.LAB_B_MAX]
    
    @classmethod
    def get_camera_config(cls):
        """获取摄像头配置"""
        return {
            'width': cls.CAMERA_WIDTH,
            'height': cls.CAMERA_HEIGHT,
            'format': cls.CAMERA_FORMAT
        }
    
    @classmethod
    def print_config(cls):
        """打印当前配置"""
        print("=== 综合激光检测器配置 ===")
        print(f"检测模式:")
        print(f"  优先使用帧差法: {cls.FRAME_DIFF_PRIORITY}")
        print(f"  切换延迟: {cls.SWITCH_DELAY} 帧")
        print(f"帧差法参数:")
        print(f"  亮度阈值: {cls.BINARY_THRESHOLD}")
        print(f"  移动阈值: {cls.MOVEMENT_THRESHOLD} 像素")
        print(f"LAB颜色检测:")
        print(f"  LAB阈值: L({cls.LAB_L_MIN}-{cls.LAB_L_MAX}) A({cls.LAB_A_MIN}-{cls.LAB_A_MAX}) B({cls.LAB_B_MIN}-{cls.LAB_B_MAX})")
        print(f"共同参数:")
        print(f"  面积范围: {cls.MIN_AREA}-{cls.MAX_AREA} 像素")
        print(f"摄像头配置:")
        print(f"  分辨率: {cls.CAMERA_WIDTH}x{cls.CAMERA_HEIGHT}")
        print("========================")

# ================================ 综合检测类 ================================
class ComprehensiveLaserDetector:
    def __init__(self, config=ComprehensiveLaserConfig):
        """初始化综合激光检测器"""
        self.config = config
        
        # 帧差法相关
        self.prev_frame = None
        self.prev_blobs = None
        
        # 检测模式管理
        self.current_mode = "FRAME_DIFF" if config.FRAME_DIFF_PRIORITY else "LAB"
        self.no_detection_count = 0
        
        # 显示相关
        self.frame_count = 0
        self.start_time = time.time()
        self.current_fps = 0
        
    def detect_frame_diff(self, img):
        """帧差法检测移动的激光点"""
        # 转换为灰度图进行处理
        gray_img = img.to_grayscale()
        
        # 如果是第一帧，保存并返回
        if self.prev_frame is None:
            self.prev_frame = gray_img.copy()
            return None
        
        # 找出当前帧的亮点
        blobs = gray_img.find_blobs([(self.config.BINARY_THRESHOLD, 255)],
                                   pixels_threshold=self.config.PIXELS_THRESHOLD,
                                   area_threshold=self.config.AREA_THRESHOLD,
                                   merge=self.config.MERGE_BLOBS)
        
        # 更新前一帧
        prev_blobs = self.prev_blobs
        self.prev_frame = gray_img.copy()
        self.prev_blobs = blobs
        
        if not blobs:
            return None
        
        # 找出移动的blob
        moving_blobs = []
        
        for blob in blobs:
            # 检查面积范围
            if not (self.config.MIN_AREA <= blob.area() <= self.config.MAX_AREA):
                continue
                
            # 检查是否在移动
            is_moving = True
            
            if prev_blobs:
                for prev_blob in prev_blobs:
                    dx = blob.cx() - prev_blob.cx()
                    dy = blob.cy() - prev_blob.cy()
                    distance = (dx*dx + dy*dy) ** 0.5
                    
                    if distance < self.config.MOVEMENT_THRESHOLD:
                        is_moving = False
                        break
            
            if is_moving:
                moving_blobs.append(blob)
        
        if not moving_blobs:
            return None
        
        # 选择最大的移动blob
        best_blob = max(moving_blobs, key=lambda b: b.area())
        
        return {
            'center': (best_blob.cx(), best_blob.cy()),
            'blob': best_blob,
            'area': best_blob.area(),
            'rect': (best_blob.x(), best_blob.y(), best_blob.w(), best_blob.h()),
            'method': 'FRAME_DIFF'
        }
    
    def detect_lab_color(self, img):
        """LAB颜色检测静止或移动的激光点"""
        # 使用LAB颜色阈值查找红色激光
        blobs = img.find_blobs([self.config.get_lab_threshold()],
                              pixels_threshold=self.config.PIXELS_THRESHOLD,
                              area_threshold=self.config.AREA_THRESHOLD,
                              merge=self.config.MERGE_BLOBS)
        
        if not blobs:
            return None
        
        # 过滤面积范围
        valid_blobs = [blob for blob in blobs 
                      if self.config.MIN_AREA <= blob.area() <= self.config.MAX_AREA]
        
        if not valid_blobs:
            return None
        
        # 选择最大的blob
        best_blob = max(valid_blobs, key=lambda b: b.area())
        
        return {
            'center': (best_blob.cx(), best_blob.cy()),
            'blob': best_blob,
            'area': best_blob.area(),
            'rect': (best_blob.x(), best_blob.y(), best_blob.w(), best_blob.h()),
            'method': 'LAB'
        }
    
    def detect_laser(self, img):
        """综合检测激光点"""
        detection_result = None
        
        # 根据当前模式进行检测
        if self.current_mode == "FRAME_DIFF":
            # 尝试帧差法检测
            detection_result = self.detect_frame_diff(img)
            
            if detection_result is None:
                self.no_detection_count += 1
                
                # 如果连续多帧没有检测到移动，切换到LAB模式
                if self.no_detection_count >= self.config.SWITCH_DELAY:
                    self.current_mode = "LAB"
                    self.no_detection_count = 0
                    
                    # 立即尝试LAB检测
                    detection_result = self.detect_lab_color(img)
            else:
                self.no_detection_count = 0
        
        else:  # LAB模式
            # 使用LAB颜色检测
            detection_result = self.detect_lab_color(img)
            
            # 同时检查是否有移动
            frame_diff_result = self.detect_frame_diff(img)
            
            # 如果检测到移动，优先使用帧差法结果
            if frame_diff_result is not None and self.config.FRAME_DIFF_PRIORITY:
                detection_result = frame_diff_result
                self.current_mode = "FRAME_DIFF"
                self.no_detection_count = 0
        
        return detection_result
    
    def draw_results(self, img, detection_result):
        """在图像上绘制检测结果"""
        if detection_result is None:
            return
        
        center = detection_result['center']
        area = detection_result['area']
        rect = detection_result['rect']
        method = detection_result['method']
        
        # 根据检测方法选择颜色
        color = self.config.COLOR_FRAME_DIFF if method == 'FRAME_DIFF' else self.config.COLOR_LAB
        
        # 绘制矩形框
        img.draw_rect(rect[0], rect[1], rect[2], rect[3], color, self.config.RECT_THICKNESS)
        
        # 绘制中心点
        img.draw_circle(center[0], center[1], self.config.CIRCLE_RADIUS_INNER, color, -1)
        img.draw_circle(center[0], center[1], self.config.CIRCLE_RADIUS_OUTER, color, self.config.CIRCLE_THICKNESS)
        
        # 绘制十字准线
        crosshair_len = self.config.CROSSHAIR_LENGTH
        img.draw_line(center[0] - crosshair_len, center[1], 
                     center[0] + crosshair_len, center[1], color, self.config.RECT_THICKNESS)
        img.draw_line(center[0], center[1] - crosshair_len, 
                     center[0], center[1] + crosshair_len, color, self.config.RECT_THICKNESS)
        
        # 显示信息
        text_x = center[0] + self.config.TEXT_OFFSET_X
        method_text = "Moving" if method == 'FRAME_DIFF' else "Static"
        img.draw_string(text_x, center[1] + self.config.TEXT_OFFSET_Y_1, 
                       f"Laser {method_text}", color, scale=self.config.TEXT_SCALE_MEDIUM)
        img.draw_string(text_x, center[1] + self.config.TEXT_OFFSET_Y_2, 
                       f"Area: {area}", color, scale=self.config.TEXT_SCALE_SMALL)
        img.draw_string(text_x, center[1] + self.config.TEXT_OFFSET_Y_3, 
                       f"Pos: ({center[0]},{center[1]})", color, scale=self.config.TEXT_SCALE_TINY)
    
    def draw_status(self, img):
        """绘制状态信息"""
        # 计算FPS
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        
        if elapsed_time > self.config.FPS_UPDATE_INTERVAL:
            self.current_fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
        
        # 显示FPS
        img.draw_string(self.config.STATUS_X, self.config.STATUS_Y_FPS, 
                       f"FPS: {self.current_fps:.1f}", self.config.COLOR_TEXT, 
                       scale=self.config.TEXT_SCALE_LARGE)
        
        # 显示当前模式
        mode_color = self.config.COLOR_FRAME_DIFF if self.current_mode == "FRAME_DIFF" else self.config.COLOR_LAB
        img.draw_string(self.config.STATUS_X, self.config.STATUS_Y_MODE, 
                       f"Mode: {self.current_mode}", mode_color, 
                       scale=self.config.TEXT_SCALE_MEDIUM)
        
        # 显示标题
        img.draw_string(self.config.STATUS_X, 
                       img.height() - self.config.STATUS_Y_TITLE_OFFSET, 
                       "Comprehensive Laser Detection", self.config.COLOR_TEXT, 
                       scale=self.config.TEXT_SCALE_MEDIUM)
        
        # 显示控制提示
        img.draw_string(self.config.STATUS_X, 
                       img.height() - self.config.STATUS_Y_CONTROL1_OFFSET, 
                       "Frame Diff + LAB Color", self.config.COLOR_TEXT, 
                       scale=self.config.TEXT_SCALE_SMALL)

def main():
    """主函数"""
    print("MaixPy 综合激光检测器启动中...")
    
    # 打印配置信息
    ComprehensiveLaserConfig.print_config()
    
    # 初始化激光检测器
    detector = ComprehensiveLaserDetector()
    
    # 初始化摄像头
    cam_config = ComprehensiveLaserConfig.get_camera_config()
    cam = camera.Camera(cam_config['width'], cam_config['height'], cam_config['format'])
    
    # 初始化显示器
    disp = display.Display()
    
    print("系统已启动，综合检测模式...")
    print("- 优先检测移动激光（绿色框）")
    print("- 自动切换到静止激光检测（黄色框）")
    
    # 主循环
    while not app.need_exit():
        # 读取图像
        img = cam.read()
        
        # 检测激光点
        detection_result = detector.detect_laser(img)
        
        # 绘制检测结果
        detector.draw_results(img, detection_result)
        
        # 绘制状态信息
        detector.draw_status(img)
        
        # 控制台输出
        if ComprehensiveLaserConfig.ENABLE_CONSOLE_OUTPUT and detection_result:
            center = detection_result['center']
            area = detection_result['area']
            method = detection_result['method']
            print(f"[{method}] 激光检测到 - 位置: ({center[0]}, {center[1]}), 面积: {area}")
        
        # 显示图像
        disp.show(img)
    
    print("程序退出")

if __name__ == "__main__":
    main()