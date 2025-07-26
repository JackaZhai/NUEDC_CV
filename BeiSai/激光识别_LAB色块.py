from maix import camera, display, image, app
import time

# ================================ 配置与常量定义 ================================
class LaserDetectionConfig:
    """红色激光检测器配置类 - 集中管理所有可配置项"""
    
    # ==================== 颜色检测参数 ====================
    # LAB颜色空间阈值 - 用于识别红色激光点
    # L: 亮度 (0-100), A: 绿-红轴 (-128到127), B: 蓝-黄轴 (-128到127)
    # 红色激光通常具有高亮度和正A值特征
    LAB_L_MIN = 20          # 最小亮度阈值 - 过滤太暗的区域
    LAB_L_MAX = 100         # 最大亮度阈值 - 过滤过度曝光区域
    LAB_A_MIN = 15          # 最小A值 - 确保检测红色偏向
    LAB_A_MAX = 127         # 最大A值 - A轴上限
    LAB_B_MIN = -128        # 最小B值 - B轴下限
    LAB_B_MAX = 127         # 最大B值 - B轴上限
    
    # ==================== 形态学检测参数 ====================
    MIN_AREA = 5            # 最小激光点面积(像素) - 过滤噪点
    MAX_AREA = 50          # 最大激光点面积(像素) - 过滤大面积误检
    PIXELS_THRESHOLD = 5    # 最小像素数阈值 - blob检测的像素数要求
    AREA_THRESHOLD = 5      # 最小面积阈值 - blob检测的面积要求
    MERGE_BLOBS = True      # 是否合并相邻blob - 将分散的激光点合并
    
    # ==================== 显示与界面参数 ====================
    # 摄像头配置
    CAMERA_WIDTH =  320     # 摄像头宽度 - 影响检测精度和性能
    CAMERA_HEIGHT = 240     # 摄像头高度 - 影响检测精度和性能
    CAMERA_FORMAT = image.Format.FMT_RGB888  # 图像格式
    
    # 绘制参数
    CIRCLE_RADIUS_INNER = 5     # 中心点内圆半径 - 标记激光点中心
    CIRCLE_RADIUS_OUTER = 15    # 中心点外圆半径 - 突出显示激光点
    CROSSHAIR_LENGTH = 20       # 十字准线长度 - 精确定位用
    RECT_THICKNESS = 2          # 矩形框线条粗细
    CIRCLE_THICKNESS = 2        # 圆圈线条粗细
    TEXT_OFFSET_X = 25         # 文本X轴偏移 - 避免遮挡激光点
    TEXT_OFFSET_Y_1 = -20      # 第一行文本Y轴偏移
    TEXT_OFFSET_Y_2 = 0        # 第二行文本Y轴偏移
    TEXT_OFFSET_Y_3 = 15       # 第三行文本Y轴偏移
    
    # 文本缩放比例
    TEXT_SCALE_LARGE = 1.2     # 大文本缩放 - FPS显示
    TEXT_SCALE_MEDIUM = 1.0    # 中等文本缩放 - 状态信息
    TEXT_SCALE_SMALL = 0.8     # 小文本缩放 - 详细信息
    TEXT_SCALE_TINY = 0.7      # 微小文本缩放 - 坐标信息
    
    # 状态信息显示位置
    STATUS_X = 10              # 状态信息X坐标
    STATUS_Y_FPS = 10          # FPS显示Y坐标
    STATUS_Y_TITLE_OFFSET = 60 # 标题距离底部偏移
    STATUS_Y_CONTROL1_OFFSET = 40  # 控制提示1距离底部偏移
    STATUS_Y_CONTROL2_OFFSET = 20  # 控制提示2距离底部偏移
    
    # ==================== 性能监控参数 ====================
    FPS_UPDATE_INTERVAL = 1.0  # FPS更新间隔(秒) - 控制FPS计算频率
    
    # ==================== 颜色常量 ====================
    COLOR_DETECTION = image.COLOR_GREEN  # 检测到激光时的颜色
    COLOR_TEXT = image.COLOR_WHITE       # 文本颜色 - 白色确保可见性
    
    # ==================== 调试与日志参数 ====================
    ENABLE_CONSOLE_OUTPUT = True        # 是否启用控制台输出
    
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
        print("=== 红色激光检测器配置 ===")
        print(f"颜色检测:")
        print(f"  LAB阈值: L({cls.LAB_L_MIN}-{cls.LAB_L_MAX}) A({cls.LAB_A_MIN}-{cls.LAB_A_MAX}) B({cls.LAB_B_MIN}-{cls.LAB_B_MAX})")
        print(f"形态学检测:")
        print(f"  面积范围: {cls.MIN_AREA}-{cls.MAX_AREA} 像素")
        print(f"  像素阈值: {cls.PIXELS_THRESHOLD}")
        print(f"摄像头配置:")
        print(f"  分辨率: {cls.CAMERA_WIDTH}x{cls.CAMERA_HEIGHT}")
        print("========================")

# ================================ 检测类 ================================
class RedLaserDetector:
    def __init__(self, config=LaserDetectionConfig):
        """初始化红色激光检测器"""
        self.config = config
        
        # 显示相关
        self.frame_count = 0
        self.start_time = time.time()
        
    def detect_laser(self, img):
        """检测红色激光点 - 纯find_blobs找色块"""
        # 使用find_blobs进行颜色检测
        blobs = img.find_blobs([self.config.get_lab_threshold()], 
                              pixels_threshold=self.config.PIXELS_THRESHOLD,
                              area_threshold=self.config.AREA_THRESHOLD,
                              merge=self.config.MERGE_BLOBS)
        
        if not blobs:
            return None
        
        # 过滤面积范围外的区域
        valid_blobs = [blob for blob in blobs 
                      if self.config.MIN_AREA <= blob.area() <= self.config.MAX_AREA]
        
        if not valid_blobs:
            return None
        
        # 选择面积最大的blob作为激光点
        best_blob = max(valid_blobs, key=lambda b: b.area())
        
        # 计算中心点
        center = (best_blob.cx(), best_blob.cy())
        
        return {
            'center': center,
            'blob': best_blob,
            'area': best_blob.area(),
            'rect': (best_blob.x(), best_blob.y(), best_blob.w(), best_blob.h())
        }
    
    def draw_results(self, img, detection_result):
        """在图像上绘制检测结果"""
        if detection_result is None:
            return
        
        center = detection_result['center']
        area = detection_result['area']
        rect = detection_result['rect']
        color = self.config.COLOR_DETECTION
        
        # 绘制矩形框
        img.draw_rect(rect[0], rect[1], rect[2], rect[3], color, self.config.RECT_THICKNESS)
        
        # 绘制中心点
        if center:
            # 绘制中心圆点
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
            img.draw_string(text_x, center[1] + self.config.TEXT_OFFSET_Y_1, 
                           f"Laser Detected", color, scale=self.config.TEXT_SCALE_MEDIUM)
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
            fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
            self.current_fps = fps
        else:
            fps = getattr(self, 'current_fps', 0)
        
        # 显示FPS
        img.draw_string(self.config.STATUS_X, self.config.STATUS_Y_FPS, 
                       f"FPS: {fps:.1f}", self.config.COLOR_TEXT, 
                       scale=self.config.TEXT_SCALE_LARGE)
        
        # 显示标题
        img.draw_string(self.config.STATUS_X, 
                       img.height() - self.config.STATUS_Y_TITLE_OFFSET, 
                       "Red Laser Detection - Basic", self.config.COLOR_TEXT, 
                       scale=self.config.TEXT_SCALE_MEDIUM)
        
        # 显示控制提示
        img.draw_string(self.config.STATUS_X, 
                       img.height() - self.config.STATUS_Y_CONTROL1_OFFSET, 
                       "Pure Blob Detection", self.config.COLOR_TEXT, 
                       scale=self.config.TEXT_SCALE_SMALL)

def main():
    """主函数"""
    print("MaixPy 红色激光检测器启动中...")
    
    # 打印配置信息
    LaserDetectionConfig.print_config()
    
    # 初始化激光检测器
    detector = RedLaserDetector()
    
    # 初始化摄像头
    cam_config = LaserDetectionConfig.get_camera_config()
    cam = camera.Camera(cam_config['width'], cam_config['height'], cam_config['format'])
    
    # 初始化显示器
    disp = display.Display()
    
    print("系统已启动，开始检测红色激光...")
    
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
        if LaserDetectionConfig.ENABLE_CONSOLE_OUTPUT and detection_result:
            center = detection_result['center']
            area = detection_result['area']
            print(f"激光检测到 - 位置: ({center[0]}, {center[1]}), 面积: {area}")
        
        # 显示图像
        disp.show(img)
    
    print("程序退出")

if __name__ == "__main__":
    main()
