from maix import camera, display, image, app
import time

# ================================ 配置与常量定义 ================================
class FrameDiffLaserConfig:
    """帧差法激光检测器配置类 - 集中管理所有可配置项"""
    
    # ==================== 摄像头配置 ====================
    CAMERA_WIDTH = 320      # 摄像头宽度(像素) - 影响检测精度和性能
    CAMERA_HEIGHT = 240     # 摄像头高度(像素) - 影响检测精度和性能
    CAMERA_FORMAT = image.Format.FMT_GRAYSCALE  # 灰度图像格式 - 帧差法用灰度图更高效
    
    # ==================== 帧差法参数 ====================
    DIFF_THRESHOLD = 30     # 帧差阈值 - 像素差异超过此值认为有变化
    MIN_AREA = 5           # 最小激光点面积(像素) - 过滤噪点
    MAX_AREA = 500         # 最大激光点面积(像素) - 过滤大面积误检
    PIXELS_THRESHOLD = 5   # 最小像素数阈值 - blob检测的像素数要求
    AREA_THRESHOLD = 5     # 最小面积阈值 - blob检测的面积要求
    MERGE_BLOBS = True     # 是否合并相邻blob - 将分散的激光点合并
    
    # ==================== 形态学处理参数 ====================
    ERODE_SIZE = 1         # 腐蚀核大小 - 去除小噪点
    DILATE_SIZE = 2        # 膨胀核大小 - 连接分散区域
    
    # ==================== 显示参数 ====================
    # 绘制参数
    CIRCLE_RADIUS_INNER = 5     # 中心点内圆半径
    CIRCLE_RADIUS_OUTER = 15    # 中心点外圆半径
    CROSSHAIR_LENGTH = 20       # 十字准线长度
    RECT_THICKNESS = 2          # 矩形框线条粗细
    CIRCLE_THICKNESS = 2        # 圆圈线条粗细
    TEXT_OFFSET_X = 25         # 文本X轴偏移
    TEXT_OFFSET_Y_1 = -20      # 第一行文本Y轴偏移
    TEXT_OFFSET_Y_2 = 0        # 第二行文本Y轴偏移
    TEXT_OFFSET_Y_3 = 15       # 第三行文本Y轴偏移
    
    # 文本缩放比例
    TEXT_SCALE_LARGE = 1.2     # 大文本缩放
    TEXT_SCALE_MEDIUM = 1.0    # 中等文本缩放
    TEXT_SCALE_SMALL = 0.8     # 小文本缩放
    TEXT_SCALE_TINY = 0.7      # 微小文本缩放
    
    # 状态信息显示位置
    STATUS_X = 10              # 状态信息X坐标
    STATUS_Y_FPS = 10          # FPS显示Y坐标
    STATUS_Y_TITLE_OFFSET = 60 # 标题距离底部偏移
    STATUS_Y_CONTROL1_OFFSET = 40  # 控制提示距离底部偏移
    
    # ==================== 性能监控参数 ====================
    FPS_UPDATE_INTERVAL = 1.0  # FPS更新间隔(秒)
    
    # ==================== 颜色常量 ====================
    COLOR_DETECTION = image.COLOR_GREEN  # 检测到激光时的颜色
    COLOR_TEXT = image.COLOR_WHITE       # 文本颜色
    COLOR_DIFF = image.COLOR_RED         # 帧差显示颜色
    
    # ==================== 调试参数 ====================
    ENABLE_CONSOLE_OUTPUT = True        # 是否启用控制台输出
    SHOW_DIFF_IMAGE = False             # 是否显示帧差图像(调试用)
    
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
        print("=== 帧差法激光检测器配置 ===")
        print(f"帧差检测:")
        print(f"  差异阈值: {cls.DIFF_THRESHOLD}")
        print(f"  面积范围: {cls.MIN_AREA}-{cls.MAX_AREA} 像素")
        print(f"形态学处理:")
        print(f"  腐蚀核大小: {cls.ERODE_SIZE}")
        print(f"  膨胀核大小: {cls.DILATE_SIZE}")
        print(f"摄像头配置:")
        print(f"  分辨率: {cls.CAMERA_WIDTH}x{cls.CAMERA_HEIGHT}")
        print(f"  格式: 灰度图像")
        print("========================")

# ================================ 帧差法检测类 ================================
class FrameDiffLaserDetector:
    def __init__(self, config=FrameDiffLaserConfig):
        """初始化帧差法激光检测器"""
        self.config = config
        
        # 前一帧图像
        self.prev_frame = None
        
        # 显示相关
        self.frame_count = 0
        self.start_time = time.time()
        self.initialized = False
        
    def detect_laser(self, img):
        """使用帧差法检测激光点 - 优化版本"""
        # 如果是第一帧，保存并返回
        if self.prev_frame is None:
            self.prev_frame = img.copy()
            return None
        
        # 使用MaixPy的内置方法计算帧差，避免逐像素操作
        # 方法1：使用图像相减和阈值处理
        try:
            # 创建当前帧的副本
            curr_frame = img.copy()
            
            # 计算绝对差值
            # 使用图像处理方法而不是逐像素操作
            diff_img = image.Image(img.width(), img.height(), image.Format.FMT_GRAYSCALE)
            
            # 转换为numpy数组进行快速计算（如果支持）
            # 或者使用MaixPy的图像处理函数
            # 这里使用二值化方法来检测变化
            
            # 方案：使用连续帧的亮度变化检测
            # 将当前帧二值化，找出亮度高的区域
            binary_threshold = 200  # 激光通常很亮
            
            # 对当前帧进行二值化，找出亮点
            blobs = curr_frame.find_blobs([(binary_threshold, 255)],
                                         pixels_threshold=self.config.PIXELS_THRESHOLD,
                                         area_threshold=self.config.AREA_THRESHOLD,
                                         merge=self.config.MERGE_BLOBS)
            
            # 对前一帧进行二值化，找出亮点
            prev_blobs = self.prev_frame.find_blobs([(binary_threshold, 255)],
                                                   pixels_threshold=self.config.PIXELS_THRESHOLD,
                                                   area_threshold=self.config.AREA_THRESHOLD,
                                                   merge=self.config.MERGE_BLOBS)
            
            # 更新前一帧
            self.prev_frame = curr_frame.copy()
            
            if not blobs:
                return None
            
            # 找出新出现的或移动的blob
            moving_blobs = []
            
            for blob in blobs:
                # 检查这个blob是否在新位置（与前一帧的blob比较）
                is_moving = True
                
                if prev_blobs:
                    for prev_blob in prev_blobs:
                        # 计算中心点距离
                        dx = blob.cx() - prev_blob.cx()
                        dy = blob.cy() - prev_blob.cy()
                        distance = (dx*dx + dy*dy) ** 0.5
                        
                        # 如果距离太小，认为是静止的
                        if distance < 5:  # 5像素的移动阈值
                            is_moving = False
                            break
                
                if is_moving:
                    # 检查面积范围
                    if self.config.MIN_AREA <= blob.area() <= self.config.MAX_AREA:
                        moving_blobs.append(blob)
            
            if not moving_blobs:
                return None
            
            # 选择面积最大的移动blob作为激光点
            best_blob = max(moving_blobs, key=lambda b: b.area())
            
            # 计算中心点
            center = (best_blob.cx(), best_blob.cy())
            
            return {
                'center': center,
                'blob': best_blob,
                'area': best_blob.area(),
                'rect': (best_blob.x(), best_blob.y(), best_blob.w(), best_blob.h()),
                'diff_img': None
            }
            
        except Exception as e:
            print(f"检测错误: {e}")
            self.prev_frame = img.copy()
            return None
    
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
                           f"Laser Motion", color, scale=self.config.TEXT_SCALE_MEDIUM)
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
                       "Frame Difference Laser Detection", self.config.COLOR_TEXT, 
                       scale=self.config.TEXT_SCALE_MEDIUM)
        
        # 显示控制提示
        img.draw_string(self.config.STATUS_X, 
                       img.height() - self.config.STATUS_Y_CONTROL1_OFFSET, 
                       "Detects moving laser points", self.config.COLOR_TEXT, 
                       scale=self.config.TEXT_SCALE_SMALL)

def main():
    """主函数"""
    print("MaixPy 帧差法激光检测器启动中...")
    
    # 打印配置信息
    FrameDiffLaserConfig.print_config()
    
    # 初始化激光检测器
    detector = FrameDiffLaserDetector()
    
    # 初始化摄像头
    cam_config = FrameDiffLaserConfig.get_camera_config()
    cam = camera.Camera(cam_config['width'], cam_config['height'], cam_config['format'])
    
    # 初始化显示器
    disp = display.Display()
    
    print("系统已启动，开始检测移动的激光...")
    print("注意: 帧差法检测移动的激光点，静止激光不会被检测到")
    
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
        if FrameDiffLaserConfig.ENABLE_CONSOLE_OUTPUT and detection_result:
            center = detection_result['center']
            area = detection_result['area']
            print(f"移动激光检测到 - 位置: ({center[0]}, {center[1]}), 面积: {area}")
        
        # 显示图像
        disp.show(img)
    
    print("程序退出")

if __name__ == "__main__":
    main()