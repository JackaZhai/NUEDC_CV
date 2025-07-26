import cv2
import numpy as np
import os
from maix import camera, display, app, time, image, touchscreen

class PhotoCaptureApp:
    def __init__(self):
        self.cam = camera.Camera(480, 320)
        self.cam.skip_frames(10)
        self.disp = display.Display()
        self.ts = touchscreen.TouchScreen()
        
        # 状态变量
        self.is_capturing = False
        self.photo_count = 0
        self.folder_count = 0
        self.max_photos_per_folder = 100
        self.base_path = '/root/zjj_project'
        
        # 确保基础目录存在
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
        
        # 按钮位置定义
        self.setup_buttons()
        
    def setup_buttons(self):
        """设置按钮位置"""
        button_width = 100
        button_height = 40
        margin = 10
        
        # 开始/停止按钮
        self.start_btn = {
            'x': margin,
            'y': margin,
            'w': button_width,
            'h': button_height,
            'label': 'START'
        }
        
        # 退出按钮
        self.exit_btn = {
            'x': self.disp.width() - button_width - margin - 30,
            'y': margin,
            'w': button_width,
            'h': button_height,
            'label': 'EXIT'
        }
        
    def is_point_in_button(self, x, y, button):
        """检查点击是否在按钮范围内"""
        return (x >= button['x'] and x <= button['x'] + button['w'] and
                y >= button['y'] and y <= button['y'] + button['h'])
    
    def draw_button(self, img, button, pressed=False):
        """绘制按钮"""
        # 绘制按钮边框（白色）
        img.draw_rect(button['x'], button['y'], button['w'], button['h'], 
                     image.COLOR_WHITE, thickness=2)
        
        # 绘制按钮文字（白色）
        text_size = image.string_size(button['label'])
        text_x = button['x'] + (button['w'] - text_size.width()) // 2
        text_y = button['y'] + (button['h'] - text_size.height()) // 2 + text_size.height()
        
        img.draw_string(text_x, text_y, button['label'], image.COLOR_WHITE)
    
    def get_current_folder(self):
        """获取当前文件夹路径"""
        folder_name = f"batch_{self.folder_count:03d}"
        folder_path = os.path.join(self.base_path, folder_name)
        
        # 如果文件夹不存在则创建
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            
        return folder_path
    
    def capture_photo(self, img_maix):
        """拍照并保存"""
        try:
            # 检查是否需要换文件夹
            if self.photo_count >= self.max_photos_per_folder:
                self.folder_count += 1
                self.photo_count = 0
                # 自动停止拍照
                self.is_capturing = False
                self.start_btn['label'] = 'START'
                print(f"已拍满100张，自动停止！切换到新文件夹: batch_{self.folder_count:03d}")
                return
            
            # 获取当前文件夹
            folder_path = self.get_current_folder()
            
            # 转换图像格式并保存
            img_cv = image.image2cv(img_maix, ensure_bgr=True, copy=True)
            file_path = os.path.join(folder_path, f"{self.photo_count:03d}.png")
            cv2.imwrite(file_path, img_cv)
            
            self.photo_count += 1
            print(f"已保存照片: {file_path}")
            
        except Exception as e:
            print(f"保存照片失败: {e}")
    
    def draw_status_info(self, img):
        """绘制状态信息"""
        status_y = 60
        
        # 拍照状态
        status_text = "Capturing..." if self.is_capturing else "Stopped"
        status_color = image.COLOR_GREEN if self.is_capturing else image.COLOR_RED
        img.draw_string(10, status_y, f"Status: {status_text}", status_color)
        
        # 照片计数
        img.draw_string(10, status_y + 25, 
                       f"Folder: batch_{self.folder_count:03d}", 
                       image.COLOR_WHITE)
        
        img.draw_string(10, status_y + 50, 
                       f"Photos: {self.photo_count}/{self.max_photos_per_folder}", 
                       image.COLOR_WHITE)
        
        # 总计数
        total_photos = self.folder_count * self.max_photos_per_folder + self.photo_count
        img.draw_string(10, status_y + 75, 
                       f"Total: {total_photos}", 
                       image.COLOR_WHITE)
    
    def handle_touch(self, x, y):
        """处理触摸事件"""
        if self.is_point_in_button(x, y, self.start_btn):
            # 切换拍照状态
            self.is_capturing = not self.is_capturing
            if self.is_capturing:
                self.start_btn['label'] = 'STOP'
                print("开始拍照...")
            else:
                self.start_btn['label'] = 'START'
                print("停止拍照")
            
        elif self.is_point_in_button(x, y, self.exit_btn):
            # 退出应用
            print("退出应用...")
            app.set_exit_flag(True)
    
    def run(self):
        """主运行循环"""
        print("=" * 50)
        print("    照片批量采集应用")
        print("=" * 50)
        print("功能说明:")
        print("- 点击 '开始拍照' 按钮开始/停止拍照")
        print("- 每个文件夹最多保存100张照片")
        print("- 超过100张会自动创建新文件夹")
        print("- 点击 '退出' 按钮退出应用")
        print("=" * 50)
        
        last_capture_time = 0
        capture_interval = 100  # 100ms间隔拍照
        
        while not app.need_exit():
            try:
                # 读取摄像头图像
                img_maix = self.cam.read()
                
                # 处理触摸事件
                x, y, pressed = self.ts.read()
                if pressed:
                    self.handle_touch(x, y)
                
                # 如果正在拍照且时间间隔足够
                current_time = time.ticks_ms()
                if (self.is_capturing and 
                    current_time - last_capture_time > capture_interval):
                    self.capture_photo(img_maix)
                    last_capture_time = current_time
                
                # 绘制界面
                self.draw_button(img_maix, self.start_btn, self.is_capturing)
                self.draw_button(img_maix, self.exit_btn)
                self.draw_status_info(img_maix)
                
                # 显示图像
                self.disp.show(img_maix)
                
                time.sleep_ms(10)  # 短暂延时
                
            except KeyboardInterrupt:
                print("\n程序被用户中断")
                break
            except Exception as e:
                print(f"程序异常: {e}")
                break
        
        self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        print("正在清理资源...")
        try:
            if self.cam:
                self.cam.close()
            if self.disp:
                self.disp.close()
            print("资源清理完成")
        except Exception as e:
            print(f"清理资源时出错: {e}")

if __name__ == '__main__':
    app_instance = PhotoCaptureApp()
    app_instance.run()