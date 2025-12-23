import math
import os
import sys
import time
import json
import cv2
import torch
import random
import numpy as np
import win32con
from mss import mss
import tkinter as tk
from PIL import Image, ImageTk
import contextlib
import ctypes
import win32api


# 鼠标按键对应的虚拟键码
VK_LBUTTON = 0x01  # 左键
VK_RBUTTON = 0x02  # 右键
VK_MBUTTON = 0x04  # 中键
VK_XBUTTON1 = win32con.VK_XBUTTON1  # 侧后键

# 配置文件路径
CONFIG_FILE = "config.json"

# 默认配置参数
DEFAULT_CONFIG = {
    "sleep_time": 0.1,
    "click_time": 0.12,
    "display": False,
    "threshold": 0.3,
    "scale": 0.5,
    "size": 100,
    "capture_x": 640,
    "capture_y": 640,
    "target_fps": 60
}


@contextlib.contextmanager
def suppress_dll_output():
    """临时重定向 stdout/stderr，屏蔽 DLL 初始化时的控制台输出。"""
    devnull = open(os.devnull, 'w')
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        stdout_fd = os.dup(1)
        stderr_fd = os.dup(2)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        os.close(stdout_fd)
        os.close(stderr_fd)
        devnull.close()


def calculate_size_max(capture_x, capture_y):
    """根据 ROI 尺寸计算瞄准范围的动态上限。"""
    return max(10, min(capture_x, capture_y))


def load_config(config_file):
    """加载配置文件，如果不存在则返回默认配置"""
    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
            # 如果配置文件缺少某个参数，补上默认值
            for key, value in DEFAULT_CONFIG.items():
                if key not in config:
                    config[key] = value
            return config
        except Exception as e:
            print("加载配置失败，使用默认配置。", e)
    return DEFAULT_CONFIG.copy()


def save_config(config_file, config):
    """保存配置到文件"""
    try:
        with open(config_file, "w") as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        print("保存配置失败：", e)


class LGDriver:
    def __init__(self, dll_path, click_time, jitter_enabled):
        self.click_time = click_time
        self.jitter_enabled = jitter_enabled
        with suppress_dll_output():
            self.lg_driver = ctypes.CDLL(dll_path)
            self.ok = self.lg_driver.device_open() == 1
        if not self.ok:
            raise Exception("驱动加载失败!")

    def press(self, code):
        """
        按下指定的鼠标按钮
        code: 1: 左键, 2: 中键, 3: 右键
        """
        if not self.ok:
            return
        self.lg_driver.mouse_down(code)

    def release(self, code):
        """
        释放指定的鼠标按钮
        code: 1: 左键, 2: 中键, 3: 右键
        """
        if not self.ok:
            return
        self.lg_driver.mouse_up(code)
        print("clicked")

    def click(self, code=1):
        """
        点击指定的鼠标按钮
        code: 1: 左键, 2: 中键, 3: 右键
        """
        if not self.ok:
            return
        self.lg_driver.mouse_down(code)
        click_time = self.click_time.get()

        if self.jitter_enabled.get():
            time_variation = random.uniform(-0.02, 0.03)
            adjusted_click_time = max(click_time + time_variation, 0)
        else:
            adjusted_click_time = max(click_time, 0)
        self.microsecond_sleep(adjusted_click_time * 1000)
        self.lg_driver.mouse_up(code)
        print("clicked")

    def scroll(self, a):
        """
        滚动鼠标滚轮
        a: 滚动的距离
        """
        if not self.ok:
            return
        self.lg_driver.scroll(a)

    def move(self, x, y):
        """
        相对移动鼠标位置
        x: 水平移动的方向和距离, 正数向右, 负数向左
        y: 垂直移动的方向和距离
        """
        if not self.ok:
            return
        if x == 0 and y == 0:
            return
        # 屏蔽底层 DLL 可能打印的输出
        with suppress_dll_output():
            self.lg_driver.moveR(int(x), int(y), True)


    @staticmethod
    def microsecond_sleep(sleep_time):
        """
        微秒级睡眠，使用 Windows 高精度计时器实现
        :param sleep_time: int, 微秒 (1e-6 秒)
        """
        kernel32 = ctypes.WinDLL('kernel32')
        freq = ctypes.c_int64()
        start = ctypes.c_int64()
        end = ctypes.c_int64()

        kernel32.QueryPerformanceFrequency(ctypes.byref(freq))
        kernel32.QueryPerformanceCounter(ctypes.byref(start))

        target_ticks = int(freq.value * sleep_time / 1e6)
        while True:
            kernel32.QueryPerformanceCounter(ctypes.byref(end))
            if end.value - start.value >= target_ticks:
                break

    def smooth_move(self, x, y, min_steps=5, max_steps=20, scale_factor=5):
        """
        平滑相对移动鼠标位置，步数根据移动距离动态计算，并避免移动偏差
        参数:
            x: 水平移动的总距离（正数向右，负数向左）
            y: 垂直移动的总距离（正数向下，负数向上）
            min_steps: 最小步数，默认 5
            max_steps: 最大步数，默认 20
            scale_factor: 距离缩放因子，用于调整步数与距离的敏感度，默认 5
            delay: 每步之间的延迟时间（秒），默认 0.01 秒
        """
        if not self.ok:
            return
        if x == 0 and y == 0:
            return

        # 计算总移动距离
        distance = math.sqrt(x ** 2 + y ** 2)
        print("smooth_move distance:", distance)

        # 动态计算步数
        steps = int(min(max(distance / scale_factor, min_steps), max_steps))

        # 初始化累积误差
        error_x = 0.0
        error_y = 0.0

        # 计算每步的理论移动量
        step_x_float = x / steps
        step_y_float = y / steps

        # 分步执行移动
        for _ in range(steps):
            # 累积浮点移动量
            error_x += step_x_float
            error_y += step_y_float

            # 计算当前步的整数移动量
            move_x = int(round(error_x))
            move_y = int(round(error_y))

            # 更新累积误差
            error_x -= move_x
            error_y -= move_y

            # 执行移动（屏蔽底层 DLL 可能的控制台输出）
            with suppress_dll_output():
                self.lg_driver.moveR(move_x, move_y, True)
            self.microsecond_sleep(2000)

def initialize_model_and_driver(click_time, jitter_enabled, retries=3, delay=5):
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(__file__)
    repo_path = os.path.join(base_path, './yolov5-master')
    model_path = os.path.join(base_path, 'runs/train/exp3/weights/best.pt')
    driver_path = os.path.join(base_path, 'driver/logitech.driver.dll')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cuda.matmul.fp32_precision = 'tf32'
        torch.backends.cudnn.conv.fp32_precision = 'tf32'

    for attempt in range(retries):
        try:
            model = torch.hub.load(repo_or_dir=repo_path,
                                   model='custom',
                                   path=model_path,
                                   source='local').to(device)
            model.eval()
            if device.type == 'cuda':
                torch.backends.cudnn.benchmark = True
                model.half()

            dummy_dtype = torch.float16 if device.type == 'cuda' else torch.float32
            with torch.inference_mode():
                model(torch.zeros(1, 3, 640, 640, device=device, dtype=dummy_dtype))

            driver = LGDriver(driver_path, click_time, jitter_enabled)
            return model, driver
        except Exception as e:
            print(f"模型或驱动加载失败 (尝试 {attempt + 1}/{retries}): {e}")
            if "logitech.driver.dll" in str(e):
                print("提示：请确保安装了 driver/lghub 目录中的驱动程序。")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                return None, None


def create_control_panel(root, sleep_time_var, click_time, display_var, threshold, scale, size, tk_window,
                        monitor_var, monitor_options, on_monitor_change, jitter_enabled, capture_x_var,
                        capture_y_var, target_fps_var):
    panel_width, panel_height = 340, 420
    root.geometry(f"{panel_width}x{panel_height}+10+25")
    root.minsize(panel_width, panel_height)
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.attributes("-alpha", 0.5)

    def start_move(event):
        root._drag_offset_x = event.x_root - root.winfo_x()
        root._drag_offset_y = event.y_root - root.winfo_y()

    def do_move(event):
        x = event.x_root - root._drag_offset_x
        y = event.y_root - root._drag_offset_y
        root.geometry(f"+{x}+{y}")

    frame = tk.Frame(root, bg="black")
    frame.pack(fill="both", expand=True)
    frame.bind("<Button-1>", start_move)
    frame.bind("<B1-Motion>", do_move)
    for col in range(4):
        frame.grid_columnconfigure(col, weight=1, minsize=70)

    label_config = {"fg": "white", "bg": "black", "font": ("Arial", 12)}
    button_config = {"bg": "black", "fg": "white", "font": ("Arial", 12)}

    labels = [
        ("sleep_time:", sleep_time_var),
        ("click_time:", click_time),
        ("threshold:", threshold),
        ("size:", size),
        ("scale:", scale),
        ("capture_x:", capture_x_var),
        ("capture_y:", capture_y_var),
        ("target_fps:", target_fps_var)
    ]

    for i, (text, var) in enumerate(labels):
        tk.Label(frame, text=text, **label_config).grid(row=i, column=1, padx=5, pady=5)
        tk.Label(frame, textvariable=var, **label_config).grid(row=i, column=2, padx=5, pady=5)

    def resolve_bound(value):
        return value() if callable(value) else value

    def create_buttons(row, b_var, min_val, max_val, step, round_digits=2):
        tk.Button(
            frame,
            text=" + ",
            command=lambda: b_var.set(
                min(
                    round(b_var.get() + step, round_digits),
                    resolve_bound(max_val)
                )
            ),
            **button_config
        ).grid(row=row, column=3, padx=5, pady=5)
        tk.Button(
            frame,
            text=" - ",
            command=lambda: b_var.set(
                max(
                    round(b_var.get() - step, round_digits),
                    resolve_bound(min_val)
                )
            ),
            **button_config
        ).grid(row=row, column=0, padx=5, pady=5)

    size_max_value = lambda: calculate_size_max(capture_x_var.get(), capture_y_var.get())

    create_buttons(0, sleep_time_var, 0.01, 0.25, 0.01)
    create_buttons(1, click_time, 0.01, 0.25, 0.01)
    create_buttons(2, threshold, 0.1, 1.0, 0.1, 1)
    create_buttons(3, size, 10, size_max_value, 10, 0)
    create_buttons(4, scale, 0.1, 2.0, 0.1, 1)
    create_buttons(5, capture_x_var, 200, 1600, 50, 0)
    create_buttons(6, capture_y_var, 200, 1600, 50, 0)
    create_buttons(7, target_fps_var, 10, 240, 5, 0)

    def toggle_display():
        display_var.set(not display_var.get())
        if display_var.get():
            width = int(capture_x_var.get() * scale.get())
            height = int(capture_y_var.get() * scale.get())
            tk_window.geometry(f"{width}x{height}+0+0")
            tk_window.deiconify()
        else:
            tk_window.withdraw()

    tk.Button(frame, text="toggle", command=toggle_display, **button_config).grid(row=8, column=1, columnspan=1, padx=5, pady=5, sticky="ew")

    monitor_menu = tk.OptionMenu(frame, monitor_var, *monitor_options, command=lambda _: on_monitor_change())
    monitor_menu.config(bg="black", fg="white", activebackground="gray30", activeforeground="white",
                        highlightthickness=0, width=8)
    monitor_menu.grid(row=8, column=2, padx=5, pady=5, sticky="ew")

    jitter_checkbox = tk.Checkbutton(frame, text="jitter", variable=jitter_enabled, onvalue=True, offvalue=False,
                                     bg="black", fg="white", selectcolor="gray20", activebackground="black",
                                     activeforeground="white")
    jitter_checkbox.grid(row=8, column=3, padx=5, pady=5)


def create_tk_window(root, scale, capture_x_var, capture_y_var):
    width = int(capture_x_var.get() * scale.get())
    height = int(capture_y_var.get() * scale.get())

    tk_window = tk.Toplevel(root)
    tk_window.overrideredirect(True)
    tk_window.attributes("-topmost", True)
    tk_window.geometry(f"{width}x{height}+0+0")
    tk_window.attributes("-alpha", 1)
    tk_window.withdraw()

    tk_window.img_label = tk.Label(tk_window)
    tk_window.img_label.pack(fill="both", expand=True)

    def start_move(event):
        tk_window._drag_offset_x = event.x_root - tk_window.winfo_x()
        tk_window._drag_offset_y = event.y_root - tk_window.winfo_y()

    def do_move(event):
        x = event.x_root - tk_window._drag_offset_x
        y = event.y_root - tk_window._drag_offset_y
        tk_window.geometry(f"+{x}+{y}")

    tk_window.img_label.bind("<Button-1>", start_move)
    tk_window.img_label.bind("<B1-Motion>", do_move)

    tk_window.fps_label = tk.Label(tk_window, text="FPS: 0", fg="white", bg="black")
    tk_window.fps_label.place(relx=0.1, rely=0.1, anchor=tk.CENTER)

    return tk_window


def get_screen_center(monitor):
    screen_width = monitor['width']
    screen_height = monitor['height']
    # 多显示器环境下，监视器的左上角可能不是 (0, 0)，需要加上偏移量
    return monitor.get('left', 0) + screen_width // 2, monitor.get('top', 0) + screen_height // 2


def capture_screen(sct, capture_area):
    screen_img = np.array(sct.grab(capture_area))
    return cv2.cvtColor(screen_img, cv2.COLOR_BGRA2BGR)


def detect_enemy(model, img, capture_x, capture_y, confidence_threshold):
    device = next(model.parameters()).device
    img = np.ascontiguousarray(img)

    with torch.inference_mode():
        if device.type == 'cuda':
            with torch.amp.autocast(device_type='cuda'):
                results = model(img, size=640)
        else:
            results = model(img, size=640)
    detections = results.xyxy[0].cpu().numpy()
    enemy_head_results = []
    enemy_results = []

    for *xyxy, conf, cls in detections:
        if conf < confidence_threshold:
            continue
        center_x = (xyxy[0] + xyxy[2]) / 2
        center_y = (xyxy[1] + xyxy[3]) / 2
        relative_x = center_x - capture_x // 2
        relative_y = center_y - capture_y // 2
        distance_to_center = np.sqrt(relative_x ** 2 + relative_y ** 2)
        if model.names[int(cls)] == 'enemy_head':
            enemy_head_results.append((relative_x, relative_y + 4, xyxy, conf, distance_to_center))
        elif model.names[int(cls)] == 'enemy':
            enemy_results.append((relative_x, relative_y, xyxy, conf, distance_to_center))

    closest_enemy_head = min(enemy_head_results, key=lambda x: x[4])[:4] if enemy_head_results else []
    closest_enemy = min(enemy_results, key=lambda x: x[4])[:4] if enemy_results else []

    return closest_enemy_head, closest_enemy


def enable_dpi_awareness():
    """在 Windows 上启用 DPI 感知，避免缩放导致的坐标偏移。"""
    try:
        if hasattr(ctypes.windll, "shcore"):
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        elif hasattr(ctypes.windll.user32, "SetProcessDPIAware"):
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception as e:
        print("启用 DPI 感知失败：", e)


def damped_smooth_move(driver, relative_x, relative_y):
    """根据距离自适应阻尼并分步移动，兼顾大范围与精度。"""
    distance = math.hypot(relative_x, relative_y)
    if distance == 0:
        return

    # 距离越远阻尼越小，采用平方根衰减，近距离保持更高阻尼以防过冲
    normalized = min(distance / 500.0, 1.0)
    damping = 0.9 - 0.4 * math.sqrt(normalized)
    move_x = relative_x * damping
    move_y = relative_y * damping

    if distance > 600:
        min_steps, max_steps, scale_factor = 12, 40, 8
    elif distance > 250:
        min_steps, max_steps, scale_factor = 10, 34, 7
    else:
        min_steps, max_steps, scale_factor = 6, 28, 6

    driver.smooth_move(move_x, move_y, min_steps=min_steps, max_steps=max_steps, scale_factor=scale_factor)


def perform_action(driver, relative_x, relative_y, sleep_time, size, head_xyxy):
    abs_x = abs(relative_x)
    abs_y = abs(relative_y)
    xyxy = head_xyxy
    x1, y1, x2, y2 = xyxy
    xx = x2 - x1

    delta_size = size * (xx / 13)
    m_x = abs((x2 - x1) / 2)
    m_y = abs((y2 - y1) / 2)

    if abs_x < m_x and abs_y < m_y:
        driver.click()
        time.sleep(sleep_time)
    else:
        if abs_x <= delta_size and abs_y <= delta_size:
            damped_smooth_move(driver, relative_x, relative_y)
            driver.click()
            time.sleep(sleep_time)


def perform_action_body(driver, relative_x, relative_y, sleep_time, size, body_xyxy):
    x1, y1, x2, y2 = body_xyxy
    delta_y = y2 - y1
    adjustment_factor = 0.34 + 0.1 * (1 - math.exp(-0.01 * (delta_y - 50)))
    relative_y -= delta_y * adjustment_factor

    abs_x = abs(relative_x)
    abs_y = abs(relative_y)
    xx = x2 - x1
    delta_size = size * (xx / 50)

    if abs_x <= delta_size and abs_y <= delta_size:
        damped_smooth_move(driver, relative_x, relative_y)
        driver.click()
        time.sleep(sleep_time)


def display_image_with_detections(img, closest_enemy_head, closest_enemy, scale, tk_window):
    if closest_enemy_head:
        label_head = "Enemy Head"
        _, _, xyxy, conf = closest_enemy_head
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, f"{label_head} : {conf:0.3}", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

    if closest_enemy:
        label_enemy = "Enemy"
        _, _, xyxy, conf = closest_enemy
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, f"{label_enemy} : {conf:0.3}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1, cv2.LINE_AA)

    height, width = img.shape[:2]
    resized_img = cv2.resize(img, (int(width * scale), int(height * scale)))
    image = Image.fromarray(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
    tk_img = ImageTk.PhotoImage(image)
    tk_window.img_label.config(image=tk_img)
    tk_window.img_label.image = tk_img


def main():
    root = tk.Tk()
    # 加载配置文件
    config = load_config(CONFIG_FILE)
    # 根据配置初始化各个参数
    sleep_time_var = tk.DoubleVar(value=config.get("sleep_time", DEFAULT_CONFIG["sleep_time"]))
    click_time = tk.DoubleVar(value=config.get("click_time", DEFAULT_CONFIG["click_time"]))
    display_var = tk.BooleanVar(value=config.get("display", DEFAULT_CONFIG["display"]))
    threshold = tk.DoubleVar(value=config.get("threshold", DEFAULT_CONFIG["threshold"]))
    scale = tk.DoubleVar(value=config.get("scale", DEFAULT_CONFIG["scale"]))
    size = tk.DoubleVar(value=config.get("size", DEFAULT_CONFIG["size"]))
    capture_x_var = tk.IntVar(value=config.get("capture_x", DEFAULT_CONFIG["capture_x"]))
    capture_y_var = tk.IntVar(value=config.get("capture_y", DEFAULT_CONFIG["capture_y"]))
    target_fps_var = tk.IntVar(value=config.get("target_fps", DEFAULT_CONFIG["target_fps"]))
    jitter_enabled = tk.BooleanVar(value=True)

    # 定义当参数发生变化时自动保存到配置文件的回调函数
    def update_config(*args):
        config["sleep_time"] = sleep_time_var.get()
        config["click_time"] = click_time.get()
        config["display"] = display_var.get()
        config["threshold"] = threshold.get()
        config["scale"] = scale.get()
        config["size"] = size.get()
        config["capture_x"] = capture_x_var.get()
        config["capture_y"] = capture_y_var.get()
        config["target_fps"] = target_fps_var.get()
        save_config(CONFIG_FILE, config)

    # 为变量添加 trace，当值发生变化时触发保存
    sleep_time_var.trace_add("write", update_config)
    click_time.trace_add("write", update_config)
    display_var.trace_add("write", update_config)
    threshold.trace_add("write", update_config)
    scale.trace_add("write", update_config)
    size.trace_add("write", update_config)
    capture_x_var.trace_add("write", update_config)
    capture_y_var.trace_add("write", update_config)
    target_fps_var.trace_add("write", update_config)

    enable_dpi_awareness()
    sct = mss()
    monitors = sct.monitors
    monitor_var = tk.StringVar(value="Display 1")

    capture_area = {'top': 0, 'left': 0, 'width': capture_x_var.get(), 'height': capture_y_var.get()}

    def clamp_size_to_roi():
        max_size = calculate_size_max(capture_x_var.get(), capture_y_var.get())
        if size.get() > max_size:
            size.set(max_size)

    def update_display_geometry():
        width = int(capture_x_var.get() * scale.get())
        height = int(capture_y_var.get() * scale.get())
        tk_window.geometry(f"{width}x{height}+0+0")

    def update_capture_area():
        monitor_label = monitor_var.get()
        try:
            monitor_index = int(''.join(filter(str.isdigit, monitor_label)))
        except ValueError:
            monitor_index = 1
        if monitor_index <= 0 or monitor_index >= len(monitors):
            monitor_index = 1
            monitor_var.set(f"Display {monitor_index}")
        monitor = monitors[monitor_index]
        screen_center_x, screen_center_y = get_screen_center(monitor)
        left = screen_center_x - capture_x_var.get() // 2
        top = screen_center_y - capture_y_var.get() // 2
        capture_area.update({'top': top, 'left': left, 'width': capture_x_var.get(), 'height': capture_y_var.get()})
        clamp_size_to_roi()
        update_display_geometry()

    monitor_options = [f"Display {i}" for i in range(1, len(monitors))] if len(monitors) > 1 else ["Display 1"]

    tk_window = create_tk_window(root, scale, capture_x_var, capture_y_var)
    control_panel_visible = True
    create_control_panel(root, sleep_time_var, click_time, display_var, threshold, scale, size, tk_window,
                        monitor_var, monitor_options, update_capture_area, jitter_enabled, capture_x_var,
                        capture_y_var, target_fps_var)

    capture_x_var.trace_add("write", lambda *_: update_capture_area())
    capture_y_var.trace_add("write", lambda *_: update_capture_area())
    scale.trace_add("write", lambda *_: update_display_geometry())

    update_capture_area()

    model, driver = initialize_model_and_driver(click_time, jitter_enabled)

    previous_scale = scale.get()
    previous_capture_x = capture_x_var.get()
    previous_capture_y = capture_y_var.get()

    # 设定目标帧率并为热键与展示添加节流，避免长按时占用过高
    frame_interval = 1.0 / max(target_fps_var.get(), 1)

    hotkey_cooldown = 0.05  # 热键按住时的最小触发间隔，单位：秒
    last_hotkey_time = 0

    display_interval = 0.1  # 展示窗口的刷新间隔，单位：秒
    last_display_time = 0

    fps_state = {'last_time': time.time(), 'count': 0}
    fps_update_interval = 1.0

    while True:
        loop_start = time.time()  # 循环开始计时

        fps_state['count'] += 1
        if loop_start - fps_state['last_time'] >= fps_update_interval:
            elapsed_time = loop_start - fps_state['last_time']
            current_fps = fps_state['count'] / elapsed_time if elapsed_time > 0 else 0
            if tk_window and hasattr(tk_window, 'fps_label'):
                tk_window.fps_label.config(text=f"FPS: {current_fps:.1f}")
            fps_state['count'] = 0
            fps_state['last_time'] = loop_start

        frame_interval = 1.0 / max(target_fps_var.get(), 1)

        current_scale = scale.get()
        current_capture_x = capture_x_var.get()
        current_capture_y = capture_y_var.get()
        if (current_scale != previous_scale or current_capture_x != previous_capture_x or
                current_capture_y != previous_capture_y):
            update_display_geometry()
            previous_scale = current_scale
            previous_capture_x = current_capture_x
            previous_capture_y = current_capture_y

        current_time = time.time()
        img = None

        if win32api.GetAsyncKeyState(VK_RBUTTON) < 0:
            if current_time - last_hotkey_time >= hotkey_cooldown:
                last_hotkey_time = current_time
                img = capture_screen(sct, capture_area)
                closest_enemy_head, closest_enemy = detect_enemy(model, img, current_capture_x, current_capture_y, threshold.get())
                if closest_enemy_head and len(closest_enemy_head) > 2:
                    perform_action(driver, *closest_enemy_head[:2], sleep_time_var.get(), size.get(), closest_enemy_head[2])
                    continue
                if closest_enemy and len(closest_enemy) > 2:
                    perform_action_body(driver, *closest_enemy[:2], sleep_time_var.get(), size.get(), closest_enemy[2])

        if win32api.GetAsyncKeyState(win32con.VK_F6) < 0:
            click_time.set(0.01)
            size.set(150)

        if win32api.GetAsyncKeyState(win32con.VK_F5) < 0:
            click_time.set(0.12)
            size.set(100)

        if win32api.GetAsyncKeyState(win32con.VK_HOME) < 0:
            if control_panel_visible:
                root.withdraw()
            else:
                root.deiconify()
            control_panel_visible = not control_panel_visible
            time.sleep(0.2)

        if display_var.get():
            if current_time - last_display_time >= display_interval:
                if img is None:
                    img = capture_screen(sct, capture_area)
                closest_enemy_head, closest_enemy = detect_enemy(model, img, current_capture_x, current_capture_y, threshold.get())
                display_image_with_detections(img, closest_enemy_head, closest_enemy, scale.get(), tk_window)
                last_display_time = current_time

        if (win32api.GetAsyncKeyState(win32con.VK_SHIFT) < 0 and
                win32api.GetAsyncKeyState(win32con.VK_ESCAPE) < 0):
            print("退出程序中...")
            break

        root.update_idletasks()
        root.update()

        # 控制帧率
        elapsed_time = time.time() - loop_start
        remaining_time = frame_interval - elapsed_time
        if remaining_time > 0:
            time.sleep(remaining_time)

if __name__ == "__main__":
    print("主进程PID:", os.getpid())
    main()
