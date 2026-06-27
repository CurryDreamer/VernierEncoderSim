# -*- coding: utf-8 -*-
"""
游标双齿轮关节绝对值编码器联动仿真器
================================================

本工具用于验证：利用转子侧 10:9（或自定义齿比）的游标齿轮，
解算 9:1（或自定义减速比）关节输出轴绝对位置（0 ~ P-1 圈）的
数学可行性与噪声极限。

技术栈：Python3 + PySide6 + pyqtgraph + numpy
运行方式：python main.py
依赖安装：pip install PySide6 numpy pyqtgraph

整体说明（核心物理/数学模型见 SimulationModel）：
  * 大表盘 = 关节输出轴，可鼠标拖拽，拖拽角度 φ ∈ [0, 1) 圈。
  * 主齿轮（转子）随之旋转 Θ_m = φ × R 圈（顺时针）。
  * 副齿轮物理旋转 Θ_s = -Θ_m × (Z1/Z2) 圈（逆时针）。
  * 利用主/副齿轮归一化角度的相位差做游标解算，得到绝对圈数 T。
"""

import sys
import math

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

# ----------------------------------------------------------------------------
# 全局常量
# ----------------------------------------------------------------------------
FULL_SCALE = 16384        # 14 位编码器满量程（2^14）
MAX_RAW = FULL_SCALE - 1  # 16383，读数最大值
SAMPLE_COUNT = 64         # 软件滤波采样点数（模拟 64ms 采样窗口）
CURVE_LEN = 600           # 实时曲线缓冲长度（横向点数）


# ============================================================================
# 一、核心物理 / 数学模型
# ============================================================================
class SimulationModel:
    """游标编码器解算模型。

    所有的状态（齿数、减速比、零位偏置、误差注入幅度）都保存在此对象中，
    GUI 仅负责把参数写进来并读取解算结果，做到逻辑与界面分离。
    """

    def __init__(self):
        # ---- 可配置参数 ----
        self.Z1 = 10          # 主齿轮齿数
        self.Z2 = 9           # 副齿轮齿数
        self.R = 9            # 关节减速比

        # ---- 零位标定偏置（LSB），标定时记录当前读数 ----
        self.main_angle_offset = 0.0
        self.sub_angle_offset = 0.0

        # ---- 误差注入幅度（单位 LSB，0~2000）----
        self.noise_lsb = 0        # 高频随机噪声
        self.backlash_lsb = 0     # 机械齿轮背隙（回差）
        self.distortion_lsb = 0   # 断崖式磁畸变

        # ---- 背隙模型需要的内部状态 ----
        self._last_phi = 0.0      # 上一次的 φ，用于判断运动方向
        self._backlash_state = 0.0  # 当前已累积的回差偏移

    # ------------------------------------------------------------------
    @property
    def vernier_period(self):
        """游标转子绝对圈数周期 P = Z2 / GCD(Z1, Z2)。

        P 表示主齿轮转过多少圈后，(主齿轮, 副齿轮) 的相位组合才会重复，
        因此 P 就是我们能够无歧义分辨的绝对圈数个数。
        """
        g = math.gcd(int(self.Z1), int(self.Z2))
        if g == 0:
            return 1
        return int(self.Z2) // g

    # ------------------------------------------------------------------
    def _raw_main_sub(self, phi):
        """第 1 步：物理联动模拟。

        输入 φ ∈ [0,1)（输出轴当前圈位置），返回主/副齿轮的原始编码器读数。
        """
        # 转子（主齿轮）绝对物理位置（圈）：随输出轴放大 R 倍
        theta_m = phi * self.R
        # 副齿轮绝对物理位置（圈）：物理上反向，且按齿比传动
        theta_s = -theta_m * (self.Z1 / self.Z2)

        # 编码器读数：把"圈"换算成 14 位读数，并对满量程取模（编码器只输出单圈值）
        raw_main = (theta_m * FULL_SCALE) % FULL_SCALE
        raw_sub_phys = (theta_s * FULL_SCALE) % FULL_SCALE
        return raw_main, raw_sub_phys

    # ------------------------------------------------------------------
    def _inject_errors(self, raw_sec, phi):
        """第 2 步：仅对副齿轮输入端注入误差。

        包含：高频随机噪声、机械背隙（回差）、断崖式磁畸变。
        最后做循环取模限幅，保证落在 0~16383。
        """
        # (a) 高频随机噪声：高斯分布，标准差取幅度的 1/3，使 ±3σ ≈ 设定幅度
        if self.noise_lsb > 0:
            raw_sec += np.random.normal(0.0, self.noise_lsb / 3.0)

        # (b) 机械齿轮背隙：换向时读数滞后。
        #     通过运动方向判断，方向改变时回差状态平滑切换到对应符号一侧。
        if self.backlash_lsb > 0:
            direction = phi - self._last_phi
            if direction > 1e-9:
                target = +self.backlash_lsb / 2.0
            elif direction < -1e-9:
                target = -self.backlash_lsb / 2.0
            else:
                target = self._backlash_state
            # 回差不会瞬间消除，用一阶低通逼近，模拟"啮合面贴合"的迟滞过程
            self._backlash_state += (target - self._backlash_state) * 0.5
            raw_sec += self._backlash_state
        self._last_phi = phi

        # (c) 断崖式磁畸变：当读数落在 4000~6500 区间时，叠加大跨度畸变波形。
        #     这里用大振幅正弦模拟磁力线交叉串扰，区间边缘畸变为 0、中心最大，
        #     表现为一段"先冲高再跌落"的断崖形态。
        if self.distortion_lsb > 0 and 4000.0 <= raw_sec <= 6500.0:
            span = 6500.0 - 4000.0
            phase = (raw_sec - 4000.0) / span  # 0~1
            raw_sec += self.distortion_lsb * math.sin(math.pi * phase)

        # 循环取模限幅，保证读数始终是合法的单圈值
        raw_sec = raw_sec % FULL_SCALE
        return raw_sec

    # ------------------------------------------------------------------
    def _filter_64(self, raw_sec):
        """第 3 步：防跨零点跳变的 64 次采样均值滤波。

        直接对 raw 求平均，在 0/16383 跨零点处会得到错误的中间值（如 8000），
        因此采用"差分累加平均法"：以第一个采样为基准，把其余采样折算到
        [-8192, 8192) 的相对差值再做平均，最后加回基准值。
        """
        # 模拟 64ms 内带少量高频抖动的采样数组（±2 LSB 抖动）
        samples = raw_sec + np.random.uniform(-2.0, 2.0, size=SAMPLE_COUNT)
        samples = np.mod(samples, FULL_SCALE)

        first_val = samples[0]
        diff = samples - first_val
        # 跨零点修正：差值过大说明发生了 0/16383 的绕回，折算回最近一圈
        diff = np.where(diff > 8192, diff - FULL_SCALE, diff)
        diff = np.where(diff < -8192, diff + FULL_SCALE, diff)
        sum_diff = np.sum(diff)
        raw_sec_avg = first_val + sum_diff / float(SAMPLE_COUNT)
        return raw_sec_avg

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize(raw_zero):
        """把去偏置后的读数归一化到 [0,1)；负数 +1，超过 1 则 -1。"""
        v = raw_zero / float(FULL_SCALE)
        if v < 0.0:
            v += 1.0
        elif v >= 1.0:
            v -= 1.0
        return v

    # ------------------------------------------------------------------
    def solve(self, phi):
        """完整解算流程：输入 φ，返回包含所有中间量与结果的字典。"""
        P = self.vernier_period

        # 第 1 步：物理联动 -> 原始读数
        raw_main, raw_sub_phys = self._raw_main_sub(phi)

        # 第 2 步：副齿轮软件方向对齐（硬件级反转补偿）后注入误差
        raw_sec = MAX_RAW - raw_sub_phys
        raw_sec = self._inject_errors(raw_sec, phi)

        # 第 3 步：64 次采样均值滤波
        raw_sec_avg = self._filter_64(raw_sec)

        # 第 4 步：零位标定偏置 + 归一化
        raw_main_zero = raw_main - self.main_angle_offset
        raw_sub_zero = raw_sec_avg - self.sub_angle_offset
        normalized_main = self._normalize(raw_main_zero)
        normalized_sub = self._normalize(raw_sub_zero)

        # 第 5 步：游标解算绝对圈数 T
        # (1) 相位差：副 - 主，若为负补一圈，得到 [0,1) 内的相位差
        delta = normalized_sub - normalized_main
        if delta < 0.0:
            delta += 1.0
        # (2) 相位差放大 P 倍，得到绝对位置的估计（含主齿轮的小数部分）
        X_estimate = delta * P
        # (3) 用 round 提取整数圈数：减去主齿轮小数部分后四舍五入。
        #     round 的作用是"吸收噪声"——只要噪声没有把估计值推过 0.5 的判定边界，
        #     就能恢复出正确的整数圈数，这正是游标法抗噪的关键。
        T = int(round(X_estimate - normalized_main))
        # (4) 圈数循环约束到 [0, P)
        T = (T % P + P) % P

        # 输出轴绝对角度：φ 即输出轴单圈位置，换算成度数
        output_angle_deg = phi * 360.0

        return {
            "phi": phi,
            "P": P,
            "raw_main": raw_main,
            "raw_sub_phys": raw_sub_phys,
            "raw_sec": raw_sec,
            "raw_sec_avg": raw_sec_avg,
            "normalized_main": normalized_main,
            "normalized_sub": normalized_sub,
            "delta": delta,
            "X_estimate": X_estimate,
            "T": T,
            "output_angle_deg": output_angle_deg,
        }

    # ------------------------------------------------------------------
    def current_raw_for_calibration(self, phi):
        """返回当前 φ 对应的 (raw_main, raw_sec_avg)，供零位标定记录偏置。"""
        raw_main, raw_sub_phys = self._raw_main_sub(phi)
        raw_sec = MAX_RAW - raw_sub_phys
        raw_sec = self._inject_errors(raw_sec, phi)
        raw_sec_avg = self._filter_64(raw_sec)
        return raw_main, raw_sec_avg


# ============================================================================
# 二、中间面板：输出端时钟大表盘（可鼠标拖拽）
# ============================================================================
class ClockDial(QtWidgets.QWidget):
    """关节输出轴时钟表盘。支持鼠标按住任意位置拖拽旋转指针。"""

    # 角度变化信号：发出当前 φ ∈ [0,1)
    phiChanged = QtCore.Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._phi = 0.0          # 当前圈位置 [0,1)
        self._dragging = False
        self.setMinimumSize(260, 260)
        self.setCursor(QtCore.Qt.OpenHandCursor)

    # ---- 对外接口 ----
    def phi(self):
        return self._phi

    def set_phi(self, phi):
        self._phi = phi % 1.0
        self.update()

    # ---- 鼠标交互：无死角拖拽 ----
    def _angle_from_pos(self, pos):
        """根据鼠标位置计算角度（0° 在 12 点钟方向，顺时针为正），返回 φ。"""
        c = QtCore.QPointF(self.width() / 2.0, self.height() / 2.0)
        dx = pos.x() - c.x()
        dy = pos.y() - c.y()
        # atan2(dx, -dy)：使 0 指向正上方，顺时针增大
        ang = math.degrees(math.atan2(dx, -dy))
        if ang < 0:
            ang += 360.0
        return ang / 360.0

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self._dragging = True
            self.setCursor(QtCore.Qt.ClosedHandCursor)
            self.set_phi(self._angle_from_pos(event.position()))
            self.phiChanged.emit(self._phi)

    def mouseMoveEvent(self, event):
        if self._dragging:
            self.set_phi(self._angle_from_pos(event.position()))
            self.phiChanged.emit(self._phi)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self._dragging = False
            self.setCursor(QtCore.Qt.OpenHandCursor)

    # ---- 绘制表盘 ----
    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        side = min(self.width(), self.height())
        cx, cy = self.width() / 2.0, self.height() / 2.0
        radius = side / 2.0 - 14

        # 外圈渐变表盘底
        grad = QtGui.QRadialGradient(cx, cy, radius)
        grad.setColorAt(0.0, QtGui.QColor(45, 48, 58))
        grad.setColorAt(1.0, QtGui.QColor(20, 22, 28))
        p.setBrush(QtGui.QBrush(grad))
        p.setPen(QtGui.QPen(QtGui.QColor(90, 170, 255), 3))
        p.drawEllipse(QtCore.QPointF(cx, cy), radius, radius)

        # 刻度：36 个，每 10° 一个，整点更长
        p.save()
        p.translate(cx, cy)
        for i in range(36):
            p.save()
            p.rotate(i * 10)
            if i % 3 == 0:
                p.setPen(QtGui.QPen(QtGui.QColor(220, 220, 230), 2))
                p.drawLine(0, int(-radius), 0, int(-radius + 16))
            else:
                p.setPen(QtGui.QPen(QtGui.QColor(120, 120, 130), 1))
                p.drawLine(0, int(-radius), 0, int(-radius + 9))
            p.restore()
        p.restore()

        # 指针：根据 φ 旋转（0 在正上方，顺时针）
        p.save()
        p.translate(cx, cy)
        p.rotate(self._phi * 360.0)
        # 指针主体
        pointer = QtGui.QPolygonF([
            QtCore.QPointF(0, -radius + 24),
            QtCore.QPointF(-9, 12),
            QtCore.QPointF(9, 12),
        ])
        p.setBrush(QtGui.QColor(255, 90, 90))
        p.setPen(QtCore.Qt.NoPen)
        p.drawPolygon(pointer)
        # 尾针
        p.setBrush(QtGui.QColor(90, 170, 255))
        p.drawPolygon(QtGui.QPolygonF([
            QtCore.QPointF(0, radius * 0.32),
            QtCore.QPointF(-6, 0),
            QtCore.QPointF(6, 0),
        ]))
        p.restore()

        # 中心轴帽
        p.setBrush(QtGui.QColor(230, 230, 235))
        p.setPen(QtCore.Qt.NoPen)
        p.drawEllipse(QtCore.QPointF(cx, cy), 9, 9)

        # 中心文字：当前角度
        p.setPen(QtGui.QColor(180, 220, 255))
        f = p.font()
        f.setPointSize(11)
        p.setFont(f)
        p.drawText(self.rect().adjusted(0, int(radius * 0.55), 0, 0),
                   QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop,
                   f"{self._phi * 360.0:.1f}°")
        p.end()


# ============================================================================
# 三、中间面板：主/副齿轮联动小表盘
# ============================================================================
class GearPairWidget(QtWidgets.QWidget):
    """绘制两个咬合的小齿轮，随主表盘旋转实时联动。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.Z1 = 10
        self.Z2 = 9
        self.main_deg = 0.0   # 主齿轮当前角度（度，顺时针）
        self.sub_deg = 0.0    # 副齿轮当前角度（度，逆时针）
        self.setMinimumHeight(190)

    def set_state(self, Z1, Z2, main_deg, sub_deg):
        self.Z1, self.Z2 = Z1, Z2
        self.main_deg, self.sub_deg = main_deg, sub_deg
        self.update()

    def _draw_gear(self, p, cx, cy, r, teeth, angle_deg, color, label):
        """画一个简化齿轮：圆 + 沿圆周的矩形齿 + 一根标记线表示转动。"""
        p.save()
        p.translate(cx, cy)
        p.rotate(angle_deg)
        # 齿
        p.setBrush(color)
        p.setPen(QtGui.QPen(color.darker(140), 1))
        teeth = max(1, int(teeth))
        tooth_w = max(2.0, 2.0 * math.pi * r / teeth * 0.45)
        for i in range(teeth):
            p.save()
            p.rotate(i * 360.0 / teeth)
            p.drawRect(QtCore.QRectF(-tooth_w / 2.0, -r - 6, tooth_w, 9))
            p.restore()
        # 轮体
        body = QtGui.QRadialGradient(0, 0, r)
        body.setColorAt(0.0, color.lighter(140))
        body.setColorAt(1.0, color.darker(120))
        p.setBrush(QtGui.QBrush(body))
        p.setPen(QtGui.QPen(color.darker(160), 1.5))
        p.drawEllipse(QtCore.QPointF(0, 0), r, r)
        # 标记线（看转动）
        p.setPen(QtGui.QPen(QtGui.QColor(20, 20, 20), 2))
        p.drawLine(0, 0, 0, int(-r + 6))
        # 中心孔
        p.setBrush(QtGui.QColor(30, 30, 36))
        p.setPen(QtCore.Qt.NoPen)
        p.drawEllipse(QtCore.QPointF(0, 0), r * 0.18, r * 0.18)
        p.restore()

    def paintEvent(self, event):
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cy = h / 2.0
        # 两齿轮半径按齿数比例，并让它们刚好咬合（圆心距 = r1 + r2）
        r1 = min(h * 0.32, w * 0.20)
        r2 = r1 * (self.Z2 / max(1, self.Z1))
        gap = 6  # 齿啮合视觉间隙
        total = r1 + r2
        cx1 = w / 2.0 - total / 2.0
        cx2 = w / 2.0 + total / 2.0 - gap

        self._draw_gear(p, cx1, cy, r1, self.Z1, self.main_deg,
                        QtGui.QColor(220, 70, 70), "主")
        self._draw_gear(p, cx2, cy, r2, self.Z2, self.sub_deg,
                        QtGui.QColor(70, 200, 110), "副")

        # 标注齿数
        p.setPen(QtGui.QColor(230, 230, 235))
        f = p.font()
        f.setPointSize(10)
        f.setBold(True)
        p.setFont(f)
        p.drawText(QtCore.QRectF(cx1 - r1, cy + r1 + 6, 2 * r1, 22),
                   QtCore.Qt.AlignCenter, f"主齿轮 Z1={self.Z1}")
        p.drawText(QtCore.QRectF(cx2 - r2, cy + r2 + 6, 2 * r2, 22),
                   QtCore.Qt.AlignCenter, f"副齿轮 Z2={self.Z2}")
        p.end()


# ============================================================================
# 四、主窗口：组装左/中/右三大面板并完成联动
# ============================================================================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("游标双齿轮关节绝对值编码器联动仿真器")
        self.resize(1500, 820)

        self.model = SimulationModel()

        # 曲线滚动缓冲（预分配，避免频繁内存分配 -> 无内存泄露）
        self.buf_main = np.zeros(CURVE_LEN)
        self.buf_sub = np.zeros(CURVE_LEN)
        self.buf_delta = np.zeros(CURVE_LEN)
        self.x_axis = np.arange(CURVE_LEN)

        # 组装界面
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.addWidget(self._build_left_panel(), 0)
        root.addWidget(self._build_center_panel(), 1)
        root.addWidget(self._build_right_panel(), 2)

        # 实时刷新定时器：即使不拖拽也持续推送采样，让噪声/曲线连续滚动
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(40)  # ~25 FPS

        self._sync_model_from_ui()
        self._update_all(self.dial.phi())

    # ------------------------------------------------------------------
    # A. 左侧面板：参数配置与误差注入
    # ------------------------------------------------------------------
    def _build_left_panel(self):
        panel = QtWidgets.QWidget()
        panel.setFixedWidth(320)
        v = QtWidgets.QVBoxLayout(panel)

        # ---- 参数配置组 ----
        cfg = QtWidgets.QGroupBox("参数配置")
        form = QtWidgets.QFormLayout(cfg)
        self.sp_z1 = QtWidgets.QSpinBox(); self.sp_z1.setRange(1, 200); self.sp_z1.setValue(10)
        self.sp_z2 = QtWidgets.QSpinBox(); self.sp_z2.setRange(1, 200); self.sp_z2.setValue(9)
        self.sp_r = QtWidgets.QSpinBox(); self.sp_r.setRange(1, 500); self.sp_r.setValue(9)
        for sp in (self.sp_z1, self.sp_z2, self.sp_r):
            sp.valueChanged.connect(self._on_param_changed)
        form.addRow("主齿轮齿数 Z1：", self.sp_z1)
        form.addRow("副齿轮齿数 Z2：", self.sp_z2)
        form.addRow("关节减速比 R：", self.sp_r)
        self.lbl_period = QtWidgets.QLabel()
        self.lbl_period.setStyleSheet("font-weight:bold;color:#5aa0ff;")
        form.addRow("游标周期 P：", self.lbl_period)
        v.addWidget(cfg)

        # ---- 零位标定组 ----
        calib = QtWidgets.QGroupBox("零位标定")
        cv = QtWidgets.QVBoxLayout(calib)
        self.btn_calib = QtWidgets.QPushButton("标定当前位置为物理零点")
        self.btn_calib.setMinimumHeight(54)
        self.btn_calib.setStyleSheet(
            "QPushButton{font-size:15px;font-weight:bold;background:#2d6cdf;"
            "color:white;border-radius:8px;}"
            "QPushButton:hover{background:#3f7df0;}")
        self.btn_calib.clicked.connect(self._on_calibrate)
        cv.addWidget(self.btn_calib)
        self.lbl_calib = QtWidgets.QLabel("尚未标定（偏置=0）")
        self.lbl_calib.setStyleSheet("color:#aaa;")
        cv.addWidget(self.lbl_calib)
        v.addWidget(calib)

        # ---- 输出轴定位组（数值输入，直接切换表盘读数）----
        pos = QtWidgets.QGroupBox("输出轴定位")
        pf = QtWidgets.QFormLayout(pos)
        self.sp_angle = QtWidgets.QDoubleSpinBox()
        self.sp_angle.setRange(0.0, 360.0)
        self.sp_angle.setDecimals(1)
        self.sp_angle.setSingleStep(1.0)
        self.sp_angle.setWrapping(True)       # 0/360 处循环，便于连续转动
        self.sp_angle.setSuffix(" °")
        self.sp_angle.valueChanged.connect(self._on_angle_input)
        pf.addRow("输出轴角度：", self.sp_angle)
        v.addWidget(pos)

        # ---- 编码器原始读数组（实时显示当前读数 0~16383，不画曲线）----
        rawbox = QtWidgets.QGroupBox("编码器原始读数 (0~16383)")
        rf = QtWidgets.QFormLayout(rawbox)
        big = "font-size:20px;font-weight:bold;color:#ffd34d;font-family:monospace;"
        self.lbl_raw_main = QtWidgets.QLabel("0")
        self.lbl_raw_main.setStyleSheet(big)
        self.lbl_raw_sub = QtWidgets.QLabel("0")
        self.lbl_raw_sub.setStyleSheet(big.replace("#ffd34d", "#5ad06a"))
        rf.addRow("主编码器读数：", self.lbl_raw_main)
        rf.addRow("副编码器读数：", self.lbl_raw_sub)
        v.addWidget(rawbox)

        # ---- 误差模拟组 ----
        v.addWidget(self._build_error_group())
        v.addStretch(1)
        return panel

    def _build_error_group(self):
        grp = QtWidgets.QGroupBox("误差模拟（注入到副齿轮读数）")
        g = QtWidgets.QVBoxLayout(grp)

        def make_slider(title):
            box = QtWidgets.QVBoxLayout()
            lbl = QtWidgets.QLabel(f"{title}：0 LSB")
            s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            s.setRange(0, 2000)
            s.setValue(0)
            box.addWidget(lbl)
            box.addWidget(s)
            g.addLayout(box)
            return s, lbl

        self.sl_noise, self.lbl_noise = make_slider("高频随机噪声")
        self.sl_backlash, self.lbl_backlash = make_slider("机械齿轮背隙")
        self.sl_distort, self.lbl_distort = make_slider("断崖式磁畸变")

        self.sl_noise.valueChanged.connect(
            lambda val: (self.lbl_noise.setText(f"高频随机噪声：{val} LSB"),
                         setattr(self.model, "noise_lsb", val)))
        self.sl_backlash.valueChanged.connect(
            lambda val: (self.lbl_backlash.setText(f"机械齿轮背隙：{val} LSB"),
                         setattr(self.model, "backlash_lsb", val)))
        self.sl_distort.valueChanged.connect(
            lambda val: (self.lbl_distort.setText(f"断崖式磁畸变：{val} LSB"),
                         setattr(self.model, "distortion_lsb", val)))
        return grp

    # ------------------------------------------------------------------
    # B. 中间面板：表盘 + 齿轮联动
    # ------------------------------------------------------------------
    def _build_center_panel(self):
        panel = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(panel)
        title = QtWidgets.QLabel("关节输出轴（拖拽指针驱动联动）")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("font-size:15px;font-weight:bold;color:#ddd;")
        v.addWidget(title)

        self.dial = ClockDial()
        self.dial.phiChanged.connect(self._on_dial_moved)
        v.addWidget(self.dial, 1)

        sub_title = QtWidgets.QLabel("主 / 副齿轮啮合联动")
        sub_title.setAlignment(QtCore.Qt.AlignCenter)
        sub_title.setStyleSheet("font-size:13px;color:#bbb;")
        v.addWidget(sub_title)

        self.gears = GearPairWidget()
        v.addWidget(self.gears)
        return panel

    # ------------------------------------------------------------------
    # C. 右侧面板：实时曲线 + 圈数解算
    # ------------------------------------------------------------------
    def _build_right_panel(self):
        panel = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(panel)

        pg.setConfigOptions(antialias=True)
        self.plot = pg.PlotWidget()
        self.plot.setBackground("k")
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setYRange(-0.05, 1.05)
        self.plot.addLegend(offset=(-10, 10))
        self.plot.setLabel("left", "归一化角度")
        self.plot.setLabel("bottom", "采样序列（向左滚动）")
        self.curve_main = self.plot.plot(pen=pg.mkPen("r", width=2), name="主齿轮归一化角度")
        self.curve_sub = self.plot.plot(pen=pg.mkPen("g", width=2), name="副齿轮归一化角度(含噪)")
        self.curve_delta = self.plot.plot(pen=pg.mkPen((90, 160, 255), width=2), name="相位差 Delta")
        v.addWidget(self.plot, 1)

        v.addWidget(self._build_result_box())
        return panel

    def _build_result_box(self):
        box = QtWidgets.QGroupBox("绝对位置解算结果")
        bv = QtWidgets.QVBoxLayout(box)
        self.lbl_turns = QtWidgets.QLabel("绝对圈数 T = 0")
        self.lbl_turns.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_turns.setStyleSheet(
            "font-size:46px;font-weight:bold;color:#ffd34d;")
        self.lbl_out_angle = QtWidgets.QLabel("输出轴角度 = 0.0°")
        self.lbl_out_angle.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_out_angle.setStyleSheet(
            "font-size:30px;font-weight:bold;color:#5ad0ff;")
        bv.addWidget(self.lbl_turns)
        bv.addWidget(self.lbl_out_angle)
        return box

    # ------------------------------------------------------------------
    # 联动逻辑
    # ------------------------------------------------------------------
    def _sync_model_from_ui(self):
        """把界面参数同步到模型，并刷新游标周期 P 显示。"""
        self.model.Z1 = self.sp_z1.value()
        self.model.Z2 = self.sp_z2.value()
        self.model.R = self.sp_r.value()
        self.lbl_period.setText(str(self.model.vernier_period))

    def _on_param_changed(self, *_):
        self._sync_model_from_ui()
        self._update_all(self.dial.phi())

    def _on_calibrate(self):
        """标定：记录当前主/副读数作为零位偏置。"""
        phi = self.dial.phi()
        raw_main, raw_sec_avg = self.model.current_raw_for_calibration(phi)
        self.model.main_angle_offset = raw_main
        self.model.sub_angle_offset = raw_sec_avg
        self.lbl_calib.setText(
            f"已标定：主偏置={raw_main:.0f}  副偏置={raw_sec_avg:.0f}")
        self._update_all(phi)

    def _on_dial_moved(self, phi):
        # 拖拽表盘时，把角度回填到输入框（阻塞信号避免循环触发）
        self.sp_angle.blockSignals(True)
        self.sp_angle.setValue(phi * 360.0)
        self.sp_angle.blockSignals(False)
        self._update_all(phi)

    def _on_angle_input(self, deg):
        """从输入框直接设定输出轴角度，驱动整个联动系统。"""
        phi = (deg / 360.0) % 1.0
        self.dial.set_phi(phi)   # 仅刷新表盘显示，不再回发信号
        self._update_all(phi)

    def _tick(self):
        """定时器：持续以当前 φ 推进，让噪声与曲线连续刷新。"""
        self._update_all(self.dial.phi())

    def _update_all(self, phi):
        """核心刷新：解算 -> 更新齿轮、曲线、结果文字。"""
        res = self.model.solve(phi)

        # 1) 齿轮联动角度（度）。主齿轮顺时针；副齿轮逆时针。
        main_deg = (phi * self.model.R * 360.0) % 360.0
        sub_deg = (-main_deg * (self.model.Z1 / self.model.Z2)) % 360.0
        self.gears.set_state(self.model.Z1, self.model.Z2, main_deg, sub_deg)

        # 2) 曲线向左滚动：缓冲整体左移一格，新值写入末尾
        self.buf_main = np.roll(self.buf_main, -1)
        self.buf_sub = np.roll(self.buf_sub, -1)
        self.buf_delta = np.roll(self.buf_delta, -1)
        self.buf_main[-1] = res["normalized_main"]
        self.buf_sub[-1] = res["normalized_sub"]
        self.buf_delta[-1] = res["delta"]
        self.curve_main.setData(self.x_axis, self.buf_main)
        self.curve_sub.setData(self.x_axis, self.buf_sub)
        self.curve_delta.setData(self.x_axis, self.buf_delta)

        # 3) 结果文字
        self.lbl_turns.setText(f"绝对圈数 T = {res['T']}")
        self.lbl_out_angle.setText(f"输出轴角度 = {res['output_angle_deg']:.1f}°")

        # 4) 左侧编码器原始读数（主=raw_main，副=注入误差并滤波后的当前读数）
        self.lbl_raw_main.setText(f"{int(round(res['raw_main'])) % FULL_SCALE}")
        self.lbl_raw_sub.setText(f"{int(round(res['raw_sec_avg'])) % FULL_SCALE}")


# ============================================================================
# 程序入口
# ============================================================================
def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    # 暗色主题，贴近示波器风格
    pal = QtGui.QPalette()
    pal.setColor(QtGui.QPalette.Window, QtGui.QColor(30, 32, 38))
    pal.setColor(QtGui.QPalette.WindowText, QtGui.QColor(225, 225, 230))
    pal.setColor(QtGui.QPalette.Base, QtGui.QColor(24, 26, 31))
    pal.setColor(QtGui.QPalette.Text, QtGui.QColor(225, 225, 230))
    pal.setColor(QtGui.QPalette.Button, QtGui.QColor(45, 48, 58))
    pal.setColor(QtGui.QPalette.ButtonText, QtGui.QColor(230, 230, 235))
    app.setPalette(pal)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
