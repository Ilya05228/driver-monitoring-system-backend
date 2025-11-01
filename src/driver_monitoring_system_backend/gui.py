# ruff: noqa
import collections
import sys
import signal
import pygame
import cv2
import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QFrame, QHBoxLayout, QLabel, QMainWindow, QVBoxLayout, QWidget

from driver_monitoring_system_backend.analizers import SleepAnalyzer
from driver_monitoring_system_backend.estimators import BothEyesEstimator, HeadCenterRelativeEstimator, XRotationEstimator
from driver_monitoring_system_backend.face_detector import FaceLandmarkDetector


class VideoWorker(QObject):
    frame_ready = Signal(np.ndarray)
    status_updated = Signal(str, bool)
    params_updated = Signal(float, float, float, float, float)

    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            print("Ошибка: камера не открыта")
            return

        self.face_detector = FaceLandmarkDetector()
        self.eyes_est = BothEyesEstimator()
        self.x_rot_est = XRotationEstimator()
        self.center_est = None
        self.analyzer = SleepAnalyzer()

        self.buf_len = 10
        self.left_buf = collections.deque(maxlen=self.buf_len)
        self.right_buf = collections.deque(maxlen=self.buf_len)
        self.angle_buf = collections.deque(maxlen=self.buf_len)
        self.x_buf = collections.deque(maxlen=self.buf_len)
        self.y_buf = collections.deque(maxlen=self.buf_len)
        self.FACE_OVAL = [
            10,
            338,
            297,
            332,
            284,
            251,
            389,
            356,
            454,
            323,
            361,
            288,
            397,
            365,
            379,
            378,
            400,
            377,
            152,
            148,
            176,
            149,
            150,
            136,
            172,
            58,
            132,
            93,
            234,
            127,
            162,
            21,
            54,
            103,
            67,
            109,
        ]
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

    def start(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)
        self.timer.start(30)

    def process_frame(self):
        ret, frame = self.cap.read()
        frame = cv2.flip(frame, 1)
        if not ret:
            return

        if self.center_est is None:
            h, w = frame.shape[:2]
            self.center_est = HeadCenterRelativeEstimator(w, h)

        disp = frame.copy()
        faces = self.face_detector.process_frame(frame)
        if not faces:
            self.frame_ready.emit(disp)
            return

        pts = faces[0]

        try:
            eyes = self.eyes_est.estimate(pts)
            x_rot = self.x_rot_est.estimate(pts)
            center = self.center_est.estimate(pts)
        except Exception as e:
            print(f"Ошибка оценки: {e}")
            self.frame_ready.emit(disp)
            return

        if not eyes or not x_rot or not center:
            self.frame_ready.emit(disp)
            return

        self.left_buf.append(eyes.left_eye.openness)
        self.right_buf.append(eyes.right_eye.openness)
        self.angle_buf.append(x_rot.angle)
        self.x_buf.append(center.x_rel)
        self.y_buf.append(center.y_rel)

        left_avg = sum(self.left_buf) / len(self.left_buf)
        right_avg = sum(self.right_buf) / len(self.right_buf)
        angle_avg = sum(self.angle_buf) / len(self.angle_buf)
        x_avg = sum(self.x_buf) / len(self.x_buf)
        y_avg = sum(self.y_buf) / len(self.y_buf)

        self.analyzer.update(left_avg, right_avg, angle_avg, x_avg, y_avg)
        is_sleeping = self.analyzer.is_sleep()

        face_pts = [pts[i] for i in self.FACE_OVAL if i in pts]
        if len(face_pts) > 1:
            for i in range(len(face_pts)):
                cv2.line(disp, face_pts[i], face_pts[(i + 1) % len(face_pts)], (0, 255, 0), 1)

        for indices, color in [(self.LEFT_EYE, (255, 0, 0)), (self.RIGHT_EYE, (255, 0, 0))]:
            epts = [pts[i] for i in indices if i in pts]
            if len(epts) == 6:
                for i in range(6):
                    cv2.line(disp, epts[i], epts[(i + 1) % 6], color, 1)

        cv2.circle(disp, (center.x, center.y), 6, (0, 255, 0), -1)

        self.frame_ready.emit(disp)
        self.status_updated.emit("СПИТ" if is_sleeping else "НЕ СПИТ", is_sleeping)
        self.params_updated.emit(left_avg, right_avg, angle_avg, x_avg, y_avg)

    def stop(self):
        self.cap.release()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Мониторинг водителя")
        self.resize(1000, 700)

        self.is_sleeping = False

        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        try:
            self.beep_sound = pygame.mixer.Sound("files/beep_1.mp3")
            self.beep_sound.set_volume(0.7)
        except Exception as e:
            print(f"Ошибка загрузки звука: {e}")
            self.beep_sound = None

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(12)

        self.status_label = QLabel("НЕ СПИТ")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setMinimumHeight(80)
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #2d2d2d;
                color: white;
                font-size: 28px;
                font-weight: bold;
                border-radius: 12px;
                padding: 16px;
            }
        """)
        layout.addWidget(self.status_label)

        self.params_frame = QFrame()
        self.params_frame.setStyleSheet("""
            QFrame {
                background-color: #1e1e1e;
                border-radius: 10px;
                padding: 12px;
            }
            QLabel {
                color: #e0e0e0;
                font-size: 13px;
            }
        """)
        params_layout = QHBoxLayout(self.params_frame)
        params_layout.setSpacing(25)

        self.labels = {}
        for name in ["Левый глаз", "Правый глаз", "Наклон", "X центр", "Y центр"]:
            lbl = QLabel(f"{name}: --")
            self.labels[name] = lbl
            params_layout.addWidget(lbl)
        layout.addWidget(self.params_frame)

        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: #0d0d0d; border-radius: 10px;")
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label, 1)

        self.worker = VideoWorker()
        self.worker.frame_ready.connect(self.update_frame)
        self.worker.status_updated.connect(self.update_status)
        self.worker.params_updated.connect(self.update_params)
        self.worker.start()

    def update_frame(self, frame: np.ndarray):
        h, w, ch = frame.shape
        qt_img = QImage(frame.data, w, h, ch * w, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qt_img)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_status(self, text: str, is_sleeping: bool):
        self.status_label.setText(text)
        color = "#f44336" if is_sleeping else "#4caf50"
        self.status_label.setStyleSheet(f"""
            background-color: {color};
            color: white;
            font-size: 28px;
            font-weight: bold;
            border-radius: 12px;
            padding: 16px;
        """)

        if is_sleeping and not self.is_sleeping:
            if self.beep_sound:
                self.beep_sound.play(loops=-1)
        elif not is_sleeping and self.is_sleeping:
            if self.beep_sound:
                self.beep_sound.stop()

        self.is_sleeping = is_sleeping

    def update_params(self, left, right, angle, x, y):
        self.labels["Левый глаз"].setText(f"Левый глаз: {left:.2f}")
        self.labels["Правый глаз"].setText(f"Правый глаз: {right:.2f}")
        self.labels["Наклон"].setText(f"Наклон: {angle:+.1f}°")
        self.labels["X центр"].setText(f"X: {x:.1f}%")
        self.labels["Y центр"].setText(f"Y: {y:.1f}%")

    def closeEvent(self, event):
        self.worker.stop()
        if self.beep_sound:
            self.beep_sound.stop()
        pygame.mixer.quit()
        event.accept()


def apply_dark_theme(app: QApplication):
    app.setStyleSheet("""
        QMainWindow, QWidget {
            background-color: #121212;
            color: #e0e0e0;
        }
        QScrollBar:vertical {
            background: #1e1e1e;
            width: 10px;
            margin: 2px;
        }
    """)


def run_gui():
    app = QApplication(sys.argv)
    apply_dark_theme(app)

    window = MainWindow()
    window.show()

    def sigint_handler(*args):
        QApplication.quit()

    signal.signal(signal.SIGINT, sigint_handler)
    signal.signal(signal.SIGTERM, sigint_handler)

    sys.exit(app.exec())
