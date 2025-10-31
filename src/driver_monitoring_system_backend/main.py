import cv2
import mediapipe as mp
import numpy as np


class FaceLandmarkDetector:
    def __init__(
        self,
        point_indices: list[int],
        face_mesh: mp.solutions.face_mesh.FaceMesh | None = None,
        static_mode: bool = False,
        max_faces: int = 3,
        refine: bool = True,
        min_det_conf: float = 0.6,
        min_track_conf: float = 0.6,
    ) -> None:
        self.face_mesh = face_mesh or mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_mode,
            max_num_faces=max_faces,
            refine_landmarks=refine,
            min_detection_confidence=min_det_conf,
            min_tracking_confidence=min_track_conf,
        )

        self.point_indices = point_indices

    def process_frame(self, frame: np.ndarray) -> list[dict[int, tuple[int, int]]] | None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        h, w, _ = frame.shape
        all_faces = []
        for face_landmarks in results.multi_face_landmarks:
            points: dict[int, tuple[int, int]] = {}
            for i in self.point_indices:
                lm = face_landmarks.landmark[i]
                px, py = int(lm.x * w), int(lm.y * h)
                points[i] = (px, py)
                cv2.circle(frame, (px, py), 2, (0, 255, 0), -1)
            all_faces.append(points)
        return all_faces


class SleepTrackerGUI:
    pass


class SleepTracker:
    def start(self) -> None:
        """Запуск трекера"""


def main() -> None:
    s_t = SleepTracker()
    s_t.start()


# # --- Логирование ---
# logging.basicConfig(
#     level=logging.INFO,
#     format="[%(asctime)s] %(levelname)s | %(message)s",
#     datefmt="%H:%M:%S",
# )
# logger = logging.getLogger(__name__)


# @dataclass
# class EyeData:
#     """Хранит степень открытости глаз."""

#     left_eye_openness: float = 0.0
#     right_eye_openness: float = 0.0


# class EyeTracker:
#     """
#     Отслеживание положения глаз с помощью MediaPipe.
#     Рисует только контуры лица и глаз, без точек.
#     """

#     def __init__(self) -> None:
#         self.mp_face_mesh = mediapipe.solutions.face_mesh
#         self.face_mesh = self.mp_face_mesh.FaceMesh(
#             max_num_faces=1,
#             refine_landmarks=True,
#             min_detection_confidence=0.7,
#             min_tracking_confidence=0.7,
#         )
#         self.data = EyeData()

#         # Контуры
#         self.FACE_OVAL = self.mp_face_mesh.FACEMESH_FACE_OVAL
#         self.LEFT_EYE = self.mp_face_mesh.FACEMESH_LEFT_EYE
#         self.RIGHT_EYE = self.mp_face_mesh.FACEMESH_RIGHT_EYE

#     def _calculate_eye_openness(self, landmarks: np.ndarray, eye_indices: List[int]) -> float:
#         """Вычисляет относительное раскрытие глаза (вертикальное/горизонтальное)."""
#         try:
#             p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_indices]
#             upper = (p2 + p3) / 2
#             lower = (p5 + p6) / 2
#             vertical = np.linalg.norm(upper - lower)
#             horizontal = np.linalg.norm(p1 - p4)
#             return float(vertical / horizontal) if horizontal > 0 else 0.0
#         except Exception as e:
#             logger.warning(f"Ошибка вычисления открытости глаза: {e}")
#             return 0.0

#     def process_frame(self, frame: np.ndarray) -> np.ndarray:
#         rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = self.face_mesh.process(rgb_frame)

#         if results.multi_face_landmarks:
#             landmarks_proto = results.multi_face_landmarks[0].landmark
#             landmarks = np.array([(lm.x, lm.y, lm.z) for lm in landmarks_proto])

#             # Вычисляем открытость глаз
#             left_open = self._calculate_eye_openness(landmarks, [33, 160, 158, 133, 153, 144])
#             right_open = self._calculate_eye_openness(landmarks, [362, 385, 387, 263, 373, 380])

#             # Обновляем данные
#             self.data.left_eye_openness = left_open
#             self.data.right_eye_openness = right_open

#             # Логирование значений
#             logger.info(f"Глаза | левый: {left_open:.3f} | правый: {right_open:.3f}")

#         return frame


# class MainWindow(qtw.QMainWindow):
#     """Qt GUI: отображает видео и значения открытости глаз."""

#     def __init__(self, tracker: EyeTracker) -> None:
#         super().__init__()
#         self.tracker = tracker
#         self.setWindowTitle("Слежение за глазами")
#         self.setGeometry(100, 100, 800, 600)

#         central = qtw.QWidget()
#         self.setCentralWidget(central)
#         layout = qtw.QVBoxLayout()
#         central.setLayout(layout)

#         # --- Верхняя часть: значения построчно ---
#         values_layout = qtw.QVBoxLayout()

#         def create_value_row(title: str) -> qtw.QLabel:
#             """Создаёт строку с подписью и значением."""
#             label = qtw.QLabel(f"{title} 0.000")
#             label.setAlignment(Qt.AlignLeft)
#             label.setStyleSheet("font-size: 20px; font-weight: bold;")
#             values_layout.addWidget(label)
#             return label

#         self.left_label = create_value_row("Левый глаз:")
#         self.right_label = create_value_row("Правый глаз:")

#         layout.addLayout(values_layout)

#         # --- Видео ---
#         self.video_label = qtw.QLabel()
#         self.video_label.setMinimumSize(640, 480)
#         self.video_label.setStyleSheet("background-color: black;")
#         self.video_label.setAlignment(Qt.AlignCenter)
#         layout.addWidget(self.video_label, stretch=1)

#         # Таймер для обновления GUI
#         self.timer = qtc.QTimer()
#         self.timer.timeout.connect(self.update_gui)
#         self.timer.start(30)

#     @qtc.Slot()
#     def update_gui(self) -> None:
#         """Обновляет данные в GUI."""
#         self.left_label.setText(f"Левый глаз: {self.tracker.data.left_eye_openness:.3f}")
#         self.right_label.setText(f"Правый глаз: {self.tracker.data.right_eye_openness:.3f}")

#         if hasattr(self, "current_frame"):
#             frame_rgb = self.current_frame
#             h, w, _ = frame_rgb.shape
#             qimg = qtg.QImage(frame_rgb.data, w, h, w * 3, qtg.QImage.Format_RGB888)
#             pix = qtg.QPixmap.fromImage(qimg).scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
#             self.video_label.setPixmap(pix)


# def main() -> None:
#     """Основной цикл: захват видео, обработка и отображение."""
#     app = qtw.QApplication(sys.argv)
#     tracker = EyeTracker()
#     window = MainWindow(tracker)
#     window.show()

#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         logger.error("Ошибка: камера не открыта")
#         return

#     logger.info("Слежение за глазами запущено. Нажмите 'q' для выхода.")

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             logger.error("Ошибка чтения кадра с камеры.")
#             break

#         frame = tracker.process_frame(frame)
#         frame = cv2.flip(frame, 1)
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         window.current_frame = frame_rgb

#         app.processEvents()

#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             logger.info("Выход из программы.")
#             break

#     cap.release()
#     app.quit()
#     sys.exit()


# if __name__ == "__main__":
#     main()
