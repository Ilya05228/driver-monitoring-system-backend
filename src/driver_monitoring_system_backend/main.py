import math
import typing

import cv2
import mediapipe as mp
import numpy as np


class HeadRotation:
    """Хранит углы поворота головы относительно осей кадра."""

    _rotation_x: float
    _rotation_y: float
    _rotation_z: float

    def __init__(self, rotation_x: float, rotation_y: float, rotation_z: float):
        self._rotation_x = round(rotation_x, 2)
        self._rotation_y = round(rotation_y, 2)
        self._rotation_z = round(rotation_z, 2)

    @property
    def rotation_x(self) -> float:
        return self._rotation_x

    @property
    def rotation_y(self) -> float:
        return self._rotation_y

    @property
    def rotation_z(self) -> float:
        return self._rotation_z

    def __repr__(self) -> str:
        return f"HeadRotation(rotation_x={self._rotation_x}, rotation_y={self._rotation_y}, rotation_z={self._rotation_z})"


class HeadCenter:
    """Хранит координаты центра головы в кадре."""

    _x: int
    _y: int

    def __init__(self, x: int, y: int):
        self._x = x
        self._y = y

    @property
    def x(self) -> int:
        return self._x

    @property
    def y(self) -> int:
        return self._y

    def __repr__(self) -> str:
        return f"HeadCenter(x={self._x}, y={self._y})"


class EyesData:
    """Хранит степень открытости глаз."""

    _left_eye_openness: float
    _right_eye_openness: float

    def __init__(self, left_eye_openness: float, right_eye_openness: float):
        self._left_eye_openness = round(left_eye_openness, 2)
        self._right_eye_openness = round(right_eye_openness, 2)

    @property
    def left_eye_openness(self) -> float:
        return self._left_eye_openness

    @property
    def right_eye_openness(self) -> float:
        return self._right_eye_openness

    def __repr__(self) -> str:
        return f"EyeData(left_eye_openness={self._left_eye_openness}, right_eye_openness={self._right_eye_openness})"


class FaceLandmarkDetector:
    """Детектор ключевых точек лица с использованием MediaPipe Face Mesh."""

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
        """Инициализирует детектор с заданными параметрами и списком индексов точек."""
        self.face_mesh = face_mesh or mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_mode,
            max_num_faces=max_faces,
            refine_landmarks=refine,
            min_detection_confidence=min_det_conf,
            min_tracking_confidence=min_track_conf,
        )
        self.point_indices = point_indices

    def process_frame(self, frame: np.ndarray) -> list[dict[int, tuple[int, int]]] | None:
        """Обрабатывает кадр и возвращает список словарей с координатами точек для каждого лица. Рисует точки на кадре."""
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


class FaceAnalyzer:
    """Базовый класс для анализа лица. Обеспечивает доступ к точкам первого лица."""

    def __init__(
        self,
        point_indices: list[int],
        static_mode: bool = False,
        max_faces: int = 3,
        refine: bool = True,
        min_det_conf: float = 0.6,
        min_track_conf: float = 0.6,
    ):
        """Инициализирует детектор с нужными точками."""
        self.detector = FaceLandmarkDetector(
            point_indices=point_indices,
            static_mode=static_mode,
            max_faces=max_faces,
            refine=refine,
            min_det_conf=min_det_conf,
            min_track_conf=min_track_conf,
        )

    def _get_face_points(self, frame: np.ndarray) -> dict[int, tuple[int, int]] | None:
        """Возвращает точки первого обнаруженного лица или None."""
        results = self.detector.process_frame(frame)
        if results and len(results) > 0:
            return results[0]
        return None

    @staticmethod
    def _euclidean_dist(p1: tuple[int, int], p2: tuple[int, int]) -> float:
        """Вычисляет евклидово расстояние между двумя точками."""
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


class HeadRotationEstimator(FaceAnalyzer):
    """Оценивает углы поворота головы (pitch, yaw, roll)."""

    _NOSE_TIP = 1
    _CHIN = 152
    _LEFT_EAR = 234
    _RIGHT_EAR = 454
    _LEFT_EYE_OUTER = 33
    _RIGHT_EYE_OUTER = 263

    def __init__(self):
        """Инициализирует с набором точек, необходимых для оценки поворота."""
        point_indices = [self._NOSE_TIP, self._CHIN, self._LEFT_EAR, self._RIGHT_EAR, self._LEFT_EYE_OUTER, self._RIGHT_EYE_OUTER]
        super().__init__(point_indices, refine=True)

    def estimate(self, frame: np.ndarray) -> HeadRotation | None:
        """Оценивает и возвращает объект HeadRotation или None при отсутствии лица."""
        points = self._get_face_points(frame)
        if not points:
            return None

        nose = points[self._NOSE_TIP]
        chin = points[self._CHIN]
        left_ear = points[self._LEFT_EAR]
        right_ear = points[self._RIGHT_EAR]

        face_vector = (chin[0] - nose[0], chin[1] - nose[1])
        ear_vector = (right_ear[0] - left_ear[0], right_ear[1] - left_ear[1])

        rotation_x = math.degrees(math.atan2(face_vector[1], face_vector[0]))
        rotation_y = math.degrees(math.atan2(-face_vector[0], face_vector[1]))
        rotation_z = math.degrees(math.atan2(ear_vector[1], ear_vector[0]))

        return HeadRotation(rotation_x=rotation_x, rotation_y=rotation_y, rotation_z=rotation_z)


class HeadCenterEstimator(FaceAnalyzer):
    """Определяет центр головы по овалу лица."""

    _FACE_OVAL: typing.ClassVar = [
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

    def __init__(self):
        """Инициализирует с точками овала лица."""
        super().__init__(self._FACE_OVAL, refine=True)

    def estimate(self, frame: np.ndarray) -> HeadCenter | None:
        """Возвращает центр головы как среднее по точкам овала или None."""
        points = self._get_face_points(frame)
        if not points:
            return None

        xs = [points[i][0] for i in self._FACE_OVAL if i in points]
        ys = [points[i][1] for i in self._FACE_OVAL if i in points]

        center_x = int(np.mean(xs))
        center_y = int(np.mean(ys))

        return HeadCenter(x=center_x, y=center_y)


class EyeOpennessEstimator(FaceAnalyzer):
    """Оценивает открытость глаз как отношение высоты к ширине (0.0 — закрыт, 1.0 — открыт)."""

    _LEFT_EYE: typing.ClassVar = [33, 160, 158, 133, 153, 144]
    _RIGHT_EYE: typing.ClassVar = [362, 385, 387, 263, 373, 380]

    def __init__(self):
        point_indices = list(set(self._LEFT_EYE + self._RIGHT_EYE))
        super().__init__(point_indices, refine=True)

    def _calculate_eye_openness(self, eye_points: list[tuple[int, int]]) -> float:
        """Вычисляет отношение высоты глаза к ширине (как в твоём рабочем коде)."""
        p1, p2, p3, p4, p5, p6 = eye_points[:6]  # Берём первые 6 точек
        upper = ((p2[0] + p3[0]) / 2, (p2[1] + p3[1]) / 2)
        lower = ((p5[0] + p6[0]) / 2, (p5[1] + p6[1]) / 2)
        vertical = math.hypot(upper[0] - lower[0], upper[1] - lower[1])
        horizontal = math.hypot(p1[0] - p4[0], p1[1] - p4[1])
        ratio = vertical / horizontal if horizontal > 0 else 0.0
        openness = (ratio - 0.1) / (0.35 - 0.1)
        return max(0.0, min(1.0, openness))

    def estimate(self, frame: np.ndarray) -> EyeData | None:
        points = self._get_face_points(frame)
        if not points:
            return None

        left_pts = [points[i] for i in self._LEFT_EYE if i in points]
        right_pts = [points[i] for i in self._RIGHT_EYE if i in points]

        if len(left_pts) < 6 or len(right_pts) < 6:
            return None

        left_open = self._calculate_eye_openness(left_pts)
        right_open = self._calculate_eye_openness(right_pts)

        return EyeData(left_eye_openness=left_open, right_eye_openness=right_open)


def main() -> None:
    """Запускает веб-камеру и выводит в консоль оценку головы и глаз в реальном времени."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: не удалось открыть камеру")
        return

    rotation_estimator = HeadRotationEstimator()
    center_estimator = HeadCenterEstimator()
    eye_estimator = EyeOpennessEstimator()

    print("Запуск... Нажмите 'q' для выхода")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не удалось получить кадр")
            break

        rotation = rotation_estimator.estimate(frame)
        center = center_estimator.estimate(frame)
        eyes = eye_estimator.estimate(frame)

        if rotation:
            print(
                f"rotation_x={round(rotation.rotation_x, -0)}, "
                f"rotation_y={round(rotation.rotation_y, -0)}, "
                f"rotation_z={round(rotation.rotation_z, -0)}"
            )

        else:
            print("Поворот головы не определен")

        if center:
            print(f"Центр головы: x={center.x}, y={center.y}")
            cv2.circle(frame, (center.x, center.y), 8, (0, 0, 255), -1)
        else:
            print("Центр головы не определен")

        if eyes:
            print(f"Открытость глаз: левый={eyes.left_eye_openness:.2f}, правый={eyes.right_eye_openness:.2f}")
        else:
            print("Открытость глаз не определена")

        cv2.imshow("Face Analysis", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# class SleepTrackerGUI:
#     pass


# class SleepTracker:
#     def start(self) -> None:
#         """Запуск трекера"""


# def main() -> None:
#     s_t = SleepTracker()
#     s_t.start()

# ======================
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
