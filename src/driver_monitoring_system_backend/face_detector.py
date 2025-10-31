import cv2
import mediapipe as mp
import numpy as np


class FaceLandmarkDetector:
    """Детектор всех ключевых точек лица с использованием MediaPipe Face Mesh."""

    def __init__(self, face_mesh: mp.solutions.face_mesh.FaceMesh | None = None) -> None:
        """Инициализация детектора."""
        self.face_mesh = face_mesh or mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

    def process_frame(self, frame: np.ndarray) -> list[dict[int, tuple[int, int]]] | None:
        """Возвращает координаты всех точек лица."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None

        h, w, _ = frame.shape
        all_faces = []
        for face_landmarks in results.multi_face_landmarks:
            points = {}
            for i, lm in enumerate(face_landmarks.landmark):  # проходим по всем точкам
                px, py = int(lm.x * w), int(lm.y * h)
                points[i] = (px, py)
                cv2.circle(frame, (px, py), 1, (0, 255, 0), -1)  # рисуем все точки
            all_faces.append(points)
        return all_faces
