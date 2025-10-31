import cv2
import mediapipe as mp
import numpy as np


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
