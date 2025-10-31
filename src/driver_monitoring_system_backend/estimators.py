import math
import typing
from abc import abstractmethod

import numpy as np

from driver_monitoring_system_backend.outputs import BothEyes, HeadCenter, HeadRotation, Output, SingleEye, XRotation, YRotation, ZRotation

T = typing.TypeVar("T", bound=Output)


class BaseEsimilator:
    """Базовый класс для анализа лица. Обеспечивает доступ к точкам первого лица."""

    @abstractmethod
    def estimate(self, frame: np.ndarray) -> T | None:
        pass


class XRotationEstimator(BaseEsimilator):
    """Оценивает угол наклона головы по оси X (x)."""

    _NOSE_TIP = 1
    _CHIN = 152

    def __init__(self):
        point_indices = [self._NOSE_TIP, self._CHIN]
        super().__init__(point_indices, refine=True)

    def estimate(self, frame: np.ndarray) -> XRotation | None:
        points = self._get_face_points(frame)
        if not points or self._NOSE_TIP not in points or self._CHIN not in points:
            return None

        nose = points[self._NOSE_TIP]
        chin = points[self._CHIN]
        face_vector = (chin[0] - nose[0], chin[1] - nose[1])
        angle = math.degrees(math.atan2(face_vector[1], face_vector[0]))

        return XRotation(angle=angle)


class YRotationEstimator(BaseEsimilator):
    """Оценивает угол поворота головы по оси Y (y)."""

    _NOSE_TIP = 1
    _CHIN = 152

    def __init__(self):
        point_indices = [self._NOSE_TIP, self._CHIN]
        super().__init__(point_indices, refine=True)

    def estimate(self, frame: np.ndarray) -> YRotation | None:
        points = self._get_face_points(frame)
        if not points or self._NOSE_TIP not in points or self._CHIN not in points:
            return None

        nose = points[self._NOSE_TIP]
        chin = points[self._CHIN]
        face_vector = (chin[0] - nose[0], chin[1] - nose[1])
        angle = math.degrees(math.atan2(-face_vector[0], face_vector[1]))

        return YRotation(angle=angle)


class ZRotationEstimator(BaseEsimilator):
    """Оценивает угол наклона головы по оси Z (z)."""

    _LEFT_EAR = 234
    _RIGHT_EAR = 454

    def __init__(self):
        point_indices = [self._LEFT_EAR, self._RIGHT_EAR]
        super().__init__(point_indices, refine=True)

    def estimate(self, frame: np.ndarray) -> ZRotation | None:
        points = self._get_face_points(frame)
        if not points or self._LEFT_EAR not in points or self._RIGHT_EAR not in points:
            return None

        left_ear = points[self._LEFT_EAR]
        right_ear = points[self._RIGHT_EAR]
        ear_vector = (right_ear[0] - left_ear[0], right_ear[1] - left_ear[1])
        angle = math.degrees(math.atan2(ear_vector[1], ear_vector[0]))

        return ZRotation(angle=angle)


class HeadRotationEstimator(BaseEsimilator):
    """Оценивает все три угла поворота головы, используя отдельные эстиматоры."""

    def __init__(self):
        point_indices = [1, 152, 234, 454]  # Все нужные точки для трех эстиматоров
        super().__init__(point_indices, refine=True)

        # Создаем эстиматоры для каждой оси
        self.x_estimator = XRotationEstimator()
        self.y_estimator = YRotationEstimator()
        self.z_estimator = ZRotationEstimator()

    def estimate(self, frame: np.ndarray) -> HeadRotation | None:
        """Оценивает все три угла поворота головы."""
        x = self.x_estimator.estimate(frame)
        y = self.y_estimator.estimate(frame)
        z = self.z_estimator.estimate(frame)

        if x and y and z:
            return HeadRotation(x=x, y=y, z=z)
        return None


# ==================== ЭСТИМАТОРЫ ДЛЯ ГЛАЗ ====================


class SingleEyeEstimator:
    """Оценивает открытость одного глаза."""

    def __init__(self, eye_indices: list[int]):
        self.eye_indices = eye_indices

    def calculate_openness(self, eye_points: list[tuple[int, int]]) -> SingleEye:
        """Вычисляет открытость одного глаза."""
        if len(eye_points) < 6:
            return SingleEye(openness=0.0)

        p1, p2, p3, p4, p5, p6 = eye_points[:6]
        upper = ((p2[0] + p3[0]) / 2, (p2[1] + p3[1]) / 2)
        lower = ((p5[0] + p6[0]) / 2, (p5[1] + p6[1]) / 2)
        vertical = math.hypot(upper[0] - lower[0], upper[1] - lower[1])
        horizontal = math.hypot(p1[0] - p4[0], p1[1] - p4[1])
        ratio = vertical / horizontal if horizontal > 0 else 0.0
        openness = (ratio - 0.1) / (0.35 - 0.1)
        openness = max(0.0, min(1.0, openness))

        return SingleEye(openness=openness)


class BothEyesEstimator(BaseEsimilator):
    """Оценивает открытость обоих глаз, используя SingleEyeEstimator."""

    _LEFT_EYE: typing.ClassVar = [33, 160, 158, 133, 153, 144]
    _RIGHT_EYE: typing.ClassVar = [362, 385, 387, 263, 373, 380]

    def __init__(self):
        point_indices = list(set(self._LEFT_EYE + self._RIGHT_EYE))
        super().__init__(point_indices, refine=True)

        # Создаем эстиматоры для каждого глаза
        self.left_eye_estimator = SingleEyeEstimator(self._LEFT_EYE)
        self.right_eye_estimator = SingleEyeEstimator(self._RIGHT_EYE)

    def estimate(self, frame: np.ndarray) -> BothEyes | None:
        points = self._get_face_points(frame)
        if not points:
            return None

        left_pts = [points[i] for i in self._LEFT_EYE if i in points]
        right_pts = [points[i] for i in self._RIGHT_EYE if i in points]

        if len(left_pts) < 6 or len(right_pts) < 6:
            return None

        left_eye = self.left_eye_estimator.calculate_openness(left_pts)
        right_eye = self.right_eye_estimator.calculate_openness(right_pts)

        return BothEyes(left_eye=left_eye, right_eye=right_eye)


class HeadCenterEstimator(BaseEsimilator):
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
        super().__init__(self._FACE_OVAL, refine=True)

    def estimate(self, frame: np.ndarray) -> HeadCenter | None:
        points = self._get_face_points(frame)
        if not points:
            return None

        xs = [points[i][0] for i in self._FACE_OVAL if i in points]
        ys = [points[i][1] for i in self._FACE_OVAL if i in points]

        center_x = int(np.mean(xs))
        center_y = int(np.mean(ys))

        return HeadCenter(x=center_x, y=center_y)
