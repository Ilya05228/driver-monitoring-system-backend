import math
import typing
from abc import ABC, abstractmethod

import numpy as np

from driver_monitoring_system_backend.outputs import (
    BothEyes,
    HeadCenter,
    HeadCenterRelative,
    Output,
    SingleEye,
    XRotation,
)

T = typing.TypeVar("T", bound=Output)


class MissingLandmarksError(Exception):
    """Ошибка, возникающая при отсутствии нужных точек лица."""


class BaseEstimator(ABC, typing.Generic[T]):
    """Базовый класс для всех эстиматоров.

    Определяет единый интерфейс estimate(face_points).
    """

    @abstractmethod
    def estimate(self, face_points: dict[int, tuple[int, int]]) -> T | None:
        """Вычисляет результат анализа."""


class XRotationEstimator(BaseEstimator):
    """Оценивает наклон головы по оси X (вверх/вниз) через средние центры глаз."""

    _LEFT_EYE_INDICES: typing.ClassVar = [33, 160, 158, 133, 153, 144]
    _RIGHT_EYE_INDICES: typing.ClassVar = [362, 385, 387, 263, 373, 380]

    def _center(self, points: list[tuple[int, int]]) -> tuple[float, float]:
        if not points:
            raise MissingLandmarksError
        x = sum(p[0] for p in points) / len(points)
        y = sum(p[1] for p in points) / len(points)
        return x, y

    def estimate(self, face_points: dict[int, tuple[int, int]]) -> XRotation | None:
        left_pts = [face_points[i] for i in self._LEFT_EYE_INDICES if i in face_points]
        right_pts = [face_points[i] for i in self._RIGHT_EYE_INDICES if i in face_points]

        if len(left_pts) < 4 or len(right_pts) < 4:
            raise MissingLandmarksError

        left_center = self._center(left_pts)
        right_center = self._center(right_pts)

        dx = right_center[0] - left_center[0]
        dy = right_center[1] - left_center[1]
        angle = math.degrees(math.atan2(dy, dx))
        return XRotation(angle=angle)


class SingleEyeEstimator(BaseEstimator[SingleEye]):
    """Оценивает открытость одного глаза."""

    def __init__(self, eye_indices: list[int]):
        self.eye_indices = eye_indices

    def estimate(self, face_points: dict[int, tuple[int, int]]) -> SingleEye | None:
        eye_pts = [face_points[i] for i in self.eye_indices if i in face_points]
        if len(eye_pts) < 6:
            raise MissingLandmarksError

        p1, p2, p3, p4, p5, p6 = eye_pts[:6]
        upper = ((p2[0] + p3[0]) / 2, (p2[1] + p3[1]) / 2)
        lower = ((p5[0] + p6[0]) / 2, (p5[1] + p6[1]) / 2)
        vertical = math.hypot(upper[0] - lower[0], upper[1] - lower[1])
        horizontal = math.hypot(p1[0] - p4[0], p1[1] - p4[1])

        ratio = vertical / horizontal if horizontal > 0 else 0.0
        openness = (ratio - 0.1) / (0.35 - 0.1)
        openness = max(0.0, min(1.0, openness))
        return SingleEye(openness=openness)


class BothEyesEstimator(BaseEstimator[BothEyes]):
    """Оценивает открытость обоих глаз, используя два SingleEyeEstimator."""

    _LEFT_EYE: typing.ClassVar[list[int]] = [33, 160, 158, 133, 153, 144]
    _RIGHT_EYE: typing.ClassVar[list[int]] = [362, 385, 387, 263, 373, 380]

    def __init__(self) -> None:
        self.left_eye_est = SingleEyeEstimator(self._LEFT_EYE)
        self.right_eye_est = SingleEyeEstimator(self._RIGHT_EYE)

    def estimate(self, face_points: dict[int, tuple[int, int]]) -> BothEyes | None:
        left = self.left_eye_est.estimate(face_points)
        right = self.right_eye_est.estimate(face_points)
        if not left or not right:
            raise MissingLandmarksError
        return BothEyes(left_eye=left, right_eye=right)


class HeadCenterEstimator(BaseEstimator[HeadCenter]):
    """Определяет координты центра головы по овалу лица."""

    _FACE_OVAL: typing.ClassVar[list[int]] = [
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

    def estimate(self, face_points: dict[int, tuple[int, int]]) -> HeadCenter | None:
        xs = [face_points[i][0] for i in self._FACE_OVAL if i in face_points]
        ys = [face_points[i][1] for i in self._FACE_OVAL if i in face_points]
        if not xs or not ys:
            raise MissingLandmarksError
        return HeadCenter(x=int(np.mean(xs)), y=int(np.mean(ys)))


class HeadCenterRelativeEstimator(BaseEstimator[HeadCenterRelative]):
    """Определяет координаты центра головы по овалу лица и вычисляет проценты относительно кадра."""

    _FACE_OVAL: typing.ClassVar[list[int]] = HeadCenterEstimator._FACE_OVAL  # noqa: SLF001

    def __init__(self, frame_width: int, frame_height: int):
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError("Ширина и высота кадра должны быть положительными числами")
        self.frame_width = frame_width
        self.frame_height = frame_height

    def estimate(self, face_points: dict[int, tuple[int, int]]) -> HeadCenterRelative | None:
        xs = [face_points[i][0] for i in self._FACE_OVAL if i in face_points]
        ys = [face_points[i][1] for i in self._FACE_OVAL if i in face_points]

        if not xs or not ys:
            raise MissingLandmarksError("Нет необходимых точек для оценки центра головы")
        x_px = int(np.mean(xs))
        y_px = int(np.mean(ys))
        x_rel = (x_px / self.frame_width) * 100
        y_rel = (y_px / self.frame_height) * 100
        return HeadCenterRelative(x=x_px, y=y_px, x_rel=x_rel, y_rel=y_rel)
