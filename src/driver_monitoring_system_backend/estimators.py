import math
import typing
from abc import ABC, abstractmethod

import numpy as np

from driver_monitoring_system_backend.outputs import (
    BothEyes,
    HeadCenter,
    HeadCenterRelative,
    HeadRotation,
    Output,
    SingleEye,
    XRotation,
    YRotation,
    ZRotation,
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


class XRotationEstimator(BaseEstimator[XRotation]):
    """Оценивает наклон головы по оси X (вверх/вниз)."""

    _NOSE_TIP = 1
    _CHIN = 152

    def estimate(self, face_points: dict[int, tuple[int, int]]) -> XRotation | None:
        if self._NOSE_TIP not in face_points or self._CHIN not in face_points:
            raise MissingLandmarksError

        nose = face_points[self._NOSE_TIP]
        chin = face_points[self._CHIN]
        dx, dy = chin[0] - nose[0], chin[1] - nose[1]
        angle = math.degrees(math.atan2(dy, dx))
        return XRotation(angle=angle - 90)


class YRotationEstimator(BaseEstimator[YRotation]):
    """Оценивает поворот головы по оси Y (влево/вправо)."""

    _NOSE_TIP = 1
    _CHIN = 152

    def estimate(self, face_points: dict[int, tuple[int, int]]) -> YRotation | None:
        if self._NOSE_TIP not in face_points or self._CHIN not in face_points:
            raise MissingLandmarksError

        nose = face_points[self._NOSE_TIP]
        chin = face_points[self._CHIN]
        dx, dy = chin[0] - nose[0], chin[1] - nose[1]
        angle = math.degrees(math.atan2(-dx, dy))
        return YRotation(angle=angle)


class ZRotationEstimator(BaseEstimator[ZRotation]):
    """Оценивает наклон головы по оси Z (наклон вбок)."""

    _LEFT_EAR = 234
    _RIGHT_EAR = 454

    def estimate(self, face_points: dict[int, tuple[int, int]]) -> ZRotation | None:
        if self._LEFT_EAR not in face_points or self._RIGHT_EAR not in face_points:
            raise MissingLandmarksError

        left_ear = face_points[self._LEFT_EAR]
        right_ear = face_points[self._RIGHT_EAR]
        dx, dy = right_ear[0] - left_ear[0], right_ear[1] - left_ear[1]
        angle = math.degrees(math.atan2(dy, dx))
        return ZRotation(angle=angle)


class HeadRotationEstimator(BaseEstimator[HeadRotation]):
    """Оценивает все три угла поворота головы (X, Y, Z)."""

    def __init__(self) -> None:
        self.x_est = XRotationEstimator()
        self.y_est = YRotationEstimator()
        self.z_est = ZRotationEstimator()

    def estimate(self, face_points: dict[int, tuple[int, int]]) -> HeadRotation | None:
        x = self.x_est.estimate(face_points)
        y = self.y_est.estimate(face_points)
        z = self.z_est.estimate(face_points)

        if x and y and z:
            return HeadRotation(x=x, y=y, z=z)
        raise MissingLandmarksError


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
