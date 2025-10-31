class Output:
    pass


class XRotation(Output):
    """Хранит угол наклона головы по оси X (x)."""

    def __init__(self, angle: float):
        self._angle = round(angle, 2)

    @property
    def angle(self) -> float:
        return self._angle

    def __repr__(self) -> str:
        return f"XRotation(angle={self.angle})"


class YRotation(Output):
    """Хранит угол поворота головы по оси Y (y)."""

    def __init__(self, angle: float):
        self._angle = round(angle, 2)

    @property
    def angle(self) -> float:
        return self._angle

    def __repr__(self) -> str:
        return f"YRotation(angle={self.angle})"


class ZRotation(Output):
    """Хранит угол наклона головы по оси Z (z)."""

    def __init__(self, angle: float):
        self._angle = round(angle, 2)

    @property
    def angle(self) -> float:
        return self._angle

    def __repr__(self) -> str:
        return f"ZRotation(angle={self.angle})"


class HeadRotation(Output):
    """Хранит все три угла поворота головы."""

    def __init__(self, x: XRotation, y: YRotation, z: ZRotation):
        self._x = x
        self._y = y
        self._z = z

    @property
    def x(self) -> XRotation:
        return self._x

    @property
    def y(self) -> YRotation:
        return self._y

    @property
    def z(self) -> ZRotation:
        return self._z

    def __repr__(self) -> str:
        return f"HeadRotation(x={self.x.angle}, y={self.y.angle}, z={self.z.angle})"


# ==================== КЛАССЫ ДЛЯ ГЛАЗ ====================


class SingleEye(Output):
    """Хранит степень открытости одного глаза."""

    def __init__(self, openness: float):
        self._openness = round(openness, 2)

    @property
    def openness(self) -> float:
        return self._openness

    def __repr__(self) -> str:
        return f"SingleEye(openness={self.openness})"


class BothEyes(Output):
    """Хранит степень открытости обоих глаз."""

    def __init__(self, left_eye: SingleEye, right_eye: SingleEye):
        self._left_eye = left_eye
        self._right_eye = right_eye

    @property
    def left_eye(self) -> SingleEye:
        return self._left_eye

    @property
    def right_eye(self) -> SingleEye:
        return self._right_eye

    def __repr__(self) -> str:
        return f"BothEyes(left={self.left_eye.openness}, right={self.right_eye.openness})"


class HeadCenter(Output):
    """Хранит координаты центра головы в кадре."""

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
        return f"HeadCenter(x={self.x}, y={self.y})"
