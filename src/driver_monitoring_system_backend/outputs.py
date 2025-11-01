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


class HeadCenterRelative(HeadCenter):
    """Расширяет HeadCenter: добавляет относительные координаты (в % от размеров кадра)."""

    def __init__(self, x: int, y: int, x_rel: float | None = None, y_rel: float | None = None):
        super().__init__(x, y)
        self._x_rel = x_rel
        self._y_rel = y_rel

    @property
    def x_rel(self) -> float | None:
        """Относительная координата X (%)"""
        return self._x_rel

    @property
    def y_rel(self) -> float | None:
        """Относительная координата Y (%)"""
        return self._y_rel

    def __repr__(self) -> str:
        base = super().__repr__()
        if self._x_rel is not None and self._y_rel is not None:
            return f"{base[:-1]}, x_rel={self.x_rel:.2f}%, y_rel={self.y_rel:.2f}%)"
        return base
