import collections
import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class BufferConfig:
    """Настройки длины буферов для анализа."""

    left_eye_len: int = 30
    right_eye_len: int = 30
    head_angle_len: int = 30
    center_len: int = 30


@dataclass(frozen=True)
class SleepParams:
    """Параметры анализа сна и отвлечения."""

    eyes_closed_threshold: float = 0.35
    head_tilt_angle_threshold: float = 5.0
    x_bounds: tuple[float, float] = (30, 70)
    y_bounds: tuple[float, float] = (30, 70)
    sleep_condition_count: int = 2


class SleepAnalyzer:
    """Анализирует состояние водителя: сон, отвлечение, выход за пределы."""

    def __init__(
        self,
        buffers: BufferConfig | None = None,
        params: SleepParams | None = None,
    ):
        self.buffers = buffers or BufferConfig()
        self.params = params or SleepParams()

        self.left_eye_buf = collections.deque(maxlen=self.buffers.left_eye_len)
        self.right_eye_buf = collections.deque(maxlen=self.buffers.right_eye_len)
        self.head_angle_buf = collections.deque(maxlen=self.buffers.head_angle_len)
        self.center_buf = collections.deque(maxlen=self.buffers.center_len)

        self._is_sleeping = False

    def update(
        self,
        left_eye: float,
        right_eye: float,
        head_angle: float,
        center_x_rel: float,
        center_y_rel: float,
    ) -> None:
        """Обновляет буферы и пересчитывает состояние сна."""
        self.left_eye_buf.append(left_eye)
        self.right_eye_buf.append(right_eye)
        self.head_angle_buf.append(head_angle)
        self.center_buf.append((center_x_rel, center_y_rel))
        self._update_sleep_state()

    def _update_sleep_state(self) -> None:
        """Пересчитывает, спит ли человек."""
        eyes_closed = self._check_eyes_closed()
        head_tilted = self._check_head_tilted()
        head_outside = self._check_head_outside()
        self._is_sleeping = sum([eyes_closed, head_tilted, head_outside]) >= self.params.sleep_condition_count or self._check_eyes_closed()

    def is_sleep(self) -> bool:
        """Возвращает текущее состояние сна."""
        return self._is_sleeping

    def _check_eyes_closed(self) -> bool:
        if len(self.left_eye_buf) < self.left_eye_buf.maxlen:
            return False
        if len(self.right_eye_buf) < self.right_eye_buf.maxlen:
            return False

        median_left = statistics.median(self.left_eye_buf)
        median_right = statistics.median(self.right_eye_buf)

        return median_left <= self.params.eyes_closed_threshold and median_right <= self.params.eyes_closed_threshold

    def _check_head_tilted(self) -> bool:
        if len(self.head_angle_buf) < self.head_angle_buf.maxlen:
            return False

        median_angle = statistics.median(self.head_angle_buf)
        return abs(median_angle) >= self.params.head_tilt_angle_threshold

    def _check_head_outside(self) -> bool:
        if len(self.center_buf) < self.center_buf.maxlen:
            return False

        x_rels, y_rels = zip(*self.center_buf, strict=False)
        x_out = any(x < self.params.x_bounds[0] or x > self.params.x_bounds[1] for x in x_rels)
        y_out = any(y < self.params.y_bounds[0] or y > self.params.y_bounds[1] for y in y_rels)
        return x_out or y_out
