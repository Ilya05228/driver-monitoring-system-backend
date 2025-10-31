import cv2

from driver_monitoring_system_backend.estimators import (
    BothEyesEstimator,
    HeadCenterEstimator,
    HeadRotationEstimator,
    SingleEyeEstimator,
    XRotationEstimator,
    YRotationEstimator,
    ZRotationEstimator,
)


def main() -> None:
    """Запускает веб-камеру и выводит в консоль оценку головы и глаз в реальном времени."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: не удалось открыть камеру")
        return

    # Эстиматоры для наклона (4 штуки)
    x_estimator = XRotationEstimator()
    y_estimator = YRotationEstimator()
    z_estimator = ZRotationEstimator()
    head_rotation_estimator = HeadRotationEstimator()

    # Эстиматоры для глаз (3 штуки)
    left_eye_estimator = SingleEyeEstimator([33, 160, 158, 133, 153, 144])
    right_eye_estimator = SingleEyeEstimator([362, 385, 387, 263, 373, 380])
    both_eyes_estimator = BothEyesEstimator()

    center_estimator = HeadCenterEstimator()

    print("Запуск... Нажмите 'q' для выхода")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не удалось получить кадр")
            break

        # Оценка наклонов (4 эстиматора)
        x = x_estimator.estimate(frame)
        y = y_estimator.estimate(frame)
        z = z_estimator.estimate(frame)
        head_rotation = head_rotation_estimator.estimate(frame)

        # Оценка глаз - используем только BothEyesEstimator, так как он уже использует SingleEyeEstimator внутри
        both_eyes = both_eyes_estimator.estimate(frame)

        center = center_estimator.estimate(frame)

        # Вывод результатов
        if head_rotation:
            print(f"Все углы: x={head_rotation.x.angle}, y={head_rotation.y.angle}, z={head_rotation.z.angle}")
        else:
            print("Поворот головы не определен")

        if x:
            print(f"X: {x.angle}")
        if y:
            print(f"Y: {y.angle}")
        if z:
            print(f"Z: {z.angle}")

        if both_eyes:
            print(f"Оба глаза: левый={both_eyes.left_eye.openness}, правый={both_eyes.right_eye.openness}")
        else:
            print("Глаза не определены")

        if center:
            print(f"Центр головы: {center}")
            cv2.circle(frame, (center.x, center.y), 8, (0, 0, 255), -1)

        cv2.imshow("Face Analysis", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
