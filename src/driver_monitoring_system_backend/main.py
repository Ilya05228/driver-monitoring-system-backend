import collections

import cv2

from driver_monitoring_system_backend.estimators import (
    BothEyesEstimator,
    HeadCenterEstimator,
    HeadCenterRelativeEstimator,
    XRotationEstimator,
)
from driver_monitoring_system_backend.face_detector import FaceLandmarkDetector


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: не удалось открыть камеру")
        return

    ret, frame = cap.read()
    if not ret:
        print("Не удалось получить кадр")
        return
    h, w = frame.shape[:2]

    face_detector = FaceLandmarkDetector()
    eyes_est = BothEyesEstimator()
    x_rot_est = XRotationEstimator()
    center_est = HeadCenterRelativeEstimator(w, h)

    FACE_OVAL = HeadCenterEstimator._FACE_OVAL  # noqa: N806, SLF001
    LEFT_EYE = BothEyesEstimator._LEFT_EYE  # noqa: N806, SLF001
    RIGHT_EYE = BothEyesEstimator._RIGHT_EYE  # noqa: N806, SLF001

    buf_len = 15
    left_eye_buf = collections.deque(maxlen=buf_len)
    right_eye_buf = collections.deque(maxlen=buf_len)
    head_buf = collections.deque(maxlen=buf_len)
    head_x_buf = collections.deque(maxlen=buf_len)
    head_y_buf = collections.deque(maxlen=buf_len)

    print("Запуск... 'q' — выход")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        disp = frame.copy()

        faces = face_detector.process_frame(frame)
        if not faces:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        pts = faces[0]

        eyes = eyes_est.estimate(pts)
        left_eye_val = eyes.left_eye.openness
        right_eye_val = eyes.right_eye.openness

        x_rot = x_rot_est.estimate(pts)
        head_angle = x_rot.angle

        center = center_est.estimate(pts)
        head_x_rel = center.x_rel
        head_y_rel = center.y_rel

        left_eye_buf.append(left_eye_val)
        right_eye_buf.append(right_eye_val)
        head_buf.append(head_angle)
        head_x_buf.append(head_x_rel)
        head_y_buf.append(head_y_rel)

        left_eye_avg = round(sum(left_eye_buf) / len(left_eye_buf), 2)
        right_eye_avg = round(sum(right_eye_buf) / len(right_eye_buf), 2)
        head_avg = round(sum(head_buf) / len(head_buf), 1)
        head_x_avg = round(sum(head_x_buf) / len(head_x_buf), 1)
        head_y_avg = round(sum(head_y_buf) / len(head_y_buf), 1)

        print(
            f"\rЛевый глаз: {left_eye_avg:.2f} | "
            f"Правый глаз: {right_eye_avg:.2f} | "
            f"Наклон головы: {head_avg:.1f} | "
            f"Положение головы: ({head_x_avg:.1f}, {head_y_avg:.1f}) | "
            f"Спит: {left_eye_avg <= 0.15 and right_eye_avg <= 0.15}",
            end="",
            flush=True,
        )

        face_pts = [pts[i] for i in FACE_OVAL if i in pts]
        if len(face_pts) > 1:
            for i in range(len(face_pts)):
                cv2.line(disp, face_pts[i], face_pts[(i + 1) % len(face_pts)], (0, 255, 0), 1)

        for indices, color in [(LEFT_EYE, (255, 0, 0)), (RIGHT_EYE, (255, 0, 0))]:
            epts = [pts[i] for i in indices if i in pts]
            if len(epts) == 6:
                for i in range(6):
                    cv2.line(disp, epts[i], epts[(i + 1) % 6], color, 1)

        if center:
            cv2.circle(disp, (center.x, center.y), 6, (0, 255, 0), -1)

        cv2.imshow("Driver Monitoring", disp)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
