import cv2

from driver_monitoring_system_backend.estimators import HeadCenterEstimator, MissingLandmarksError
from driver_monitoring_system_backend.face_detector import FaceLandmarkDetector


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: не удалось открыть камеру")
        return

    face_detector = FaceLandmarkDetector()
    center_estimator = HeadCenterEstimator()

    print("Запуск... Нажмите 'q' для выхода")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = face_detector.process_frame(frame)
        if not faces:
            cv2.imshow("Face Center", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        face_points = faces[0]

        try:
            center = center_estimator.estimate(face_points)
        except MissingLandmarksError:
            print("Недостаточно точек для анализа")
            continue

        if center:
            print(f"Позиция лица: x={center.x}, y={center.y}")
            cv2.circle(frame, (center.x, center.y), 6, (0, 0, 255), -1)

        cv2.imshow("Face Center", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
