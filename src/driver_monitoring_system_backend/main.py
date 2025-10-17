import cv2
import mediapipe as mp
import numpy as np
import math

# Инициализация MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Функции EAR, MAR, наклон головы
def eye_aspect_ratio(eye_points):
    p2_p6 = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
    p3_p5 = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
    p1_p4 = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
    return (p2_p6 + p3_p5) / (2.0 * p1_p4)

def head_tilt(landmarks):
    left_ear = np.array([landmarks[234].x, landmarks[234].y])
    right_ear = np.array([landmarks[454].x, landmarks[454].y])
    ear_vector = right_ear - left_ear
    return math.degrees(math.atan2(ear_vector[1], ear_vector[0]))

def mouth_aspect_ratio(mouth_points):
    p2_p8 = np.linalg.norm(np.array(mouth_points[1]) - np.array(mouth_points[7]))
    p3_p7 = np.linalg.norm(np.array(mouth_points[2]) - np.array(mouth_points[6]))
    p4_p6 = np.linalg.norm(np.array(mouth_points[3]) - np.array(mouth_points[5]))
    p1_p5 = np.linalg.norm(np.array(mouth_points[0]) - np.array(mouth_points[4]))
    return (p2_p8 + p3_p7 + p4_p6) / (3.0 * p1_p5)

# Поиск доступной камеры
def find_camera(max_cameras=5):
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"[OK] Камера {i} доступна")
            return cap
        cap.release()
    print("Не найдено доступных камер!")
    return None

def main():
    cap = find_camera()
    if cap is None:
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    print(f"Подключено к камере: {width}x{height}, FPS: {fps}")

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,255), thickness=1))

                    landmarks = face_landmarks.landmark

                    left_eye_points = [[landmarks[i].x*width, landmarks[i].y*height] for i in [33,160,158,133,153,144]]
                    right_eye_points = [[landmarks[i].x*width, landmarks[i].y*height] for i in [362,385,387,263,373,380]]
                    mouth_points = [[landmarks[i].x*width, landmarks[i].y*height] for i in [61,291,0,17,78,308,84,181]]

                    avg_ear = (eye_aspect_ratio(left_eye_points) + eye_aspect_ratio(right_eye_points)) / 2.0
                    tilt_angle = head_tilt(landmarks)
                    mar = mouth_aspect_ratio(mouth_points)

                    eyes_status = "закрыты" if avg_ear < 0.2 else "открыты"
                    yawn_status = "да" if mar > 0.6 else "нет"
                    tilt_dir = "влево" if tilt_angle > 5 else "вправо" if tilt_angle < -5 else "прямо"

                    print(f"Глаза: {eyes_status} (EAR: {avg_ear:.2f}) | "
                          f"Наклон головы: {abs(tilt_angle):.1f} градусов {tilt_dir} | "
                          f"Зевота: {yawn_status} (MAR: {mar:.2f})")

            cv2.imshow("Face Detection", frame)
            if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
