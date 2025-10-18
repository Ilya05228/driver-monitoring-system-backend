import cv2
import mediapipe as mp
import numpy as np
import math
import logging
import pygame
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DriverMonitor:
    def __init__(
        self,
        eye_closed_thresh=0.15, 
        mar_thresh=0.6,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        max_cameras_to_check=5,
        draw_landmarks=True,
        sound_on_closed=True
    ):
        self.eye_closed_thresh = eye_closed_thresh  
        self.mar_thresh = mar_thresh
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.max_cameras_to_check = max_cameras_to_check
        self.draw_landmarks = draw_landmarks
        self.sound_on_closed = sound_on_closed
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        
        self.cap = None
        self.width = None
        self.height = None
        self.fps = None

        # Инициализация звука
        if self.sound_on_closed:
            pygame.mixer.init()
            frequency = 1000
            duration = 500
            sample_rate = 44100
            n_samples = int(sample_rate * duration / 1000)
            max_amplitude = 32767
            t = np.linspace(0, duration / 1000, n_samples, False)
            sine_wave = max_amplitude * np.sin(2 * np.pi * frequency * t)
            stereo_wave = np.column_stack((sine_wave, sine_wave)).astype(np.int16)
            self.beep_sound = pygame.mixer.Sound(stereo_wave.tobytes())

    def find_camera(self):
        for i in range(self.max_cameras_to_check):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                logger.info(f"Камера {i} доступна")
                return cap
            cap.release()
        logger.error("Не найдено доступных камер!")
        return None

    def eye_aspect_ratio(self, eye_points):
        try:
            p2_p6 = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
            p3_p5 = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
            p1_p4 = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
            return (p2_p6 + p3_p5) / (2.0 * p1_p4)
        except Exception as e:
            logger.error(f"Ошибка в eye_aspect_ratio: {e}")
            return 0.0

    def get_head_pose(self, landmarks):
        try:
            left_ear = np.array([landmarks[234].x, landmarks[234].y])
            right_ear = np.array([landmarks[454].x, landmarks[454].y])
            ear_vector = right_ear - left_ear
            roll = math.degrees(math.atan2(ear_vector[1], ear_vector[0]))
            
            nose_tip = landmarks[1]
            left_eye_inner = landmarks[33]
            right_eye_inner = landmarks[263]
            center_x = (left_eye_inner.x + right_eye_inner.x) / 2
            yaw = (nose_tip.x - center_x) * 100
            
            chin = landmarks[152]
            forehead = landmarks[10]
            vertical_vector = np.array([chin.y - forehead.y, chin.z - forehead.z])
            pitch = math.degrees(math.atan2(vertical_vector[1], vertical_vector[0])) if vertical_vector[0] != 0 else 0
            
            return roll, yaw, pitch
        except Exception as e:
            logger.error(f"Ошибка в get_head_pose: {e}")
            return 0.0, 0.0, 0.0

    def draw_face_contour(self, frame, landmarks):
        face_contour_indices = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ]
        points = [(int(landmarks[i].x * self.width), int(landmarks[i].y * self.height)) for i in face_contour_indices]
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], (255, 255, 255), 1)
        cv2.line(frame, points[-1], points[0], (255, 255, 255), 1)

    def draw_eye_contours(self, frame, eye_points, is_closed):
        color = (0, 0, 255) if is_closed else (0, 255, 0)
        points = [(int(p[0]), int(p[1])) for p in eye_points]
        cv2.polylines(frame, [np.array(points, dtype=np.int32)], True, color, 1)

    def start(self):
        if self.cap is None:
            self.cap = self.find_camera()
            if self.cap is None:
                return
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            logger.info(f"Подключено к камере: {self.width}x{self.height}, FPS: {self.fps}")
        cv2.namedWindow("Face Detection", cv2.WINDOW_NORMAL)

        last_beep_time = 0
        beep_cooldown = 1

        while True:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Не удалось получить кадр с камеры")
                continue

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = face_landmarks.landmark

                    left_eye_points = [[landmarks[i].x * self.width, landmarks[i].y * self.height] for i in [33, 160, 158, 133, 153, 144]]
                    right_eye_points = [[landmarks[i].x * self.width, landmarks[i].y * self.height] for i in [362, 385, 387, 263, 373, 380]]

                    left_ear = self.eye_aspect_ratio(left_eye_points)
                    right_ear = self.eye_aspect_ratio(right_eye_points)
                    roll, yaw, pitch = self.get_head_pose(landmarks)

                    left_is_closed = left_ear < self.eye_closed_thresh
                    right_is_closed = right_ear < self.eye_closed_thresh

                    left_status = "закрыт" if left_is_closed else "открыт"
                    right_status = "закрыт" if right_is_closed else "открыт"
                    roll_dir = "влево" if roll > 5 else "вправо" if roll < -5 else "прямо"
                    yaw_dir = "влево" if yaw > 5 else "вправо" if yaw < -5 else "прямо"
                    pitch_dir = "вверх" if pitch > 5 else "вниз" if pitch < -5 else "прямо"

                    logger.info(
                        f"Левый глаз: {left_status} (EAR: {left_ear:.2f}) | "
                        f"Правый глаз: {right_status} (EAR: {right_ear:.2f}) | "
                        f"Наклон головы (roll): {abs(roll):.1f}° {roll_dir} | "
                        f"Поворот головы (yaw): {abs(yaw):.1f}° {yaw_dir} | "
                        f"Наклон вверх/вниз (pitch): {abs(pitch):.1f}° {pitch_dir} | "
                    )

                    logger.info("-"*20)
                    if self.draw_landmarks:
                        self.draw_face_contour(frame, landmarks)
                        self.draw_eye_contours(frame, left_eye_points, left_is_closed)
                        self.draw_eye_contours(frame, right_eye_points, right_is_closed)

                    if self.sound_on_closed and (left_is_closed or right_is_closed) and time.time() - last_beep_time > beep_cooldown:
                        self.beep_sound.play()
                        last_beep_time = time.time()

            else:
                cv2.putText(frame, "Лицо не обнаружено", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Face Detection", frame)
            if cv2.waitKey(int(1000 / self.fps)) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

def main():
    monitor = DriverMonitor(sound_on_closed=True)
    monitor.start()