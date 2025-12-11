import os
import cv2
import numpy as np
import shutil
from insightface.app import FaceAnalysis
import time

def recognize_face(face_embedding, reference_faces, threshold=0.2):
    best_name = "Unknown"
    best_similarity = 0.0

    for name, ref_emb in reference_faces.items():
        similarity = np.dot(face_embedding, ref_emb) / (np.linalg.norm(face_embedding) * np.linalg.norm(ref_emb))
        if similarity > best_similarity:
            best_similarity = similarity
            best_name = name

    if best_similarity < threshold:
        return "Unknown", 0.0

    return best_name, best_similarity


reference_dir = "/mnt/d/Python_programms/projectSearchFaces/references"

# Путь к директории с исходными изображениями
#input_dir = "/mnt/c/Users/eleprog/Downloads/Telegram Desktop/ChatExport_2025-06-11 (1)/photos"
input_dir = "/mnt/d/Python_programms/projectSearchFaces/photos"

# Путь к директории для сохранения изображений с лицами
output_dir = "/mnt/c/Python_programms/projectSearchFaces/photosFound"
os.makedirs(output_dir, exist_ok=True)

# Инициализация модели SCRFD
app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 для CUDA

reference_faces = {}

# Проход по всем файлам в директории
for filename in os.listdir(reference_dir):
    file_path = os.path.join(reference_dir, filename)

    if os.path.isfile(file_path):
        name = os.path.splitext(filename)[0]
        reference_faces[name] = app.get(cv2.imread(file_path))[0].normed_embedding



# Обработка изображений
for filename in os.listdir(input_dir):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        path = os.path.join(input_dir, filename)
        try:
            img = cv2.imread(path)
            if img is None:
                print(f"Ошибка чтения файла: {filename}")
                continue
            faces = app.get(img)

            if len(faces) == 0:
                continue

            
            detection_str = ""

            for face in faces:
                name, confidence = recognize_face(face.normed_embedding, reference_faces)

                if name == "Unknown":
                    continue
                
                img_height, img_width = img.shape[:2]

                bbox = face.bbox.astype(int)  # Получаем координаты
                det_width = bbox[2] - bbox[0]
                det_height = bbox[3] - bbox[1]

                if det_width < 40:
                    continue

                detection_str += name + "_"

                # Отрисовка детекций распознанных лиц
                x1, y1, x2, y2 = bbox
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Получаем уверенность модели
                confidence = face.det_score
                label = f"{confidence:.2f}"  # Округление до 2 знаков
                
                # Параметры текста
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_thickness = 1
                text_color = (255, 255, 255)  # Белый цвет текста
                bg_color = (0, 255, 0)        # Цвет фона (совпадает с цветом рамки)

                # Размер текста
                (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
                
                # Рисуем фон для текста
                cv2.rectangle(img, (x1, y1 - 20), (x1 + text_width + 5, y1), bg_color, -1)
                # Рисуем текст
                cv2.putText(img, label, (x1 + 2, y1 - 5), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

            detection_str = detection_str[:-1]

            
            if detection_str == "":

                dir_name = output_dir + "/Unknown"
                continue
            else:
                dir_name = output_dir + "/" + detection_str
                print(f"Найдено фото c: {detection_str.replace("_", ", ")}")

            os.makedirs(dir_name, exist_ok=True)
            shutil.copy(path, os.path.join(dir_name, filename))

            os.makedirs(dir_name + "/debug", exist_ok=True)
            cv2.imwrite(dir_name + "/debug/" + filename, img)


        except Exception as e:
            print(f"Ошибка обработки {filename}: {e}")

            