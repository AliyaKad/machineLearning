import cv2
import mediapipe as mp
from deepface import DeepFace

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def count_fingers(hand_landmarks):
    finger_tips = [8, 12, 16, 20]
    thumb_tip = 4
    count = 0

    if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[thumb_tip - 1].x:
        count += 1

    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            count += 1

    return count

def recognize_face(face_image):
    try:
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        result = DeepFace.verify(face_image, "image.jpg", model_name="Facenet")
        return result["verified"]
    except Exception as e:
        print(f"Ошибка при распознавании лица: {e}")
        return False

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hand_results = hands.process(rgb_frame)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        face_image = frame[y:y + h, x:x + w]
        is_known = recognize_face(face_image)

        if is_known:
            label = "Aliya"
        else:
            label = "Unknown"

        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    text = ""
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            finger_count = count_fingers(hand_landmarks)

            if finger_count == 1:
                text = "Aliya"
            elif finger_count == 2:
                text = "Kadirova"
            elif finger_count == 3:
                try:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    emotion_result = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)
                    dominant_emotion = emotion_result[0]['dominant_emotion']
                    text = f"Emotion: {dominant_emotion}"
                except Exception as e:
                    print(f"Ошибка при анализе эмоций: {e}")
                    text = "Emotion not detected"
            else:
                text = ""

    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()