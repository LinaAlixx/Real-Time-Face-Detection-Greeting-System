import cv2 as cv
import pyttsx3

# تحميل نموذج كشف الوجه
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# تهيئة محرك الصوت
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # ضبط سرعة الصوت
engine.setProperty('volume', 1.0)  # ضبط مستوى الصوت

def speak(text):
    engine.say(text)
    engine.runAndWait()

# فتح الكاميرا
cap = cv.VideoCapture(0)
face_detected = False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    image_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)

    if len(faces) > 0 and not face_detected:
        speak("Hi Welcome to ai club")
        face_detected = True
    elif len(faces) == 0:
        face_detected = False
    
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv.imshow("Face Detected", frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

