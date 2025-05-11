import cv2
import time
import torch
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from playsound import playsound
from threading import Thread

# Load the pretrained EfficientNetB0 model
model = EfficientNet.from_pretrained('efficientnet-b0')
model._fc = torch.nn.Linear(model._fc.in_features, 2)  # 2 classes: Awake, Drowsy
model.load_state_dict(torch.load('drowsiness_model.pth', map_location=torch.device('cpu')))
model.eval()

def play_alert():
    playsound('alert.mp3')

# Preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load Haar cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY))

        if len(eyes) >= 2:
            # Predict drowsiness
            input_img = transform(roi_color).unsqueeze(0)
            with torch.no_grad():
                output = model(input_img)
                _, predicted = torch.max(output, 1)

            if predicted.item() == 1:  # Drowsy
                cv2.putText(frame, 'Drowsy', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                Thread(target=play_alert).start()
            else:
                cv2.putText(frame, 'Awake', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow('Driver Drowsiness Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
