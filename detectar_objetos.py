from ultralytics import YOLO
import cv2  

model = YOLO("yolov8n.pt")  

cap = cv2.VideoCapture(0) 


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    
    results = model(frame)

    
    for result in results:
        frame = result.plot()

    cv2.imshow("YOLO Detecção", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('e'):
        break

cap.release()
cv2.destroyAllWindows()
