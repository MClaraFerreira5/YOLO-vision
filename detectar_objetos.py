from ultralytics import YOLO
import cv2  

model = YOLO("yolov8n.pt")  

cap = cv2.VideoCapture(0) 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Fazer a detecção
    results = model(frame)

    # Exibir os resultados
    for result in results:
        frame = result.plot()

    cv2.imshow("YOLO Detecção", frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
