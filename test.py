from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

results = model("https://ultralytics.com/images/bus.jpg")
img = results[0].plot()  # dibuja cajas en la imagen (numpy array)

cv2.imshow("YOLOv8 Test", img)
cv2.waitKey(0)           # espera hasta que presiones una tecla
cv2.destroyAllWindows()
