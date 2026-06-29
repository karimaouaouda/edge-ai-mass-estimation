import cv2 as cv




cam = cv.VideoCapture(1)


while True:
    ret, frame = cam.read()
    
    if not ret:
        continue
    
    cv.imshow("Camera Feed", frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break