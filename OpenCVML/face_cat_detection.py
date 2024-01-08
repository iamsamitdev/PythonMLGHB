import cv2

cat_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalcatface.xml"
)

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

if video_capture.isOpened():
    ret, frame = video_capture.read()
else:
    ret = False

while ret:
    cv2.imshow("Video", frame)
    ret, frame = video_capture.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
    
    # แปลงภาพเป็นสีเทา
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ตรวจจับใบหน้าแมวในภาพ
    faces = cat_cascade.detectMultiScale(gray, 1.1, 3)
    print("Found {0} cat faces!".format(len(faces)))

    # วาดสี่เหลี่ยมรอบใบหน้าแมว
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 255), 4) # สีเหลี่ยมสีเหลือง ความหนา 4
        cv2.putText(frame, 'Cat face', (x, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1) # ข้อความ สีเขียว ความหนา 1

video_capture.destroyWindow("Video") # ปิดหน้าต่าง
video_capture.release() # ปิดการใช้งานกล้อง