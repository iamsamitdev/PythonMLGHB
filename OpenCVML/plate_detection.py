import cv2
import pytesseract

# Path to tesseract executable
# Windows
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# MacOS
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

# Load the cascade
plate_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
)

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if video_capture.isOpened():
    ret, frame = video_capture.read()
else:
    raise Exception("Could not open video device")

while ret:
    cv2.imshow("Video", frame)
    ret, frame = video_capture.read()
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
    
    # แปลงภาพเป็นสีเทา
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ตรวจจับทะเบียนรถในภาพ
    plates = plate_cascade.detectMultiScale(gray, 1.1, 3)
    print("Found {0} plates!".format(len(plates)))

    # วาดสี่เหลี่ยมรอบทะเบียนรถ
    for (x, y, w, h) in plates:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 255), 4)

        # อ่านข้อความในทะเบียนรถ
        roi = frame[y:y+h, x:x+w]
        text = pytesseract.image_to_string(roi, lang="tha+eng")
        cv2.putText(frame, text.strip(), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        print("Found plate: ", text.strip())

video_capture.destroyWindow("Video") # ปิดหน้าต่าง
video_capture.release() # ปิดการใช้งานกล้อง