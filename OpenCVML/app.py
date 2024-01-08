import streamlit as st
import cv2
import time

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Load the cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Funtion to detect faces
def detech_face(img):

    # แปลงภาพเป็นสีเทา
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ตรวจจับใบหน้า
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # วาดสี่เหลี่ยมรอบใบหน้า
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 4)
    return img

st.title("Face Detection")
run = st.checkbox("Run", key="run")

FRAME_WINDOW = st.image([])

while run:
    ret, frame = cap.read()
    if not ret:
        continue

    # Detect faces
    frame = detech_face(frame)

    # Update session state with the latest frame
    st.session_state.latest_frame = frame

    # Display the output frame on Streamlit
    FRAME_WINDOW.image(frame, channels="BGR")

    # Save image if button is clicked
    current_time = time.time()
    if st.button('Save Image', key=current_time):
        
        timestamp = int(time.time())
        filename = f"saved_image_{timestamp}.jpg"
        
        # Save the image to the root directory
        cv2.imwrite(f"images/{filename}", frame)

cap.release()
