# Face Recognition Attendance System

A real-time face recognition-based attendance system using **TensorFlow**, **OpenCV**, and **Tkinter**. This system detects and recognizes faces from a live webcam feed, logs attendance, and provides accuracy metrics.

## 🚀 Features

- Face enrollment with two-session capture (varied lighting/angles)
- Real-time face detection and recognition
- Attendance logging with CSV file creation per day
- Duplicate entry prevention (5-minute cooldown)
- Basic GUI using Tkinter
- Accuracy calculation based on detection confidence

## 🖼 GUI Preview

> GUI includes a video frame, control buttons (Enroll, Start, Stop), and a live accuracy display.

## 📦 Requirements

- Python 3.x
- OpenCV
- TensorFlow
- NumPy
- Pandas
- PIL (Pillow)

Install dependencies with:

```
pip install opencv-python tensorflow numpy pandas pillow
```

## 📁 Directory Structure

```
FaceRecognitionSystem/
├── enrolled_faces/         # Saved face samples during enrollment
├── Attendance_Records/     # Daily CSV logs of attendance
├── face_recognition_app.py # Main application script
```

## 🛠 How to Run

```
python face_recognition_app.py
```

## Enrollment Steps

1. Click **"Enroll New Face"**.
2. Enter a name.
3. Look at the camera – Session 1 begins (50 samples).
4. Move to different lighting – Session 2 begins (50 samples).
5. Face samples saved and feature vectors extracted.

## Recognition Mode

- Click **"Start Recognition"** to begin.
- Detected faces are matched with enrolled data.
- Attendance is marked in `Attendance_Records/attendance_YYYY-MM-DD.csv`.

## Accuracy

- Accuracy displayed live in the GUI.
- Based on cosine similarity between known and detected faces.
