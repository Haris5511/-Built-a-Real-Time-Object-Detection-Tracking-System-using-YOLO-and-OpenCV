# Real-Time Object Detection & Tracking using YOLO and OpenCV

## Overview

This project implements a real-time object detection and tracking system using Python, YOLO, and OpenCV. It supports webcam input and video file processing, detects objects, draws bounding boxes, and displays tracking IDs.

## Features

- Real-time object detection using YOLO
- Webcam and video file input support
- Object tracking with tracking IDs
- Bounding boxes and detection labels
- Video processing using OpenCV
- Command-line options for selecting the input source

## Tech Stack

- Python
- YOLO (Ultralytics)
- OpenCV
- Computer Vision

## Project Structure

```text
object-detection-tracking/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
└── screenshots/
    └── detection_demo.png
```

*Note: Replace `main.py` with your actual Python script filename if it is different.*

## Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd object-detection-tracking
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

## How to Run

### Run with webcam

```bash
python main.py --source 0
```

### Run with a video file

Place your video file in the project folder, then run:

```bash
python main.py --source video.mp4
```

Make sure the video filename and path are correct.

## Demo

The system displays detected objects with bounding boxes, class labels, and tracking IDs.

Add a screenshot of your actual project output in the `screenshots/` folder.

![Object Detection and Tracking Demo](screenshots/detection_demo.png)

### Demo Video

Add your LinkedIn or YouTube demo link here.

**Demo:** YOUR_DEMO_VIDEO_URL

## Learning Outcomes

Through this project, I gained practical experience in:

- Applying YOLO for object detection
- Processing webcam streams and video files with OpenCV
- Understanding object tracking and tracking IDs
- Building and testing a computer vision application

## Future Improvements

- Add FPS and performance monitoring
- Track and count objects across video frames
- Save detection results and tracking statistics
- Build a web-based interface for the application

## Author

**Haris Azeem**

GitHub: https://github.com/Haris5511

LinkedIn: https://www.linkedin.com/in/haris-azeem-a05187317/

## License

This project is intended for educational and portfolio purposes.
