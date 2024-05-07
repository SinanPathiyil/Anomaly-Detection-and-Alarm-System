import tkinter as tk
from tkinter import filedialog
from functools import partial
import cv2
import torch
import numpy as np
import pygame
import time

# Path to the alarm sound
path_alarm = "alarm.wav"

# Initializing pygame
pygame.init()
pygame.mixer.music.load(path_alarm)

# Loading the model for crowd detection
model_crowd = torch.hub.load('ultralytics/yolov5', 'custom', path="best5cwd.pt")

# Loading the model for violence detection
model_violence = torch.hub.load('ultralytics/yolov5', 'custom', path="best5new.pt")

# Target classes for crowd detection
target_classes_crowd = ['crowd', 'person']

# Target classes for violence detection
target_classes_violence = ['weapon', 'Violence', 'Non-violence']

# Global variables for polygon drawing
pts = []

# Global variable for video capture
cap = None

# Global variable for playback speed
playback_speed = 120

# Global variable for output video frame rate
output_fps = 60

# Global variable to store the time when alarm was last triggered
alarm_triggered_time = 0

# Function to draw polygon
def draw_polygon(event, x, y, flags, param):
    global pts
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        pts.clear()  # Clear the list of points when right-clicked

# Function to check if a point is inside a polygon
def inside_polygon(point, polygon):
    if len(polygon) < 3:
        return False  # Polygon should have at least 3 points

    # Convert polygon to numpy array if it's not already
    polygon = np.array(polygon)

    # Ensure that the polygon has the correct shape (Nx1x2)
    if len(polygon.shape) == 2:
        if polygon.shape[1] != 2:
            return False
        polygon = polygon.reshape((-1, 1, 2))

    # Perform point-in-polygon test
    result = cv2.pointPolygonTest(polygon, (point[0], point[1]), False)
    return result >= 0

# Function to preprocess the image
def preprocess(img):
    height, width = img.shape[:2]
    ratio = height / width
    img = cv2.resize(img, (640, int(640 * ratio)))
    return img

# Function to set the frame rate
def set_frame_rate(cap, playback_speed):
    current_fps = cap.get(cv2.CAP_PROP_FPS)
    new_fps = current_fps * playback_speed
    cap.set(cv2.CAP_PROP_FPS, new_fps)

# Function to start detection from file
def start_detection_from_file(video_path, model, target_classes):
    global cap
    cap = cv2.VideoCapture(video_path)
    set_frame_rate(cap, playback_speed)  # Set the frame rate
    detect_objects(cap, model, target_classes)

# Function to detect objects
def detect_objects(cap, model, target_classes):
    global pts
    global playback_speed
    global output_fps
    global alarm_triggered_time

    # Get the frame width and height
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, output_fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_detected = frame.copy()
        frame = preprocess(frame)
        results = model(frame)
        # Reset the alarm flag at the beginning of each frame
        alarm_triggered = False
        for index, row in results.pandas().xyxy[0].iterrows():
            center_x = None
            center_y = None
            if row['name'] in target_classes:
                name = str(row['name'])
                x1 = int(row['xmin'])
                y1 = int(row['ymin'])
                x2 = int(row['xmax'])
                y2 = int(row['ymax'])
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 3)
                cv2.putText(frame, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                if time.time() - alarm_triggered_time >= 6:  # Check if 5 seconds have passed since the alarm started
                    pygame.mixer.music.stop()  # Stop the alarm
                    alarm_triggered_time = 0  # Reset the alarm timestamp
                if name == 'crowd' or name == 'Violence' or name == 'weapon':
                    if inside_polygon((center_x, center_y), np.array(pts)):
                        if not pygame.mixer.music.get_busy() or (time.time() - alarm_triggered_time) >= 6:
                            pygame.mixer.music.play()
                            alarm_triggered_time = time.time()  # Update the timestamp
                        elif time.time() - alarm_triggered_time >= 6:  # Check if 5 seconds have passed since the alarm started
                            pygame.mixer.music.stop()  # Stop the alarm
                            alarm_triggered_time = 0  # Reset the alarm timestamp
                            
        if len(pts) >= 4:
            frame_copy = frame.copy()
            cv2.fillPoly(frame_copy, np.array([pts]), (0, 255, 0))
            frame = cv2.addWeighted(frame_copy, 0.1, frame, 0.9, 0)
            cv2.polylines(frame, [np.array(pts)], True, (0, 255, 0), 2)
            cv2.putText(frame, "Polygon", (pts[0][0], pts[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Video", frame)
        out.write(frame)  # Write the frame to the output video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()  # Release the VideoWriter
    cv2.destroyAllWindows()

# Bind draw_polygon function to the video window
cv2.namedWindow("Video")
cv2.setMouseCallback("Video", draw_polygon)

def run_detection(video_path, detection_code):
    global model
    global target_classes

    if detection_code == 'crowd':
        model = model_crowd
        target_classes = target_classes_crowd
    elif detection_code == 'violence':
        model = model_violence
        target_classes = target_classes_violence

    start_detection_from_file(video_path, model, target_classes)

def select_video_file(label_var):
    video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
    label_var.set(video_path)

def show_detection_options():
    root_main.withdraw()  # Hide the main window
    root_detection = tk.Toplevel()
    root_detection.title("Detection Options")
    root_detection.geometry("800x600")  # Set window size

    # Center the detection options window
    screen_width = root_main.winfo_screenwidth()
    screen_height = root_main.winfo_screenheight()
    x = (screen_width - 800) // 2
    y = (screen_height - 600) // 2
    root_detection.geometry(f"800x600+{x}+{y}")

    selected_video_label_var = tk.StringVar()
    selected_video_label = tk.Label(root_detection, textvariable=selected_video_label_var, wraplength=400)
    selected_video_label.grid(row=0, column=0, columnspan=2)

    select_video_button = tk.Button(root_detection, text="Select Video", command=partial(select_video_file, selected_video_label_var), width=20, height=2)
    select_video_button.grid(row=1, column=0, columnspan=2, pady=10, padx=10, sticky=tk.W+tk.E)

    detection_code_var = tk.StringVar()
    detection_code_var.set("crowd")
    detection_code_label = tk.Label(root_detection, text="Select Detection Code:")
    detection_code_label.grid(row=2, column=0, columnspan=1, pady=10, padx=10)
    detection_code_menu = tk.OptionMenu(root_detection, detection_code_var, "crowd", "violence")
    detection_code_menu.grid(row=2, column=1, columnspan=2, pady=10, padx=10)

    run_detection_button = tk.Button(root_detection, text="Run Detection", command=lambda: run_detection(selected_video_label_var.get(), detection_code_var.get()), width=20, height=2)
    run_detection_button.grid(row=3, column=0, columnspan=2, pady=10, padx=10)

    def show_main_window():
        root_detection.destroy()
        root_main.deiconify()

    btn_back = tk.Button(root_detection, text="Back", command=show_main_window, width=20, height=2)
    btn_back.grid(row=4, column=0, columnspan=2, pady=10)

    # Center buttons horizontally
    root_detection.grid_rowconfigure(3, weight=1)
    root_detection.grid_columnconfigure(0, weight=1)
    root_detection.grid_columnconfigure(1, weight=1)

root_main = tk.Tk()
root_main.title("Object Detection")

canvas_main = tk.Canvas(root_main, width=640, height=480)  # Reduced canvas size
canvas_main.pack()

btn_detection = tk.Button(root_main, text="Start", command=show_detection_options, bg="#cccccc", width=20, height=2)
btn_detection.place(relx=0.5, rely=0.5, anchor=tk.CENTER, y=0)

root_main.mainloop()
