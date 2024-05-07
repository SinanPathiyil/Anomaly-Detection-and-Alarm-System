# Object Detection with Alarm System

This program detects objects in a video feed using YOLOv5 models for crowd detection and violence detection. It provides an option to draw a polygon on the video frame, and if any detected object enters this polygon, an alarm sound is played.

## Requirements

- Python 3.x
- OpenCV
- PyTorch
- NumPy
- Pygame
- Tkinter

## Installation

1. Clone this repository to your local machine:

git clone <repository_url>


## Usage

1. Run the program using the following command:

python interface.py



2. Click on the "Start" button to select a video file and choose the detection code (crowd or violence).

3. Draw a polygon on the video frame by clicking points with the left mouse button. Right-click to clear the polygon.

4. The program will play an alarm sound if any detected object enters the drawn polygon. The alarm stops after 5 seconds of continuous triggering.

## Additional Files

- `alarm.wav`: Alarm sound file.
- `best5cwd.pt`: YOLOv5 model file for crowd detection.
- `best5new.pt`: YOLOv5 model file for violence detection.
- `yolov5_custom_training.ipynb`: Colab Notebook file for custom dataset training of YOLOv5 models. (both models given here are made using custom dataset in YOLOv5)

## Credits

- YOLOv5: [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)

## License

This project is licensed under the MIT License.
