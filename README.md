# Project Description

This repository contains the code for a senior design/research project developed at Arkansas Tech University. The project aims to implement real-time semantic segmentation on a Raspberry Pi 4 using Coral acceleration for real-world litter detection. 

The purpose for this repository is to provide a work around for Python version dependencies. When trying to use `imshow` function in a Python 3.9 venv with Picamera2 frame capture() function, the window does not appear.

It utilizes the custom YOLOv11n-seg models created via `ultralytics` and leverages Google's `pycoral` library for hardware acceleration on a Coral USB Accelerator.
The models are in a seperate repository due to licensing and can be found here: [https://github.com/benl4212/trash-bot-models](https://github.com/benl4212/trash-bot-models)

**Note:** This project was developed for academic purposes (undergraduate senior design/research).


## Features

* POSIX_IPC to connect processes
   1. Camera Frame Capture
   2. Accelerated Model Inference

* Outputs an inference window to visualize segmentation / detections
  * Shows FPS and number of objects detected

* Compatible Models: yolo11n-seg_int8_edgetpu.tflite
  * Will require changes for box models



# Setup & Usage

## Prerequisites 
### Change RPI4 Windowing System
* You will need to change the RPI4's windowing system from Wayland to X11 for compatibility with Opencv's GUI
* You can easily check the windowing system with the command: `echo $XDG_CURRENT_DESKTOP`
* To manually change to X11, see: [How to Switch from Wayland to X11 on Raspberry Pi OS Bookworm](https://www.geeks3d.com/20240509/how-to-switch-from-wayland-to-x11-on-raspberry-pi-os-bookworm/)

### Create Python 3.11 virtual environment for picamera2 compatibility
Open new terminal on RPI4
```bash
# In your Python 3.11+ environment
# Follow setup instructions for picamera2 (if necessary)

pip install posix_ipc numpy picamera2
```
* For full list of dependencies, see this installation guide:   [Picamera2 Installation](https://github.com/raspberrypi/picamera2/blob/main/README.md)


### Create Python 3.9 virtual environment for pycoral compatibility
Open new terminal on RPI4
```bash
# Activate your Python 3.9 environment
# Follow setup instructions for pycoral
pip install posix_ipc numpy opencv-python pycoral
```
* For full list of dependencies, see this installation guide:   [Get Started with Pycoral USB](https://coral.ai/docs/accelerator/get-started/)

## REBOOT RPI4 AFTER INSTALLING NEW SOFTWARE AND CHANGING WINDOWING SYSTEM!


## Running Scripts Seperately
* To run these scripts in their respective environments, use the provided CLI line
  underneath "RUN THIS SCRIPT FIRST" or "RUN THIS SCRIPT 2nd"
* You will probably have to change directory paths based on your setup.


## Ending the Processes
* Use 'ctrl + C' in one of the terminals
* Close both terminals.
* There is an issue with the rpicamera not ending its process correctly, you will get an error such as 'pipeline in use' if you try to rerun in the same terminals.



# Licensing:
This project combines code developed by the authors with several third-party libraries under various licenses. The original code specific to this project (i.e., the code written by the project authors, excluding third-party libraries) is licensed under the MIT License. A copy of the MIT License can be found in the LICENSE file in the root of this repository. You are free to use, modify, and distribute this specific code under the terms of the MIT License.MIT License

Copyright (c) 2025 Benjamin Leon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
