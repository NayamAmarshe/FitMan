# FitMan
Get fit while playing games.😄  
Fitman python script allows you to play your favorite racing games using your body movement.  
**Work while you play.**

## Installation

* Install Python3 in your PC.
* Open terminal/powershell/cmd and enter: `pip install mediapipe opencv-python pyautogui`
* Then `git clone https://github.com/NayamAmarshe/FitMan.git && cd FitM*`
* Then launch the AIController.py script by double clicking and opening with Python  
* **OR**  
`python3 AIController*`  
**OR**
`python AIController*` (Whatever works for you)  

**Linux users need to download older commit, that uses pyautogui and then need to ** `sudo apt-get install python3-tk python3-dev`

## Usage

### **IDLE**  
<a href="https://ibb.co/rbdG9v3"><img src="https://i.ibb.co/n3PspRB/image.png" alt="image" border="0" width="200px"></a>

### **ACCELERATE**  
<a href="https://ibb.co/56xwV7h"><img src="https://i.ibb.co/s2JzrB9/image.png" alt="image" border="0" width="200px"></a>

### **TURN RIGHT WITH ACCELERATION**  
<a href="https://ibb.co/ZXL7zxz"><img src="https://i.ibb.co/DWVTfQf/image.png" alt="image" border="0" width="200px"></a>

### **TURN RIGHT WITHOUT ACCELERATION**
<a href="https://ibb.co/sbT9XdL"><img src="https://i.ibb.co/QpWvx3s/image.png" alt="image" border="0" width="200px"></a>

### **TURN LEFT WITH ACCELERATION**  
<a href="https://ibb.co/8PHTHts"><img src="https://i.ibb.co/h2ShSw9/image-4.png" alt="image-4" border="0" width="200px"></a>

### **TURN LEFT WITHOUT ACCELERATION**
<a href="https://ibb.co/5BDLXSR"><img src="https://i.ibb.co/f4RHVyx/New-Project-5.png" alt="New-Project-5" border="0" width="200px"></a>

### **STOP/REVERSE**  
<a href="https://ibb.co/ZGzDNSN"><img src="https://i.ibb.co/9cH5GhG/image.png" alt="image" border="0" width="200px"></a>

### **PRESS ENTER OR SPACE (TO RESTART GAMES)**
<a href="https://ibb.co/zNchQFV"><img src="https://i.ibb.co/M9TDZfk/image.png" alt="image" border="0" width="200px"></a>

### Quit the Window using Esc on Keyboard

## Troubleshooting

If your camera doesn't work and shows a blank screen, please try changing the number in line 38
`cap = cv2.VideoCapture(0)`
You can try 1,2,3 instead of 0 to see which port is working.
