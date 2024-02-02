import cv2
import numpy as np
import mss
import threading
import win32con
import win32api
import time
import os
import csv
from queue import Queue
import keyboard

# Define your screen width and height
SCREEN_WIDTH = 1920  # Replace with your actual screen width
SCREEN_HEIGHT = 1080  # Replace with your actual screen height

# Create data directory based on starting timestamp
starting_timestamp = int(time.time())
data_directory = fr"..\..\artifacts\data\{starting_timestamp}"
os.makedirs(os.path.join(data_directory, "frames"), exist_ok=True)

# CSV file to store mouse actions
csv_file_path = os.path.join(data_directory, "actions.csv")

# Queue to synchronize frames and mouse click status
frame_queue = Queue()

# Create an Event to control the pause/resume state
pause_event = threading.Event()
pause_event.set()  # Start with the recording in the paused state


# Function to capture game screen using mss
def capture_game_screen():
    monitor = {"top": 0, "left": 0, "width": SCREEN_WIDTH, "height": SCREEN_HEIGHT}
    target_height = 224
    target_width = int((target_height / SCREEN_HEIGHT) * SCREEN_WIDTH)

    with mss.mss() as sct:
        count = 0
        while True:
            pause_event.wait()  # Wait if the recording is paused
            img = np.array(sct.grab(monitor))

            # Resize captured image to 224xX, retaining the aspect ratio
            img = cv2.resize(img, (target_width, target_height))

            timestamp = int(time.time() * 1000)  # Current time in milliseconds
            img_filename = f"{timestamp}.png"
            img_path = os.path.join(data_directory, "frames", img_filename)
            cv2.imwrite(img_path, img)

            # Log the timestamp, labels for mouse and keyboard actions, and mouse coordinates in the frame queue
            mouse_x, mouse_y = win32api.GetCursorPos()
            _, left_clicked, _ = mouse_click_check(0, win32con.VK_LBUTTON)
            left_click = 1 if left_clicked else 0
            _, right_clicked, _ = mouse_click_check(0, win32con.VK_RBUTTON)
            right_click = 1 if right_clicked else 0
            key_states = {
                'w_key': keyboard.is_pressed('w'),
                'a_key': keyboard.is_pressed('a'),
                's_key': keyboard.is_pressed('s'),
                'd_key': keyboard.is_pressed('d'),
                'space_key': keyboard.is_pressed('space'),
                'ctrl_key': keyboard.is_pressed('ctrl'),
                'shift_key': keyboard.is_pressed('shift'),
                '1_key': keyboard.is_pressed('1'),
                '2_key': keyboard.is_pressed('2'),
                '3_key': keyboard.is_pressed('3'),
                'r_key': keyboard.is_pressed('r'),
                'f_key': keyboard.is_pressed('f'),
                'c_key': keyboard.is_pressed('c')
            }

            # Process key states
            processed_key_states = {key: 1 if state else 0 for key, state in key_states.items()}
            count += 1
            frame_queue.put(
                (img_filename, mouse_x, mouse_y, left_click, right_click, *processed_key_states.values()))
            print(count, (int(time.time() - starting_timestamp)) // 60, img_filename, mouse_x, mouse_y, left_click,
                  right_click,
                  *processed_key_states.values())

            time.sleep(1 / 16)  # Capture at 16 FPS


# Function to log mouse actions and keyboard inputs
def log_actions():
    # Open the CSV file in write mode and add the header row
    with open(csv_file_path, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(
            ["Image Filename", "Mouse X", "Mouse Y", "Left Click", "Right Click", "W Key", "A Key", "S Key", "D Key",
             "Space Key", "Ctrl Key", "Shift Key", "1 Key", "2 Key", "3 Key", "R Key", "F Key", "C Key"])

    while True:
        pause_event.wait()  # Wait if the recording is paused
        img_filename, mouse_x, mouse_y, left_click, right_click, w_key, a_key, s_key, d_key, space_key, ctrl_key, \
        shift_key, one_key, two_key, three_key, r_key, f_key, c_key = frame_queue.get()

        # Log the frame filename and labels in the CSV file with rearranged order
        with open(csv_file_path, "a", newline="") as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(
                [img_filename, mouse_x, mouse_y, left_click, right_click, w_key, a_key, s_key, d_key, space_key,
                 ctrl_key, shift_key, one_key, two_key, three_key, r_key, f_key, c_key])

        frame_queue.task_done()
        time.sleep(1 / 16)  # Log actions at 16 FPS


# Your existing mouse check function
def mouse_click_check(previous_status, button):
    current_status = win32api.GetKeyState(button)
    clicked = (current_status < 0) and (previous_status >= 0)
    return current_status, clicked, 0


# Function to toggle pause/resume state using the 'p' key
def toggle_pause():
    while True:
        keyboard.wait("p")  # Wait for 'p' key press
        pause_event.clear()  # Pause the recording
        print("Recording Paused")
        keyboard.wait("p")  # Wait for another 'p' key press to resume
        pause_event.set()  # Resume the recording
        print("Recording Resumed")


# Create four threads for capturing game screen, logging mouse and keyboard actions, and toggling pause/resume
screen_thread = threading.Thread(target=capture_game_screen)
actions_thread = threading.Thread(target=log_actions)
toggle_pause_thread = threading.Thread(target=toggle_pause)

print('Press \'p\' to start recording')
keyboard.wait("p")  # Wait for 'p' key press
# Start the threads
screen_thread.start()
actions_thread.start()
toggle_pause_thread.start()

# Wait for the threads to finish (you can add a termination condition)
screen_thread.join()
actions_thread.join()
toggle_pause_thread.join()
