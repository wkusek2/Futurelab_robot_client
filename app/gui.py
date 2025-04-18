import customtkinter as ctk
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import asyncio
import time
import math
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import app.ui as ui
from app.plot import Plot
from robot.robot import Robot
from robot.triangulation import Triangulation as Tri
import numpy as np


# COLUMNS_TEXT PROB NOT NEEDED
COLUMNS_TEXT = ("Id", "Voltage", "Current", "Temperature", "Position", "Load")
COLUMNS_HEADER = ["Id", "Voltage", "Current", "Temperature", "Position", "Load"]

class App:
    def __init__(self, between_cameras, camera_mode_width, camera_mode_height) -> None:
        ########################
        ############## GUI CONST
        ctk.set_appearance_mode("dark")  # Set dark mode
        ctk.set_default_color_theme("blue")  # Set default color theme

        self.root = ctk.CTk()  # Use CTk instead of Tk
        self.root.title("Robot Arm Inverse Kinematics")
        self.root.geometry("2880x1800")

        ########################
        ################ NAVBAR AND FRAMES
        self.navbar = ctk.CTkFrame(self.root)
        self.navbar.grid(row=0, column=0, sticky="ew")

        self.main_frame = ctk.CTkFrame(self.root)
        self.camera_frame = ctk.CTkFrame(self.root)
        self.camera_frame.grid(row=0, column=0, sticky="nsew")
        self.data_frame = ctk.CTkFrame(self.root)

        self.frames = {
            "main_frame": self.main_frame,
            "camera_frame": self.camera_frame,
            "data_frame": self.data_frame
        }

        self.points = None

        ########################
        ################ OBJECTS
        self.plot = Plot()
        self.robot = Robot()
        self.triangulation = Tri()

        ########################
        ########## GUI VARIABLES
        self.sliders = []

        ###############################
        ########## camera
        self.camera_mode_width = camera_mode_width
        self.camera_mode_height = camera_mode_height

        self.camera0_label = ctk.CTkLabel(self.camera_frame, text="")
        self.camera0_label.grid(row=0, column=8, padx=1, rowspan=5, pady=10)

        self.camera1_label = ctk.CTkLabel(self.camera_frame, text="")
        self.camera1_label.grid(row=0, column=9, padx=1, rowspan=5, pady=10)

        ########################
        ########## positions
        self.robot_position_label = ctk.CTkLabel(self.camera_frame, text="Robot Position")
        self.robot_position_label.grid(row=5, column=5, padx=1, pady=10)

        self.label_coord = ctk.CTkLabel(self.camera_frame, text=
                                        "End-Effector Coordinates:\nX: 0.00 \nY: 0.00 \nZ: 0.00")
        self.label_coord.grid(row=6, column=5, rowspan=2, padx=10, pady=0, sticky="nsew")

        ############################
        ########## plot_robot
        self.plot.plot_robot(self.robot, math.pi, -math.pi/2, 0, 0)
        self.canvas = FigureCanvasTkAgg(self.plot.fig, master=self.camera_frame)
        self.canvas.get_tk_widget().grid(column=3, row=0, rowspan=4, columnspan=4)

        #################################
        ########## data
        self.table = ui.table(self.data_frame, COLUMNS_TEXT, "headings", COLUMNS_HEADER, 160, 6, 16, 20, 20, 'nsew')
        
        # Initialize the camera point as None so we can update it later
        self.camera_point = None

        #########################
        ########## ws_client
        self.ws_client = None  # Placeholder for WebSocket client

        ##########################
        ########## database
        self.database = None  # Placeholder for database
        
        ##########################
        ########## Sliders

        
        self.offset1 = ui.slider(self.camera_frame, 0, 8, 0, 4096, 180, 0, 0, 1,
                                 "nsew",dataType="offset0", database=self.database)
        self.offset2 = ui.slider(self.camera_frame, 1, 8, 0, 4096, 180, 0, 1, 1,
                                 "nsew",dataType="offset1", database=self.database)
        self.offset3 = ui.slider(self.camera_frame, 2, 8, 0, 4096, 180, 0, 2, 1,
                                 "nsew",dataType="offset2", database=self.database)
        self.offset4 = ui.slider(self.camera_frame, 3, 8, 0, 4096, 180, 0, 3, 1,
                                 "nsew",dataType="offset3", database=self.database)
        self.offset5 = ui.slider(self.camera_frame, 4, 8, 0, 4096, 180, 0, 4, 1,
                                 "nsew",dataType="offset4", database=self.database)

        
         
        

    def gui(self):
        """Allocate buttons and objects on window"""

        ########################
        ########## CREATE FRAMES (TABS ON NAVBAR)
        def show_frame(frame):
            for f in self.frames.values():
                f.grid_forget()
            frame.grid(row=1, column=0, sticky="nsew")

        button1 = ctk.CTkButton(self.navbar, text="Program", command=lambda: show_frame(self.main_frame))
        button1.grid(row=0, column=0, padx=5, pady=10)
        button2 = ctk.CTkButton(self.navbar, text="Camera", command=lambda: show_frame(self.camera_frame))
        button2.grid(row=0, column=1, padx=5, pady=10)
        button3 = ctk.CTkButton(self.navbar, text="Data", command=lambda: show_frame(self.data_frame))
        button3.grid(row=0, column=2, padx=5, pady=10)

        ################
        ########## SLIDERS
        
        

        # show main frame by default
        show_frame(self.main_frame)

    async def run_async(self):
        """Run the GUI in an asynchronous loop."""
        self.gui()
        asyncio.create_task(self.update_camera_visualization())
        while True:
            self.canvas.draw()
            self.root.update()
            await asyncio.sleep(0.01)

    def update_camera_frames(self, frame0, frame1):
        """Update the GUI with new camera frames."""
        # Resize frames to fit in the GUI
        frame0_resized = cv2.resize(frame0, (self.camera_mode_width, self.camera_mode_height))
        frame1_resized = cv2.resize(frame1, (self.camera_mode_width, self.camera_mode_height))

        # Convert frames to ImageTk format
        frame0_image = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame0_resized, cv2.COLOR_BGR2RGB)))
        frame1_image = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame1_resized, cv2.COLOR_BGR2RGB)))

        # Update labels with new images
        self.camera0_label.configure(image=frame0_image)
        self.camera0_label.image = frame0_image  # Keep a reference to prevent garbage collection

        self.camera1_label.configure(image=frame1_image)
        self.camera1_label.image = frame1_image  # Keep a reference to prevent garbage collection

    def triangulation_operation(self):
        """Perform triangulation to calculate 3D position from detections"""
        scale_factor = 4.672916
        scale_factor1 = 2.58125
        
        # Get detection information from YOLO
        obj0 = self.get_detection_from_yolo(0)
        obj1 = self.get_detection_from_yolo(1)
        
        if not obj0 or not obj1:
            return None
            
        # Scale the detection coordinates
        obj0 = (
            obj0[0] * scale_factor1,
            obj0[1] * scale_factor,
            obj0[2] * scale_factor1,
            obj0[3] * scale_factor
        )
        
        obj1 = (
            obj1[0] * scale_factor1,
            obj1[1] * scale_factor,
            obj1[2] * scale_factor1,
            obj1[3] * scale_factor
        )

        # Calculate 3D position using triangulation
        self.points = self.triangulation.get_3d_position(obj0, obj1)
        return self.points
    
    def get_detection_from_yolo(self, camera_index):
        """Get detection information from YOLO for a specific camera"""
        from camera.detection import yolo, yolo1
        
        if camera_index == 0:
            return yolo.get_detections_info()
        else:
            return yolo1.get_detections_info()
    
    async def update_camera_visualization(self):
        """Update the 3D visualization based on camera triangulation"""
        while True:
            # Get triangulated coordinates
            c = self.triangulation_operation()
            if c is None:
                c = [0, 0, 0]  # Default value if triangulation fails

            # Remove previous camera point if it exists
            if self.camera_point is not None:
                self.camera_point.remove()

            # Draw new point and save reference
            self.camera_point = self.plot.plot_camera(c[0], c[1], c[2])
            
            # Calculate distance
            distance = (np.linalg.norm(c) * 100) * 1

            # Update canvas and display information
            self.canvas.draw_idle()  # Use draw_idle() for better performance
            self.label_coord.configure(text=f"Calculated distance: \n{distance:.2f} [cm] \n" + 
                                         f"X: {c[0]:.2f}, Y: {c[1]:.2f}, Z: {c[2]:.2f}")

            # Schedule next update
            await asyncio.sleep(0.25)
    
    def set_websocket_client(self, client):
        """Set the WebSocket client for the application."""
        self.ws_client = client

    async def send_to_queue(self, message_type, message):
        """Send a message to the WebSocket queue."""
        if self.ws_client:
            await self.ws_client.put_in_queue(message_type, message)
        else:
            print("WebSocket client not initialized.")

    def set_database(self, database):
        """Set the database for the application."""
        self.database = database