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
# from app.conn import Communicator
from robot.robot import Robot
from robot.triangulation import Triangulation as Tri
import numpy as np
from camera.detection import yolo, yolo1


# COLUMNS_TEXT PROB NOT NEEDED
COLUMNS_TEXT = ("Id", "Voltage", "Current", "Temperature", "Position", "Load")
COLUMNS_HEADER = ["Id", "Voltage", "Current", "Temperature", "Position", "Load"]

class App:
    def __init__(self, between_cameras, camera_mode_width, camera_mode_height) -> None:
        ########################
        ############## GUI CONST
        ctk.set_appearance_mode("dark")  # Ustawienie ciemnego trybu
        ctk.set_default_color_theme("blue")  # Ustawienie domyślnego motywu kolorystycznego

        self.root = ctk.CTk()  # Użycie CTk zamiast Tk
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
        self.camera0_label.grid(row=0, column=8, padx=1,rowspan=5, pady=10)

        self.camera1_label = ctk.CTkLabel(self.camera_frame, text="")
        self.camera1_label.grid(row=0, column=9, padx=1,rowspan=5, pady=10)

        ########################
        ########## GRID POSITIONS

        ########################
        ########## connection

        ############################
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
        self.canvas.get_tk_widget().grid(column=3, row=0, rowspan=2, columnspan=4)

        #################################
        ########## data
        self.table = ui.table(self.data_frame, COLUMNS_TEXT, "headings", COLUMNS_HEADER, 160, 6, 16, 20, 20, 'nsew')

    def gui(self):
        "Allocate buttons and objects on window"

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
        button3 = ctk.CTkButton(self.navbar, text="Dane", command=lambda: show_frame(self.data_frame))
        button3.grid(row=0, column=2, padx=5, pady=10)

        # show main frame by default
        show_frame(self.main_frame)

        ########################### 
        ########## MAIN FRAME

        ########################
        ########## CAMERA SECTION

        ########################
        ########## CONNECTION SECTION

        ###########################
        ########## DATA FRAME



    async def run_async(self):
        """Run the GUI in an asynchronous loop."""
        self.gui()
        asyncio.create_task(self.update_camera())
        while True:
            self.canvas.draw()
            self.root.update()
            await asyncio.sleep(0.01)

########################################################
####### camera

    def update_camera_frames(self, frame0, frame1):
        """Update the GUI with new camera frames."""
        frame0_resized = cv2.resize(frame0, (self.camera_mode_width, self.camera_mode_height))
        frame1_resized = cv2.resize(frame1, (self.camera_mode_width, self.camera_mode_height))

        frame0_image = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame0_resized, cv2.COLOR_BGR2RGB)))
        frame1_image = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame1_resized, cv2.COLOR_BGR2RGB)))

        self.camera0_label.configure(image=frame0_image)
        self.camera0_label.image = frame0_image

        self.camera1_label.configure(image=frame1_image)
        self.camera1_label.image = frame1_image


    def triangulation_operation(self):
        scale_factor = 4.672916
        scale_factor1 = 2.58125
        
        obj0 = yolo.get_detections_info()
        obj1 = yolo1.get_detections_info()
        if not obj0 or not obj1:
            return None
        x_min0, y_min0, x_max0, y_max0 = obj0
        x_min1, y_min1, x_max1, y_max1 = obj1

        obj0 = (
            x_min0 * scale_factor1,
            y_min0 * scale_factor,
            x_max0 * scale_factor1,
            y_max0 * scale_factor
        )
        
        
        obj1 = (
            x_min1 * scale_factor1,
            y_min1 * scale_factor,
            x_max1 * scale_factor1,
            y_max1 * scale_factor
        )

        # Triangulate the points
        if obj0 is None or obj1 is None:
            return None
        self.points = self.triangulation.get_3d_position(obj0, obj1)
        return self.points
    
    async def update_camera(self):
        while True:
            c = self.triangulation_operation()
            if c is None:
                c = [0, 0, 0]  # Default value if triangulation fails


            # Jeśli istnieje poprzedni punkt, usuń go
            if hasattr(self, 'camera_point'):
                self.camera_point.remove()

            # Narysuj nowy punkt i zapisz referencję
            self.camera_point = self.plot.plot_camera(c[0], c[1], c[2])
            distance = (np.linalg.norm(c) * 100) * 1

            # Aktualizacja canvas
            self.canvas.draw_idle()  # używamy draw_idle() dla lepszej wydajności
            self.label_coord.configure(text=f"Calculated distance: \n{distance:.2f} [cm] \n" + 
                                           f"X: {c[0]:.2f}, Y: {c[1]:.2f}, Z: {c[2]:.2f}")

            # Zaplanuj następną aktualizację
            await asyncio.sleep(0.25)


        
