import os
import time
import cv2
import numpy as np
from collections import deque
import torch
from ultralytics import YOLO




class YOLODetector:
    def __init__(self, model_fn, min_conf_threshold=0.25, imgW=480, imgH=480):
        """
        Initialize YOLO detector with GPU support
        
        Args:
            model_fn: path to YOLO model file
            min_conf_threshold: minimum confidence threshold for detections
            imgW: width to resize input frame to
            imgH: height to resize input frame to
        """
        # Set path to model
        cwd = os.getcwd()
        self.model_path = os.path.join(cwd, model_fn)
        
        # Set parameters
        self.imgW = imgW
        self.imgH = imgH
        self.min_conf_threshold = min_conf_threshold
        
        # Set up buffer for frame rate calculation
        self.frame_rate_calcs = deque([], maxlen=100)
        self.frame_rate_avg = 0
        self.detections_info = []
        self.detections_online = []
        # Check for CUDA availability
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model and labels with GPU support
        self.model = YOLO(self.model_path)  # pretrained YOLOv8 model
        # Explicitly set the model to use GPU
        if torch.cuda.is_available():
            self.model.to(self.device)
        self.labels = self.model.names
        
        # Define bounding box colors
        self.bbox_colors = [(167,121,78), (43,142,242), (89,87,225), (178,183,118), 
                          (79,161,89), (72,201,237), (161,122,176), (167,157,255), 
                          (95,117,156), (175,176,186)]
    
    def process_frame(self, frame):
        """
        Process a single frame with YOLO detection - limited to detecting only one object
        with the highest confidence score
        
        Args:
            frame: input frame to process
            
        Returns:
            processed_frame: frame with detection results drawn
            detections_info: list containing at most one detection with the highest confidence
        """
        if frame is None or len(frame) == 0:
            return None, []
        
        # Start timer for calculating framerate
        t_start = time.perf_counter()
        
        # Resize frame
        resized_frame = cv2.resize(frame, (self.imgW, self.imgH))
        
        # Run inference on GPU
        results = self.model(resized_frame, verbose=False, device=self.device)
        detections = results[0].boxes
        
        # Create a copy of the frame to draw on
        processed_frame = resized_frame.copy()
        
        # Store detection information
        
        
        # Find the detection with highest confidence
        best_detection = None
        highest_conf = 0
        
        # Go through each detection
        for i in range(len(detections)):
            # Get bounding box confidence
            conf = detections[i].conf.item()
            
            # Check if this detection has higher confidence
            if conf > highest_conf and conf > self.min_conf_threshold:
                highest_conf = conf
                best_detection = i
        
        # Process only the best detection if one was found
        if best_detection is not None:
            i = best_detection
            
            # Get bounding box coordinates
            xyxy_tensor = detections[i].xyxy.cpu()  # Move tensor to CPU for numpy conversion
            xyxy = xyxy_tensor.numpy().squeeze()
            xmin, ymin, xmax, ymax = xyxy.astype(int)
            
            # Get bounding box class ID and name
            classidx = int(detections[i].cls.item())
            classname = self.labels[classidx]
            
            # Get bounding box confidence
            conf = detections[i].conf.item()
            
            # Store detection info
            detection_info = {
                'bbox': (xmin, ymin, xmax, ymax),
                'class_id': classidx,
                'class_name': classname,
                'confidence': conf
            }
            self.detections_online = [xmin, ymin, xmax, ymax]   

            self.detections_info.append(detection_info)
            
            # Draw box
            color = self.bbox_colors[classidx % 10]
            cv2.rectangle(processed_frame, (xmin, ymin), (xmax, ymax), color, 2)
            
            label = f'{classname}: {int(conf*100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)
            cv2.rectangle(processed_frame, (xmin, label_ymin-labelSize[1]-10), 
                        (xmin+labelSize[0], label_ymin+baseLine-10), color, cv2.FILLED)
            cv2.putText(processed_frame, label, (xmin, label_ymin-7), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Always set object count to 0 or 1
        object_count = 1 if best_detection is not None else 0
        
        # Add FPS and object count info
        cv2.putText(processed_frame, f'FPS: {self.frame_rate_avg:0.2f}', (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(processed_frame, f'Number of objects: {object_count}', (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Calculate FPS
        t_stop = time.perf_counter()
        t_total = t_stop - t_start
        self.frame_rate_calcs.appendleft(1/t_total)
        self.frame_rate_avg = np.mean(self.frame_rate_calcs)
        
        return processed_frame
    
    def get_detections_info(self):
        """
        Get the detection information
        
        Returns:
            detections_info: list containing detection information
        """
        return self.detections_online
    
yolo = YOLODetector(model_fn="my_model.pt")
yolo1 = YOLODetector(model_fn="my_model.pt")