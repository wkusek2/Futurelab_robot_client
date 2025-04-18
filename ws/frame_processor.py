import struct
import cv2
import numpy as np
from camera.distortion import distortion
from camera.detection import yolo, yolo1
import os
from datetime import datetime
import concurrent.futures
import threading
import queue

class FrameProcessor:
    def __init__(self, save_frames=False, max_queue_size=5):
        """
        Initialize the frame processor that handles decoding and processing camera frames.
        
        Args:
            save_frames (bool): Whether to save frames to disk for debugging/analysis
            max_queue_size (int): Maximum size of the processing queue
        """
        self.save_frames = save_frames
        self.frame_count = 0
        self.processing_queue = queue.Queue(maxsize=max_queue_size)
        self.result_queue = queue.Queue(maxsize=max_queue_size)
        
        # Create a thread pool executor
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
        # Start the processing thread
        self.processing_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self.processing_thread.start()
        
        # Set up directories for saving frames if needed
        if self.save_frames:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.camera0_dir = f"camera0_frames_{self.timestamp}"
            self.camera1_dir = f"camera1_frames_{self.timestamp}"
            
            os.makedirs(self.camera0_dir, exist_ok=True)
            os.makedirs(self.camera1_dir, exist_ok=True)
            
            print(f"[+] Saving frames from camera 0 to folder: {self.camera0_dir}")
            print(f"[+] Saving frames from camera 1 to folder: {self.camera1_dir}")
    
    def _processing_worker(self):
        """Worker thread that processes frames from the queue"""
        while True:
            try:
                # Get a task from the queue
                task_type, data = self.processing_queue.get()
                
                if task_type == "decode":
                    # Decode the frame data
                    frame0, frame1 = self._decode_frame_data(data)
                    
                    # If saving is enabled, submit a separate save task
                    if self.save_frames:
                        self.executor.submit(self._save_frames, frame0, frame1)
                    
                    # Process the frames in a separate thread and get a future
                    future = self.executor.submit(self._process_frames, frame0, frame1)
                    
                    # Add the result to the result queue when done
                    processed_frames = future.result()
                    self.result_queue.put(processed_frames)
                
                # Mark task as done
                self.processing_queue.task_done()
            
            except Exception as e:
                print(f"Error in processing worker: {e}")
                # Mark task as done even if there was an error
                self.processing_queue.task_done()
    
    def _decode_frame_data(self, data):
        """
        Decode binary frame data received from WebSocket.
        
        Args:
            data (bytes): Raw binary data containing frames from both cameras
            
        Returns:
            tuple: A tuple containing frames from both cameras (frame0, frame1)
        """
        # Unpack data
        offset = 0
        
        # Extract first frame length and data
        len0 = struct.unpack('>I', data[offset:offset+4])[0]
        offset += 4
        img0 = np.frombuffer(data[offset:offset+len0], dtype=np.uint8)
        offset += len0
        
        # Extract second frame length and data
        len1 = struct.unpack('>I', data[offset:offset+4])[0]
        offset += 4
        img1 = np.frombuffer(data[offset:offset+len1], dtype=np.uint8)
        
        # Decode JPEG data to frames
        frame0 = cv2.imdecode(img0, cv2.IMREAD_COLOR)
        frame1 = cv2.imdecode(img1, cv2.IMREAD_COLOR)
        
        # Apply distortion correction
        frame0, frame1 = distortion(frame0, frame1)
        
        return frame0, frame1
    
    def _save_frames(self, frame0, frame1):
        """Save frames to disk if enabled"""
        try:
            frame_filename0 = os.path.join(self.camera0_dir, f"frame_{self.frame_count:06d}.jpg")
            frame_filename1 = os.path.join(self.camera1_dir, f"frame_{self.frame_count:06d}.jpg")
            
            cv2.imwrite(frame_filename0, frame0)
            cv2.imwrite(frame_filename1, frame1)
            
            self.frame_count += 1
            
            # Display progress every 100 frames
            if self.frame_count % 100 == 0:
                print(f"[+] Saved {self.frame_count} frames")
        
        except Exception as e:
            print(f"Error saving frames: {e}")
    
    def _process_frames(self, frame0, frame1):
        """
        Process frames with YOLO detection.
        
        Args:
            frame0 (np.ndarray): First camera frame
            frame1 (np.ndarray): Second camera frame
            
        Returns:
            tuple: A tuple containing processed frames from both cameras
        """
        # Apply YOLO detection to frames
        processed_frame0 = yolo.process_frame(frame0)
        processed_frame1 = yolo1.process_frame(frame1)
        
        return processed_frame0, processed_frame1
    
    async def decode_and_process(self, data):
        """
        Asynchronous function to decode and process frames.
        This function doesn't block the async event loop as processing happens in separate threads.
        
        Args:
            data (bytes): Raw binary data containing frames from both cameras
            
        Returns:
            tuple: A tuple containing processed frames from both cameras
        """
        # Put the data in the processing queue
        if not self.processing_queue.full():
            self.processing_queue.put(("decode", data))
        else:
            print("Warning: Processing queue is full, skipping frame")
            # If the queue is full, we need to return something
            # Return None values that the caller should handle
            return None, None
        
        # Check if we have processed frames available
        if not self.result_queue.empty():
            return self.result_queue.get()
        else:
            # If no processed frames are ready yet, return None values
            # The caller should handle this case (e.g., by keeping previous frames)
            return None, None