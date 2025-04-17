import asyncio
import websockets
import struct
import cv2
import numpy as np
from app.gui import App
from camera.distortion import distortion
from camera.detection import YOLODetector, yolo, yolo1
import os
from datetime import datetime
import time
import queue

send_queue = asyncio.Queue(100)


save = False

if save:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    camera0_dir = f"camera0_frames_{timestamp}"
    camera1_dir = f"camera1_frames_{timestamp}"

    os.makedirs(camera0_dir, exist_ok=True)
    os.makedirs(camera1_dir, exist_ok=True)

    print(f"[+] Zapisywanie klatek z kamery 0 do folderu: {camera0_dir}")
    print(f"[+] Zapisywanie klatek z kamery 1 do folderu: {camera1_dir}")


async def send_time_loop():
    while True:
        now = datetime.now().strftime("%H:%M:%S")
        message = f"Czas serwera: {now}".encode("utf-8")
        await send_queue.put(message)
        await asyncio.sleep(1)



async def receive_frames(app, websocket):
        frame_count = 0
        while True:
            data = await websocket.recv()
            
            # Unpack data
            offset = 0
            len0 = struct.unpack('>I', data[offset:offset+4])[0]
            offset += 4
            img0 = np.frombuffer(data[offset:offset+len0], dtype=np.uint8)
            offset += len0

            len1 = struct.unpack('>I', data[offset:offset+4])[0]
            offset += 4
            img1 = np.frombuffer(data[offset:offset+len1], dtype=np.uint8)

            # Decode JPEG
            frame0 = cv2.imdecode(img0, cv2.IMREAD_COLOR)
            frame1 = cv2.imdecode(img1, cv2.IMREAD_COLOR)

            frame0, frame1 = distortion(frame0, frame1)
            
         # Zapisujemy klatki jako pliki JPG
            
            
            #
       
       
            if save:
                frame_filename0 = os.path.join(camera0_dir, f"frame_{frame_count:06d}.jpg")
                frame_filename1 = os.path.join(camera1_dir, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(frame_filename0, frame0)
                cv2.imwrite(frame_filename1, frame1)
                frame_count += 1
                # 
                # Co 100 klatek wyświetlamy informację
                if frame_count % 100 == 0:
                    print(f"[+] Zapisano {frame_count} klatek")
            # Update GUI with frames
            frame0 = yolo.process_frame(frame0)
            frame1 = yolo1.process_frame(frame1)

            

            
            app.update_camera_frames(frame0, frame1)
            
            
async def send_handler(websocket):
    try:
        while True:
            data =  await send_queue.get()
            await websocket.send(data)
            print("Wys")
            send_queue.task_done()

    except websockets.exceptions.ConnectionClosed:
        print("Connection closed while sending")

    except Exception as e:
        print(f"Error in send handler: {e}")

async def websocket_loop(app):
    uri = "ws://192.168.1.29:8765"  # IP serwera
    async with websockets.connect(uri) as websocket:
        # Uruchom równolegle odbieranie i wysyłanie
        await asyncio.gather(
            receive_frames(app, websocket),
            send_handler(websocket),
            send_time_loop()
            )

async def main():
    # Initialize the GUI
    

    app = App(between_cameras=0, camera_mode_width=432, camera_mode_height=768)

    # Run the GUI and WebSocket receiver concurrently
    await asyncio.gather(
        websocket_loop(app),
        app.run_async()
    )

if __name__ == "__main__":
    asyncio.run(main())
