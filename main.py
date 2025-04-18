import asyncio
from app.gui import App
from ws.frame_processor import FrameProcessor
from ws.ws import WebSocketClient
from database.database import Database

# Configuration
WEBSOCKET_URI = "ws://192.168.1.29:8765"
SAVE_FRAMES = False
CAMERA_WIDTH = 432
CAMERA_HEIGHT = 768



async def main():
    """Main application function that coordinates GUI and WebSocket communication"""
    # Initialize the GUI app
    app = App(between_cameras=0, camera_mode_width=CAMERA_WIDTH, camera_mode_height=CAMERA_HEIGHT)
    
    # Initialize frame processor
    frame_processor = FrameProcessor(save_frames=SAVE_FRAMES)
    
    # Store the last valid frames to handle cases when new processed frames aren't ready
    last_valid_frames = (None, None)
    
    # Create a callback function to handle incoming frames
    async def handle_frame_data(data):
        nonlocal last_valid_frames
        
        # Decode and process the incoming frames (this now happens in separate threads)
        processed_frames = await frame_processor.decode_and_process(data)
        
        # Check if we got valid frames back
        if processed_frames[0] is not None and processed_frames[1] is not None:
            # Update our cached frames
            last_valid_frames = processed_frames
            
            # Update the GUI with processed frames
            app.update_camera_frames(processed_frames[0], processed_frames[1])
        elif last_valid_frames[0] is not None and last_valid_frames[1] is not None:
            # If no new frames are ready but we have previous ones, use those
            app.update_camera_frames(last_valid_frames[0], last_valid_frames[1])
    
    # Initialize WebSocket client with our frame handler
    ws_client = WebSocketClient(
        uri=WEBSOCKET_URI,
        frame_callback=handle_frame_data,
        message_callback=lambda msg: print(f"Received message: {msg}")
    )

    db = Database()  # Initialize the database

    app.set_websocket_client = ws_client  # Store the WebSocket client in the app for potential use
    app.set_database = db  # Store the database in the app for potential use    
    
    # Run all components concurrently
    try:
        await asyncio.gather(
            ws_client.run(),
            app.run_async()
        )
    except KeyboardInterrupt:
        print("\n[+] Shutting down...")
    finally:
        # Ensure proper cleanup
        await ws_client.disconnect()

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())