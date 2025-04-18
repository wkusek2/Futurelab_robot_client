import asyncio
import websockets
import threading
import queue
import hashlib

class WebSocketClient:
    def __init__(self, uri, frame_callback=None, message_callback=None, max_queue_size=10, deduplication_timeout=5):
        """
        A WebSocket client that handles frame data and messaging with thread support.
        
        Args:
            uri (str): WebSocket server URI (e.g., "ws://192.168.1.29:8765")
            frame_callback (callable): Async function to call when frames are received
            message_callback (callable): Async function to call when text messages are received
            max_queue_size (int): Maximum size of the send queue
            deduplication_timeout (float): Time in seconds to remember sent messages for deduplication
        """
        self.uri = uri
        self.frame_callback = frame_callback
        self.message_callback = message_callback
        self.send_queue = asyncio.Queue(max_queue_size)
        self.websocket = None
        self.running = False
        
        # Create an event to signal when the connection is established
        self.connected_event = asyncio.Event()
        
        # Message deduplication system
        self.message_cache = {}  # Maps message hash to timestamp
        self.deduplication_timeout = deduplication_timeout
    
    def _get_message_hash(self, message_type, data):
        """
        Generate a hash for a message to detect duplicates.
        
        Args:
            message_type (str): Type of message ('msg', 'img', etc.)
            data (bytes): Message content
            
        Returns:
            str: Hash representation of the message
        """
        # For binary data, create a hash of the content
        if message_type == 'img':
            # For images, we might only want to hash a small portion or metadata
            # to avoid performance issues with large frames
            return hashlib.md5(data[:1024]).hexdigest()  # Hash first 1KB for performance
        elif message_type == 'msg':
            # For text messages, hash the entire content
            return hashlib.md5(data).hexdigest()
        else:
            # For other types, combine type and a hash of the data
            return f"{message_type}_{hashlib.md5(str(data).encode()).hexdigest()}"
    
    def _is_duplicate_message(self, message_type, data):
        """
        Check if a message is a duplicate of a recently sent message.
        
        Args:
            message_type (str): Type of message
            data (bytes): Message content
            
        Returns:
            bool: True if message is a duplicate, False otherwise
        """
        # Skip deduplication for certain message types
        if message_type == 'close':
            return False
        
        # Calculate message hash
        msg_hash = self._get_message_hash(message_type, data)
        
        # Get current time
        current_time = asyncio.get_event_loop().time()
        
        # Clean expired entries from the cache
        expired_hashes = []
        for hash_key, timestamp in self.message_cache.items():
            if current_time - timestamp > self.deduplication_timeout:
                expired_hashes.append(hash_key)
        
        for hash_key in expired_hashes:
            del self.message_cache[hash_key]
        
        # Check if this is a duplicate message
        if msg_hash in self.message_cache:
            return True
        
        # Not a duplicate, add to cache
        self.message_cache[msg_hash] = current_time
        return False
    
    async def connect(self):
        """Establish WebSocket connection and start handler tasks"""
        try:
            self.websocket = await websockets.connect(self.uri)
            self.running = True
            self.connected_event.set()  # Signal that connection is established
            print(f"[+] Connected to WebSocket server at {self.uri}")
            return True
        except Exception as e:
            print(f"[-] Connection error: {e}")
            return False
    
    async def disconnect(self):
        """Close the WebSocket connection gracefully"""
        if self.websocket:
            await self.send_queue.put(('close', None))
            self.running = False
    
    async def run(self):
        """Run the WebSocket client tasks"""
        if not self.websocket:
            success = await self.connect()
            if not success:
                return
        
        # Start tasks for receiving and sending data
        receive_task = asyncio.create_task(self.receive_loop())
        send_task = asyncio.create_task(self.send_loop())
        
        # Wait for all tasks to complete
        await asyncio.gather(receive_task, send_task)
    
    async def receive_loop(self):
        """Handle incoming WebSocket messages"""
        try:
            while self.running:
                data = await self.websocket.recv()
                
                # Check if it's a text or binary message
                if isinstance(data, str):
                    if data == 'close':
                        await self.disconnect()
                        break
                    # Handle text message
                    if self.message_callback:
                        asyncio.create_task(self.message_callback(data))
                else:
                    # Handle binary frame data - don't block the receive loop
                    if self.frame_callback:
                        asyncio.create_task(self.frame_callback(data))
        
        except websockets.exceptions.ConnectionClosed:
            print("[-] Connection closed by server")
        except Exception as e:
            print(f"[-] Error in receive loop: {e}")
        finally:
            self.running = False
            self.connected_event.clear()
    
    async def send_loop(self):
        """Send queued messages to the WebSocket server"""
        try:
            while self.running:
                message_type, data , = await self.send_queue.get()
                
                # Check if this is a duplicate message
                if self._is_duplicate_message(message_type, data):
                    print(f"[i] Skipping duplicate message of type: {message_type}")
                    self.send_queue.task_done()
                    continue
                
                if message_type == 'close':
                    if self.websocket and self.websocket.open:
                        await self.websocket.close()
                    break
                elif message_type == 'msg-servo-st':
                    if self.websocket and self.websocket.open:
                        await self.websocket.send([message_type,data])
                elif message_type == 'msg-servo-9g':
                    if self.websocket and self.websocket.open:
                        await self.websocket.send([message_type,data])
                elif message_type == 'img':
                    if self.websocket and self.websocket.open:
                        await self.websocket.send(data)
                else:
                    print(f"[-] Unknown message type: {message_type}")
                
                self.send_queue.task_done()
        
        except websockets.exceptions.ConnectionClosed:
            print("[-] Connection closed while sending")
        except Exception as e:
            print(f"[-] Error in send loop: {e}")
        finally:
            self.running = False
            self.connected_event.clear()
    
    async def send_message(self, message):
        """Queue a text message to be sent"""
        if isinstance(message, str):
            message = message.encode('utf-8')
        await self.send_queue.put((message))
    
    async def send_binary(self, data):
        """Queue binary data to be sent"""
        await self.send_queue.put(('img', data))

    def put_in_queue(self, message_type, data):
        """Put a message in the send queue"""
        if self.running:
            asyncio.run_coroutine_threadsafe(self.send_queue.put((message_type, data)), asyncio.get_event_loop())
        else:
            print("[-] WebSocket client is not running")