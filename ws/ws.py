import asyncio
import websockets


class wsManager:
    def __init__(self, websocket):
        self.send_queue = asyncio.Queue(10)  # Poprawiona nazwa zmiennej
        self.websocket = websocket
        
    async def connection_handler(self):
        try:
            await asyncio.gather(
                self.recv(),
                self.sender()  # Zmieniona nazwa metody żeby nie kolidowała z nazwą zmiennej
            )
        except Exception as e:
            print(f"Error in connection handler: {e}")
            await self.websocket.close()
        finally:
            await self.websocket.close()
            print("WebSocket connection closed")

    async def sender(self):  # Zmieniona nazwa metody
        while True:
            frame_data = await self.send_queue.get()
            if frame_data[0] == 'img':
                await self.websocket.send(frame_data)
            elif frame_data[0] == 'msg':
                await self.websocket.send(frame_data)
            elif frame_data[0] == 'close':
                await self.websocket.close()
                break
            else:
                print("Unknown frame type")
                break
                
    async def recv(self):
        while True:
            try:
                data = await self.websocket.recv()
                if data == 'close':
                    await self.websocket.close()
                    break
                else:
                    print("Received data:", data)
            except websockets.ConnectionClosed:
                print("Connection closed")
                break
            except Exception as e:
                print(f"Error receiving data: {e}")
                break