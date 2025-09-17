import asyncio
import websockets
import json

async def test_websocket():
    uri = "ws://localhost:8000/ws/test-client"

    try:
        async with websockets.connect(uri) as websocket:
            print("Connected to WebSocket")

            # Send a test message
            test_message = {
                "message": "Hello, can you help me test the file search tool?",
                "tools": ["file_search"],
                "model": "gemma3:latest"
            }

            await websocket.send(json.dumps(test_message))
            print("Sent test message")

            # Listen for responses
            timeout = 30  # 30 seconds timeout
            start_time = asyncio.get_event_loop().time()

            while True:
                try:
                    # Check timeout
                    if asyncio.get_event_loop().time() - start_time > timeout:
                        print("Timeout reached")
                        break

                    response = await asyncio.wait_for(websocket.recv(), timeout=5)
                    data = json.loads(response)
                    print(f"Received: {data.get('type', 'unknown')} - {data}")

                    if data.get('type') == 'assistant_response_complete':
                        print("Assistant response completed")
                        break

                except asyncio.TimeoutError:
                    print("Waiting for response...")
                    continue
                except Exception as e:
                    print(f"Error: {e}")
                    break

    except Exception as e:
        print(f"Connection error: {e}")

if __name__ == "__main__":
    asyncio.run(test_websocket())