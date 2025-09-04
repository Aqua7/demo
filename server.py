# This is the Python WebSocket server.
# 
# To use this, save this code as a file named `server.py`.
# You will also need to install the `websockets` library:
#   `pip install websockets`
# 
# Then, run the server from your terminal:
#   `python server.py`
# 
# The server will run on `localhost:8081`.
# It receives game state from the client, calculates the AI's input,
# and sends it back to the client.

import asyncio
import json
import websockets
import math # Make sure to import the math library
# This AI logic is a simple demonstration. It aims at the enemy
# when one is present. You can modify this function to create
# more complex and intelligent behavior.
def get_ai_input(state):
    """
    Analyzes the game state and returns the AI's input.
    """
    cannon_angle = state['cannon']['angle']
    input = {
        'left': False,
        'right': False,
        'shoot': False
    }

    # If an enemy exists, aim at it
    if state['enemy']:
        enemy_x = state['enemy']['x']
        enemy_y = state['enemy']['y']
        cannon_x = state['cannon']['x']
        cannon_y = state['cannon']['y']
        
        # Calculate the angle to the enemy using atan2
        dx = enemy_x - cannon_x
        dy = enemy_y - cannon_y
        
        # atan2 returns the angle in radians. The canvas rotation is also in radians.
        target_angle = math.atan2(dy, dx)
        
        # Adjust cannon angle to face the target
        angle_diff = target_angle - cannon_angle
        
        # Normalize the angle difference to be within -pi to pi
        if angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        if angle_diff < -math.pi:
            angle_diff += 2 * math.pi
            
        # The AI will rotate to face the target
        if angle_diff > 0.05: # Use a small tolerance
            input['right'] = True
        elif angle_diff < -0.05:
            input['left'] = True
        
        # The AI will shoot when it's pointed roughly at the enemy
        if abs(angle_diff) < 0.1: # A small tolerance in radians
            input['shoot'] = True
            
    # The AI will also shoot if there's no enemy, just for fun
    else:
        # A simple, non-aimed shot every 30 frames
        if state['frameCount'] % 30 == 0:
            input['shoot'] = True

    return input

async def handler(websocket, path):
    """
    Handles a single WebSocket connection.
    """
    try:
        # The AI receives game state from the client
        async for message in websocket:
            state = json.loads(message)
            
            # The AI makes its decision
            ai_input = get_ai_input(state)
            
            # The AI's input is sent back to the client
            await websocket.send(json.dumps(ai_input))
            
    except websockets.exceptions.ConnectionClosed as e:
        print(f"Client disconnected: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

async def main():
    """
    Starts the WebSocket server.
    """
    # The server will run on localhost, port 8081
    async with websockets.serve(handler, "localhost", 8081):
        print("WebSocket server started on ws://localhost:8081")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
