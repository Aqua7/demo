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
import numpy as np
import math # Make sure to import the math library
# This AI logic is a simple demonstration. It aims at the enemy
# when one is present. You can modify this function to create
# more complex and intelligent behavior.
from Neural_heartv import NeuralNetwork
import pickle

with open('last_trained.pkl', 'rb') as h:
    nn = pickle.load(h)
   

def get_ai_input(state):
    """
    Analyzes the game state and returns the AI's input.
    """
    cannon_angle = (state['cannon']['angle'] + np.pi) % (2 * np.pi)- np.pi

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
        
        # Calculate the relative enemy position
        relative_x = (enemy_x - cannon_x) #/(0.5 wedth  )
        relative_y = (enemy_y - cannon_y) #/(0.5 Highth)
        # remove above coments to normalize relative position between -1,1
        calibrate=nn.predict(np.array([relative_x,relative_y,cannon_angle]))
        #calibrate has the form [[value]] use calibrate[0][0]
        #now you have the predicted value of angle difference
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
        # i think you can erase this fucking else.
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
