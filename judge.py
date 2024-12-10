import base64
from openai import OpenAI
import numpy as np
import numpy as np
from PIL import Image
import io
import os
import base64
from typing import Dict, Any
import logging
import traceback
import time
from dataclasses import dataclass

# Function to encode an image to base64 format


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to process and encode all images


def process_images():
    current_image = encode_image("images/current_camera_image.jpg")
    subgoal_image = encode_image("images/subgoal_camera_image.jpg")
    action_a_scene = encode_image("images/action_0_scene.jpg")
    action_b_scene = encode_image("images/action_1_scene.jpg")

    return current_image, subgoal_image, action_a_scene, action_b_scene

# Function to prepare and return OpenAI messages


def prepare_messages(task_instruction, current_image, subgoal_image, action_a_scene, action_b_scene, action0, action1):
    # Prepare the messages
    messages = [
        {
            "role": "system",
            "content": f"You are a judge of the planning performance of a robot manipulation system. The robot manipulation arm is trying to {task_instruction}."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Here is the current scene from the wrist-mounted camera:"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{current_image}"
                    }
                },
                {
                    "type": "text",
                    "text": "Here is the intermediate subgoal the robot arm is trying to reach:"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{subgoal_image}"
                    }
                },
                {
                    "type": "text",
                    "text": "Here are three scenes after applying different actions. Scene A:"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{action_a_scene}"
                    }
                },
                {
                    "type": "text",
                    "text": "Scene B:"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{action_b_scene}"
                    }
                },
                {
                    "type": "text",
                    "text": f"Info about the robot actions:\nAction A: {action0}\nAction B: {action1}\n Each robot action is a seven dimensions array for arm movement in the format of (delta x, delta y, delta z, delta roll, delta pitch, delta yaw, opening of the gripper). The position and rotation are relative changes to the end-effector pose, and the gripper dimension is an absolute control between 0 (open) and 1 (closed). You should provide explanations for all actions. \n Based on your knowledge of how humans would control the robot arm and your awareness of the situation, judge which one of the given actions is the best. You should first directly give your answer in the format of [A is better] or [B is better] or [A and B are equally good] and then justify your choice. Consider different factors, especially interactions with surrounding objects and human preferences."
                }
            ]
        }
    ]
    return messages


class RobotJudge:
    def __init__(self):
        # self.client = OpenAI(api_key="...")

    def judge(self, prompt, action0, action1):
        current_image, subgoal_image, action_a_scene, action_b_scene = process_images()
        messages = prepare_messages(prompt, current_image, subgoal_image,
                                    action_a_scene, action_b_scene, str(action0), str(action1))

        # Make the API call
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=10000
        )

        result = response.choices[0].message.content

        if "[A is better]" in result:
            return {"action": action0, "explanation": result}
        elif "[B is better]" in result:
            return {"action": action1, "explanation": result}
        else:
            return {"action": 0, "explanation": result}
