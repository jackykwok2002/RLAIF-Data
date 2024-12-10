import numpy as np
import sglang as sgl
from token2action import TokenToAction
import pandas as pd
import json
import os
import glob
import time
import pickle
from tqdm import tqdm
import openpyxl

converter = TokenToAction()

@sgl.function
def image_qa(s, image_path, question):
    s += sgl.image(image_path) + question
    s += sgl.gen("action")

def get_unique_actions(generated_actions, num_unique=20):
    """
    Extract unique actions from a list of generated actions.
    
    Args:
        generated_actions (list): List of generated action tokens/ids
        num_unique (int): Number of unique actions to return (default: 20)
        
    Returns:
        list: List of unique converted actions, limited to num_unique
    """
    # Convert actions to string representations
    converted_actions = [converter.convert(action) for action in generated_actions]
    
    # Use a dictionary to maintain order and track unique string representations
    unique_dict = {}
    for i, action_str in enumerate(converted_actions):
        action_str_key = str(action_str)  # Convert to string for hashing
        if action_str_key not in unique_dict:
            unique_dict[action_str_key] = generated_actions[i]
            
            # Break if we have enough unique actions
            if len(unique_dict) >= num_unique:
                break
    
    # Return the unique generated actions
    return list(unique_dict.values())[:num_unique]

def batch(image, instruction, batch_size, temp):
    arguments = [
        {
            "image_path": image,
            "question": f"In: What action should the robot take to {instruction}?\nOut:",
        }
    ] * batch_size
    states = image_qa.run_batch(
        arguments,
        max_new_tokens=7,
        temperature=temp
    )
    return [s.get_meta_info("action")["output_ids"] for s in states]

def append_to_excel(data_dict):
    """Append a row of data to the Excel file"""
    df = pd.DataFrame([data_dict])
    if os.path.exists('dataset_results.xlsx'):
        existing_df = pd.read_excel('dataset_results.xlsx')
        updated_df = pd.concat([existing_df, df], ignore_index=True)
    else:
        updated_df = df
    updated_df.to_excel('dataset_results.xlsx', index=False)

def run_experiment(dataset, image_num=10000, num_samples=20, temperature=2.0):
    # Set up runtime with 4-bit quantization
    runtime = sgl.Runtime(
        model_path="openvla/openvla-7b",
        tokenizer_path="openvla/openvla-7b",
        disable_cuda_graph=True,
        disable_radix_cache=True,
        chunked_prefill_size=-1,
        torchao_config="int4wo-128"
    )
    sgl.set_default_backend(runtime)
    
    print(f"=== Dataset: {dataset}, Samples per image: {num_samples}, Temperature: {temperature} ===")
    
    images = glob.glob(f"images/*.png")
    baseline_actions = pickle.load(open(f"data/{dataset}_pkl/action_dict.pkl", "rb"))
    instructions = pickle.load(open(f"data/{dataset}_pkl/instruction_dict.pkl", "rb"))
    
    for idx, image in tqdm(enumerate(images[:image_num])):
        # Get multiple samples for each image
        generated_actions = batch(image, instructions[idx], num_samples, temp=temperature)
        unique_actions = get_unique_actions(generated_actions)

        # Create multiple data points for each generated action
        for pair_idx, generated_action in enumerate(unique_actions):
            data_dict = {
                'index': idx,
                'instruction': instructions[idx],
                'true_action': baseline_actions[idx],
                'pair_index': pair_idx,
                'action0': baseline_actions[idx],
                'action1': converter.convert(generated_action),
                'chosen_action': baseline_actions[idx]
            }
            append_to_excel(data_dict)
            
    runtime.shutdown()

if __name__ == '__main__':
    # Run for the 'ind' dataset
    run_experiment(
        dataset="ind",
        image_num=10,
        num_samples=100,
        temperature=2.0
    )