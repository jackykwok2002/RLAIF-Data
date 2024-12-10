import pickle
import numpy as np
from PIL import Image
import pandas as pd
import openpyxl
import os
import shutil
from gaussian import MultivariateGaussianSampler
from judge import RobotJudge
from ap2p import ActionPix2Pix

def create_excel_file():
    """Create Excel file with appropriate headers if it doesn't exist"""
    if not os.path.exists('dataset_results.xlsx'):
        df = pd.DataFrame(columns=[
            'index',
            'instruction',
            'true_action',
            'pair_index',
            'action0',
            'action1',
            'chosen_action',
            'explanation'
        ])
        df.to_excel('dataset_results.xlsx', index=False)
    return pd.read_excel('dataset_results.xlsx')

def append_to_excel(data_dict):
    """Append a row of data to the Excel file"""
    df = pd.DataFrame([data_dict])
    if os.path.exists('dataset_results.xlsx'):
        existing_df = pd.read_excel('dataset_results.xlsx')
        updated_df = pd.concat([existing_df, df], ignore_index=True)
    else:
        updated_df = df
    updated_df.to_excel('dataset_results.xlsx', index=False)

def ensure_directory(directory):
    """Create directory if it doesn't exist"""
    os.makedirs(directory, exist_ok=True)

# Create main directories
ensure_directory("dataset")
ensure_directory("images")

# Create Excel file
create_excel_file()

with open("image_dict.pkl", "rb") as f:
    image_dict = pickle.load(f)

with open("instruction_dict.pkl", "rb") as f:
    instruction_dict = pickle.load(f)

with open("action_dict.pkl", "rb") as f:
    action_dict = pickle.load(f)

with open("subgoal_dict.pkl", "rb") as f:
    subgoal_dict = pickle.load(f)

# Initialize Global Sampler
# Define your means and variances and clipping values
means = np.array([0.00042126, 0.00000924, 0.00033116, 0.00020022, -0.00143980, 0.00021884, 0.63406380])
variances = np.array([0.00010339, 0.00024961, 0.00017812, 0.00062750, 0.00096115, 0.00291224, 0.22945852])
min_values = np.array([-0.07631552, -0.15209596, -0.15171302, -0.22191394, -0.34210532, -0.73485305, 0.00000000])
max_values = np.array([0.08137930, 0.14595977, 0.14885315, 0.22793450, 0.20718527, 0.78006949, 1.00000000])


# Initialize Judge
judge = RobotJudge()

# Initialize ActionPix2Pix
ap2p = ActionPix2Pix()

# main loop
# define number of datapoints
num_datapoints = 1000
stopped = 500

global_sampler = MultivariateGaussianSampler(means, variances, random_seed=stopped)

for index in range(stopped, num_datapoints):
    current_folder = os.path.join("dataset", str(index))
    ensure_directory(current_folder)

    input_image = image_dict[index]
    Image.fromarray(input_image).save("images/current_camera_image.jpg")
    Image.fromarray(input_image).save(os.path.join(current_folder, f"current_camera_image.jpg"))

    subgoal_image = subgoal_dict[index]
    Image.fromarray(subgoal_image).save("images/subgoal_camera_image.jpg")
    Image.fromarray(subgoal_image).save(os.path.join(current_folder, f"subgoal_camera_image.jpg"))

    instruction = instruction_dict[index]
    true_action = action_dict[index]

    # Create sampler instance with true action for this datapoint and variances from all bridge data
    true_action_sampler = MultivariateGaussianSampler(true_action, variances, random_seed=index)
    arr_new = np.vstack((true_action, true_action_sampler.generate_samples(5), global_sampler.generate_samples(5))) # append true_action, true action with noise, bridgedata with noise
    print(arr_new)

    arr_new[:, 6] = np.where(arr_new[:, 6] > 0.634, 1, 0)
    arr_new = np.clip(arr_new, min_values, max_values)
    print(arr_new)

    pairs = global_sampler.get_pairwise_comparisons(arr_new)

    # Process each pair
    for pair_idx, (action0, action1, _, _) in enumerate(pairs):
        if pair_idx < 10:
            data_dict = {
                'index': index,
                'instruction': instruction,
                'true_action': str(true_action.tolist()),
                'pair_index': pair_idx,
                'action0': str(action0.tolist()),
                'action1': str(action1.tolist()),
                'chosen_action': str(true_action.tolist()),
                'explanation': "N/A"
            }
            append_to_excel(data_dict)
            continue


        pair_folder = os.path.join(current_folder, f"pair_{pair_idx}")
        ensure_directory(pair_folder)
        
        # Generate and save images for both actions
        for i, action in enumerate([action0, action1]):
            generated_img = ap2p.generate_image(Image.fromarray(input_image), action)
            Image.fromarray(generated_img).save(
                os.path.join(pair_folder, f"pair_{pair_idx}_action_{i}_scene.jpg")
            )
            Image.fromarray(generated_img).save(f"images/action_{str(i)}_scene.jpg")
        
        # Get judgment
        result = judge.judge(
            prompt=instruction,
            action0=action0,
            action1=action1
        )
        
        # Save data to Excel
        data_dict = {
            'index': index,
            'instruction': instruction,
            'true_action': str(true_action.tolist()),
            'pair_index': pair_idx,
            'action0': str(action0.tolist()),
            'action1': str(action1.tolist()),
            'chosen_action': result['action'],
            'explanation': result['explanation']
        }
        append_to_excel(data_dict)
        
        print(f"Processed index {index}, pair {pair_idx}")
        print(f"Chosen action: {result['action']}")
        print(f"Explanation: {result['explanation']}")