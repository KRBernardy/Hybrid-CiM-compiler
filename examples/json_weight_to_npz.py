import os
import json
import numpy as np
import argparse
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def json_weight_to_npz(json_file_path, output_npz_path=None):
    """
    Convert weight data from JSON format to NPZ format.
    
    Args:
        json_file_path (str): Path to the JSON weight file
        output_npz_path (str, optional): Path where the NPZ file should be saved.
            If None, will use the same name as JSON file with .npz extension.
    
    Returns:
        str: Path to the saved NPZ file
    """
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"JSON file not found: {json_file_path}")
    
    # Determine output path
    if output_npz_path is None:
        output_npz_path = os.path.splitext(json_file_path)[0] + '.npz'
    
    logger.info(f"Converting {json_file_path} to {output_npz_path}")
    
    # Load JSON data
    with open(json_file_path, 'r') as f:
        weight_data = json.load(f)
    
    # Organize weights hierarchically
    weight_dict = {}
    
    for weight in weight_data:
        # Create a unique key for each weight array
        node_id = 0
        tile_id = weight.get('tile', 0)
        core_id = weight.get('core', 0)
        mvmu_id = weight.get('mvmu', 0)
        
        key = f"node{node_id}_tile{tile_id}_core{core_id}_mvmu{mvmu_id}"
        
        # Convert weight values to numpy array
        value = np.array(weight['value'], dtype=np.float32)
        
        # Do the reshaping
        side_length = int(np.sqrt(value.size))
        if side_length * side_length != value.size:
            raise ValueError(f"Weight array size {value.size} is not a perfect square.")
        weight_dict[key] = value.reshape((side_length, side_length))
    
    # Save as NPZ file
    np.savez(output_npz_path, **weight_dict)
    
    logger.info(f"Successfully converted weights to {output_npz_path}")
    return output_npz_path

def main():
    parser = argparse.ArgumentParser(description='Convert JSON weight files to NPZ format')
    parser.add_argument('json_file', help='Path to the JSON weight file')
    parser.add_argument('-o', '--output', help='Output NPZ file path (optional)')
    
    args = parser.parse_args()
    
    try:
        output_path = json_weight_to_npz(args.json_file, args.output)
        print(f"Weights saved to {output_path}")
    except Exception as e:
        logger.error(f"Error converting weights: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())