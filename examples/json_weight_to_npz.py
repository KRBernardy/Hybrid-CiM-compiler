import os
import json
import numpy as np
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def json_weight_to_npz(json_file_path, output_npz_path=None):
    """
    Convert constant matrix weights and vectors from JSON format to an NPZ archive.

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

    weight_dict = {}
    vector_dict = {}

    for record in weight_data:
        node_id = 0  # Currently only a single node is emitted.
        tile_id = record.get('tile')
        core_id = record.get('core')

        if tile_id is None or core_id is None:
            raise ValueError("JSON record missing required 'tile' or 'core' fields.")

        raw_values = record.get('value', [])
        if raw_values is None:
            raise ValueError("JSON record missing 'value' field.")

        if 'mvmu' in record:
            mvmu_id = record['mvmu']
            key = f"weight_node{node_id}_tile{tile_id}_core{core_id}_mvmu{mvmu_id}"

            value = np.asarray(raw_values, dtype=np.uint32)

            side_length = int(np.sqrt(value.size))
            if side_length * side_length != value.size:
                raise ValueError(
                    f"Weight array for {key} has size {value.size}, which is not a perfect square."
                )

            weight_dict[key] = value.reshape((side_length, side_length))
        elif 'reg' in record:
            reg_addr = record['reg']
            key = f"vector_node{node_id}_tile{tile_id}_core{core_id}_reg{reg_addr}"

            value = np.asarray(raw_values, dtype=np.float32)

            if value.size == 0:
                logger.warning("Vector %s has no entries; storing as empty array.", key)

            vector_dict[key] = value
        else:
            raise ValueError("JSON record is neither matrix weight nor constant vector (missing 'mvmu' or 'reg').")

    combined = {}
    combined.update(weight_dict)
    combined.update(vector_dict)

    if not combined:
        raise ValueError("No weights or vectors were found in the provided JSON file.")

    np.savez(output_npz_path, **combined)

    logger.info(
        "Converted %d weight tiles and %d constant vectors to %s",
        len(weight_dict),
        len(vector_dict),
        output_npz_path,
    )
    
    logger.info(f"Successfully converted weights to {output_npz_path}")
    return output_npz_path

def main():
    parser = argparse.ArgumentParser(description='Convert JSON weight files to NPZ format')
    parser.add_argument('json_file', help='Path to the JSON weight file')
    parser.add_argument('-o', '--output', help='Output NPZ file path (optional)')
    
    args = parser.parse_args()
    
    try:
        output_path = json_weight_to_npz(args.json_file, args.output)
        print(f"Weights and vectors saved to {output_path}")
    except Exception as e:
        logger.error(f"Error converting weights: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())