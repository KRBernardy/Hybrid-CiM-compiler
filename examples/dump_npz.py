import numpy as np
import os
import sys

if len(sys.argv) != 2:
    print("Usage: python dump_npz.py <npz_file>")
    sys.exit(1)

npz_file = sys.argv[1]
data = np.load(npz_file)

# Create directory based on npz name
if 'resnet20' in npz_file.lower():
    dir_name = 'ResNet_20_weights'
else:
    dir_name = os.path.splitext(npz_file)[0] + '_weights'
os.makedirs(dir_name, exist_ok=True)

def write_array(filename, array, as_int=False):
    with open(os.path.join(dir_name, filename), 'w') as f:
        for val in array:
            if as_int:
                f.write(f"{int(val)}\n")
            else:
                f.write(f"{val}\n")

# Find all layers
conv_layers = []
bn_layers = []
linear_found = False

for key in data.keys():
    if key.endswith('.weight_quantized'):
        layer = key[:-len('.weight_quantized')]
        if layer == 'linear':
            linear_found = True
        else:
            conv_layers.append(layer)
    elif key.endswith('.bn_weight'):
        layer = key[:-len('.bn_weight')]
        bn_layers.append(layer)

# Sort for consistency
conv_layers.sort()
bn_layers.sort()

print(f"Found conv layers: {conv_layers}")
print(f"Found bn layers: {bn_layers}")
print(f"Linear found: {linear_found}")

# Write conv weights as ints
for layer in conv_layers:
    weights = data[layer + '.weight_quantized']
    write_array(f"{layer}_weights.txt", weights, as_int=True)

# Write bn params as floats
for layer in bn_layers:
    gamma = data[layer + '.bn_weight']
    beta = data[layer + '.bn_bias']
    running_mean = data[layer + '.bn_running_mean']
    running_var = data[layer + '.bn_running_var']
    epsilon = data.get(layer + '.bn_eps', 1e-5)

    scale = gamma / np.sqrt(running_var + epsilon)
    shift = beta - running_mean * scale

    write_array(f"{layer}_scale.txt", scale)
    write_array(f"{layer}_shift.txt", shift)

# Write linear
if linear_found:
    weights = data['linear.weight_quantized']
    biases = data['linear.bias']
    write_array("linear_weights.txt", weights, as_int=True)
    write_array("linear_biases.txt", biases)

print(f"All txt files written to {dir_name}/")