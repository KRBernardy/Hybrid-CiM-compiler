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
        flat = np.asarray(array).reshape(-1)
        for val in flat:
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

# Helper to pretty-print arrays concisely
def _format_array(arr):
    a = np.asarray(arr)
    # Print using maximum practical precision for floats so we don't truncate values.
    # Use floatmode='maxprec' to let numpy decide how many digits are required to
    # uniquely represent the floating values.
    try:
        return np.array2string(a, precision=17, separator=', ', threshold=10000, floatmode='maxprec')
    except Exception:
        return repr(a)


def print_layer_scales(layer_name):
    """Find and print any weight/activation scale entries for a given layer.

    Will match keys that start with the layer name and contain either
    'weight_scale' or 'act_scale' to be tolerant of small naming variants.
    """
    matches = [k for k in data.keys() if k.startswith(layer_name) and ('weight_scale' in k or 'act_scale' in k)]
    if not matches:
        print(f"{layer_name}: no scale entries found")
        return

    print(f"Scales for layer '{layer_name}':")
    for k in sorted(matches):
        v = data[k]
        print(f"  {k}: shape={np.shape(v)} value={_format_array(v)}")


# Print scales for each conv layer
for layer in conv_layers:
    print_layer_scales(layer)

# Print scales for linear if present
if linear_found:
    print_layer_scales('linear')

# Write conv weights as ints
for layer in conv_layers:
    weights = data[layer + '.weight_quantized']
    shape = data[layer + '.weight_shape']
    if weights.size != int(np.prod(shape)):
        raise ValueError(
            f"{layer}: weight tensor size {weights.size} does not match expected shape product {int(np.prod(shape))}"
        )
    reshaped = weights.reshape(shape)
    reordered = np.transpose(reshaped, (2, 3, 0, 1))  # ky, kx, out, in
    write_array(f"{layer}_weights.txt", reordered, as_int=True)

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

"""
Conv Weight Layout

In operations.cpp (~467-520) each convolution step grabs mat = M->getTile(kh, kw, h, w) and multiplies it with imageStream->get(hi, wi). Rows of that tile map to output channels, columns map to input channels for a specific kernel (kh, kw).
instance.cpp (~137-174) shows how the runtime expects the raw buffer: it indexes weights as ((kh * kernelWidth + kw) * nOutChannels + oc) * nInChannels + ic, and writes them row-major inside each MVMU tile. That means the fastest-changing index is the input channel, then output channel, then kernel-x, then kernel-y.
So the native layout here is [kernel_y][kernel_x][out_channel][in_channel], not PyTorch’s default [out][in][kernel_y][kernel_x]. Feeding PyTorch’s flattened tensor directly will mix kw with ic, giving wrong results.
To hand weights over, reorder once with something like weight.permute(2, 3, 0, 1).contiguous().view(-1) (and pad per MVMU_DIM if needed). After that, the layout matches what generateData() consumes.
Next step: update your data pipeline to emit conv weights in that [k_y, k_x, out, in] order before binding them.
"""