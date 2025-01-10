#include <iostream>
#include <fstream>
#include "puma.h"
#include "linear.h"
#include "conv.h"

ImagePixelStream convRelu(Model model, std::string name, unsigned int kernel_size[4], ImagePixelStream in_stream, unsigned int stride, unsigned int groups = 1) {
    unsigned int padding = (kernel_size[0] - 1) / 2;
    auto out_conv = conv2d(model, name + "_conv", kernel_size, in_stream, stride, stride, padding, padding);
    auto out_conv_activated = relu(out_conv);
    return out_conv_activated;
}

// Needs group convolution, which is not supported by the current version of Hybrid-CiM-compiler
ImagePixelStream invertedResidual(Model model, std::string name, ImagePixelStream in_stream, unsigned int in_channels, unsigned int out_channels, unsigned int stride, unsigned int expand_ratio) {
    unsigned int hidden_dim = in_channels * expand_ratio;
    ImagePixelStream out;
    if (expand_ratio != 1) {
        auto out_expand = convRelu(model, name + "_expand", {1, 1, in_channels, hidden_dim}, in_stream, 1);
        auto out_depthwise = convRelu(model, name + "_depthwise", {3, 3, hidden_dim, hidden_dim}, out_expand, stride, hidden_dim);
        out = conv2d(model, name + "_project", {1, 1, hidden_dim, out_channels}, out_depthwise, 1, 1, 0, 0);
    } else {
        auto out_depthwise = convRelu(model, name + "_depthwise", {3, 3, hidden_dim, hidden_dim}, in_stream, stride, hidden_dim);
        out = conv2d(model, name + "_project", {1, 1, hidden_dim, out_channels}, out_depthwise, 1, 1, 0, 0);
    }
    if (stride == 1 && in_channels == out_channels) {
        out = out + in_stream;
    }
    return out;
}

int main() {

    Model model = Model::create("MobileNet_v2");

    // Layer parameters
    unsigned int in_size_x = 32;
    unsigned int in_size_y = 32;

    unsigned int in_channels = 3;
    unsigned int out_channels_conv1 = 16;
    unsigned int k_size_x = 3;
    unsigned int k_size_y = 3;
    unsigned int kernel_size_conv1[4] = {k_size_x, k_size_y, in_channels, out_channels_conv1};

    unsigned int in_channels_layer1 = out_channels_conv1;
    unsigned int out_channels_layer1 = 16;
    unsigned int kernel_size_layer1[4] = {k_size_x, k_size_y, in_channels_layer1, out_channels_layer1};

    unsigned int in_channels_layer2 = out_channels_layer1;
    unsigned int out_channels_layer2 = 32;
    unsigned int kernel_size_layer2_block1_conv1[4] = {k_size_x, k_size_y, in_channels_layer2, out_channels_layer2};
    unsigned int kernel_size_layer2_side[4] = {1, 1, in_channels_layer2, out_channels_layer2};
    unsigned int kernel_size_layer2[4] = {k_size_x, k_size_y, out_channels_layer2, out_channels_layer2};

    unsigned int in_channels_layer3 = out_channels_layer2;
    unsigned int out_channels_layer3 = 64;
    unsigned int kernel_size_layer3_block1_conv1[4] = {k_size_x, k_size_y, in_channels_layer3, out_channels_layer3};
    unsigned int kernel_size_layer3_side[4] = {1, 1, in_channels_layer3, out_channels_layer3};
    unsigned int kernel_size_layer3[4] = {k_size_x, k_size_y, out_channels_layer3, out_channels_layer3};

    unsigned int in_size_fc = out_channels_layer3;
    unsigned int out_size_fc = 10;

    // I/O stream
    auto in_stream = InputImagePixelStream::create(model, "in_stream", in_size_x, in_size_y, in_channels);
    auto out_vector = OutputVector::create(model, "out_vector", out_size_fc);

    // Model definition
    // First convolutional layer
    auto out_conv1 = conv2d(model, "conv1", kernel_size_conv1, in_stream, 1, 1, 1, 1);
    auto out_conv1_activated = relu(out_conv1);

    // Residual block 1
    auto out_layer1_block1_conv1 = conv2d(model, "layer1_block1_conv1", kernel_size_layer1, out_conv1_activated, 1, 1, 1, 1);
    auto out_layer1_block1_conv1_activated = relu(out_layer1_block1_conv1);
    auto out_layer1_block1_conv2 = conv2d(model, "layer1_block1_conv2", kernel_size_layer1, out_layer1_block1_conv1_activated, 1, 1, 1, 1);
    auto out_layer1_block1 = relu(out_layer1_block1_conv2);
    auto out_layer1_block2_conv1 = conv2d(model, "layer1_block2_conv1", kernel_size_layer1, out_layer1_block1, 1, 1, 1, 1);
    auto out_layer1_block2_conv1_activated = relu(out_layer1_block2_conv1);
    auto out_layer1_block2_conv2 = conv2d(model, "layer1_block2_conv2", kernel_size_layer1, out_layer1_block2_conv1_activated, 1, 1, 1, 1);
    auto out_layer1_block2 = relu(out_layer1_block2_conv2);
    auto out_layer1_block3_conv1 = conv2d(model, "layer1_block3_conv1", kernel_size_layer1, out_layer1_block2, 1, 1, 1, 1);
    auto out_layer1_block3_conv1_activated = relu(out_layer1_block3_conv1);
    auto out_layer1_block3_conv2 = conv2d(model, "layer1_block3_conv2", kernel_size_layer1, out_layer1_block3_conv1_activated, 1, 1, 1, 1);
    auto out_layer1 = relu(out_layer1_block3_conv2);

    // Residual block 2
    auto out_layer2_block1_conv1 = conv2d(model, "layer2_block1_conv1", kernel_size_layer2_block1_conv1, out_layer1, 2, 2, 1, 1);
    auto out_layer2_block1_conv1_activated = relu(out_layer2_block1_conv1);
    auto out_layer2_block1_conv2 = conv2d(model, "layer2_block1_conv2", kernel_size_layer2, out_layer2_block1_conv1_activated, 1, 1, 1, 1);
    auto out_layer2_block1_side = conv2d(model, "layer2_block1_side", kernel_size_layer2_side, out_layer1, 2, 2, 0, 0);
    auto out_layer2_block1 = relu(out_layer2_block1_conv2 + out_layer2_block1_side);
    auto out_layer2_block2_conv1 = conv2d(model, "layer2_block2_conv1", kernel_size_layer2, out_layer2_block1, 1, 1, 1, 1);
    auto out_layer2_block2_conv1_activated = relu(out_layer2_block2_conv1);
    auto out_layer2_block2_conv2 = conv2d(model, "layer2_block2_conv2", kernel_size_layer2, out_layer2_block2_conv1_activated, 1, 1, 1, 1);
    auto out_layer2_block2 = relu(out_layer2_block2_conv2);
    auto out_layer2_block3_conv1 = conv2d(model, "layer2_block3_conv1", kernel_size_layer2, out_layer2_block2, 1, 1, 1, 1);
    auto out_layer2_block3_conv1_activated = relu(out_layer2_block3_conv1);
    auto out_layer2_block3_conv2 = conv2d(model, "layer2_block3_conv2", kernel_size_layer2, out_layer2_block3_conv1_activated, 1, 1, 1, 1);
    auto out_layer2 = relu(out_layer2_block3_conv2);

    // Residual block 3
    auto out_layer3_block1_conv1 = conv2d(model, "layer3_block1_conv1", kernel_size_layer3_block1_conv1, out_layer2, 2, 2, 1, 1);
    auto out_layer3_block1_conv1_activated = relu(out_layer3_block1_conv1);
    auto out_layer3_block1_conv2 = conv2d(model, "layer3_block1_conv2", kernel_size_layer3, out_layer3_block1_conv1_activated, 1, 1, 1, 1);
    auto out_layer3_block1_side = conv2d(model, "layer3_block1_side", kernel_size_layer3_side, out_layer2, 2, 2, 0, 0);
    auto out_layer3_block1 = relu(out_layer3_block1_conv2 + out_layer3_block1_side);
    auto out_layer3_block2_conv1 = conv2d(model, "layer3_block2_conv1", kernel_size_layer3, out_layer3_block1, 1, 1, 1, 1);
    auto out_layer3_block2_conv1_activated = relu(out_layer3_block2_conv1);
    auto out_layer3_block2_conv2 = conv2d(model, "layer3_block2_conv2", kernel_size_layer3, out_layer3_block2_conv1_activated, 1, 1, 1, 1);
    auto out_layer3_block2 = relu(out_layer3_block2_conv2);
    auto out_layer3_block3_conv1 = conv2d(model, "layer3_block3_conv1", kernel_size_layer3, out_layer3_block2, 1, 1, 1, 1);
    auto out_layer3_block3_conv1_activated = relu(out_layer3_block3_conv1);
    auto out_layer3_block3_conv2 = conv2d(model, "layer3_block3_conv2", kernel_size_layer3, out_layer3_block3_conv1_activated, 1, 1, 1, 1);
    auto out_layer3 = relu(out_layer3_block3_conv2);

    // Average pooling
     auto out_avg_pool = maxpool(out_layer3, 8, 8);

    // Flatten
    auto out_flatten = flatten(out_avg_pool);

    // Fully connected layer
    out_vector = linear(model, "fc_layer", in_size_fc, out_size_fc, out_flatten);

    // Compile the model
    model.compile();

    // Destroy the model
    model.destroy();

    return 0;
}