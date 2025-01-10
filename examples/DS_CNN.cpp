#include <iostream>
#include <fstream>
#include "puma.h"
#include "linear.h"
#include "conv.h"

int main() {

    Model model = Model::create("DS_CNN");

    // Layer parameters
    unsigned int in_size_x = 32;
    unsigned int in_size_y = 32;

    unsigned int in_channels = 3;
    unsigned int out_channels_conv1 = 32;
    unsigned int k_size_x = 3;
    unsigned int k_size_y = 3;
    unsigned int kernel_size_conv1[4] = {k_size_x, k_size_y, in_channels, out_channels_conv1};

    unsigned int in_channels_layer1 = out_channels_conv1;
    unsigned int out_channels_layer1 = 64;
    unsigned int kernel_size_layer1_conv1[4] = {k_size_x, k_size_y, in_channels_layer1, in_channels_layer1};
    unsigned int kernel_size_layer1_conv2[4] = {1, 1, in_channels_layer1, out_channels_layer1};

    unsigned int in_channels_layer2 = out_channels_layer1;
    unsigned int out_channels_layer2 = 128;
    unsigned int kernel_size_layer2_conv1[4] = {k_size_x, k_size_y, in_channels_layer2, in_channels_layer2};
    unsigned int kernel_size_layer2_conv2[4] = {1, 1, in_channels_layer2, out_channels_layer2};

    unsigned int in_channels_layer3 = out_channels_layer2;
    unsigned int out_channels_layer3 = 256;
    unsigned int kernel_size_layer3_conv1[4] = {k_size_x, k_size_y, in_channels_layer3, in_channels_layer3};
    unsigned int kernel_size_layer3_conv2[4] = {1, 1, in_channels_layer3, out_channels_layer3};

    unsigned int in_channels_layer4 = out_channels_layer3;
    unsigned int out_channels_layer4 = 512;
    unsigned int kernel_size_layer4_conv1[4] = {k_size_x, k_size_y, in_channels_layer4, in_channels_layer4};
    unsigned int kernel_size_layer4_conv2[4] = {1, 1, in_channels_layer4, out_channels_layer4};

    unsigned int in_size_fc1 = out_channels_layer4;
    unsigned int out_size_fc1 = 256;
    unsigned int in_size_fc2 = out_size_fc1;
    unsigned int out_size_fc2 = 10;

    // I/O stream
    auto in_stream = InputImagePixelStream::create(model, "in_stream", in_size_x, in_size_y, in_channels);
    auto out_vector = OutputVector::create(model, "out_vector", out_size_fc2);

    // Model definition
    auto out_conv1 = conv2d(model, "conv1", kernel_size_conv1, in_stream, 1, 1, 1, 1);

    auto out_layer1_conv1 = conv2d(model, "layer1_conv1", kernel_size_layer1_conv1, out_conv1, 1, 1, 1, 1);
    auto out_layer1_conv2 = conv2d(model, "layer1_conv2", kernel_size_layer1_conv2, out_layer1_conv1, 1, 1, 0, 0);

    auto out_layer2_conv1 = conv2d(model, "layer2_conv1", kernel_size_layer2_conv1, out_layer1_conv2, 1, 1, 1, 1);
    auto out_layer2_conv2 = conv2d(model, "layer2_conv2", kernel_size_layer2_conv2, out_layer2_conv1, 1, 1, 0, 0);

    auto out_layer3_conv1 = conv2d(model, "layer3_conv1", kernel_size_layer3_conv1, out_layer2_conv2, 1, 1, 1, 1);
    auto out_layer3_conv2 = conv2d(model, "layer3_conv2", kernel_size_layer3_conv2, out_layer3_conv1, 1, 1, 0, 0);

    auto out_layer4_conv1 = conv2d(model, "layer4_conv1", kernel_size_layer4_conv1, out_layer3_conv2, 1, 1, 1, 1);
    auto out_layer4_conv2 = conv2d(model, "layer4_conv2", kernel_size_layer4_conv2, out_layer4_conv1, 1, 1, 0, 0);

    auto out_avg_pool = avgpool(out_layer4_conv2, in_size_x, in_size_y);

    auto out_flatten = flatten(out_avg_pool);

    auto out_fc1 = linear(model, "fc_layer", in_size_fc1, out_size_fc1, out_flatten);
    auto out_fc2 = linear(model, "fc_layer", in_size_fc2, out_size_fc2, out_fc1);

    out_vector = out_fc2;

    // Compile the model
    model.compile();

    // Destroy the model
    model.destroy();

    return 0;
}