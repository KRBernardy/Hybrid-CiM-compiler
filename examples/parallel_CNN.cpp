#include <iostream>
#include <fstream>
#include <vector>
#include "puma.h"
#include "linear.h"
#include "conv.h"

int main() {

    Model model = Model::create("parallel_CNN");

    // Layer parameters
    unsigned int in_size_x = 32;
    unsigned int in_size_y = 32;

    unsigned int in_channels = 3;
    unsigned int out_channels_conv = 32;
    unsigned int pooling_size = 8;
    
    unsigned int kernel_size_path1_conv1[4] = {1, 1, in_channels, out_channels_conv};
    unsigned int kernel_size_path1_conv2[4] = {1, 1, out_channels_conv, out_channels_conv};
    unsigned int kernel_size_path2_conv1[4] = {3, 3, in_channels, out_channels_conv};
    unsigned int kernel_size_path2_conv2[4] = {3, 3, out_channels_conv, out_channels_conv};
    unsigned int kernel_size_path3_conv1[4] = {5, 5, in_channels, out_channels_conv};
    unsigned int kernel_size_path3_conv2[4] = {5, 5, out_channels_conv, out_channels_conv};

    unsigned int in_size_fc1 = in_size_x / pooling_size * in_size_y / pooling_size * out_channels_conv * 3; // 3 paths
    unsigned int out_size_fc1 = 512;
    unsigned int in_size_fc2 = out_size_fc1;
    unsigned int out_size_fc2 = 100;

    // I/O stream
    auto in_stream = InputImagePixelStream::create(model, "in_stream", in_size_x, in_size_y, in_channels);
    auto out_vector = OutputVector::create(model, "out_vector", out_size_fc2);

    // Model definition
    auto out_path1_conv1 = conv2d(model, "path1_conv1", kernel_size_path1_conv1, in_stream, 1, 1, 0, 0);
    auto out_path1_conv2 = conv2d(model, "path1_conv2", kernel_size_path1_conv2, out_path1_conv1, 1, 1, 0, 0);
    auto out_path1_pooling = avgpool(out_path1_conv2, pooling_size, pooling_size);

    auto out_path2_conv1 = conv2d(model, "path2_conv1", kernel_size_path2_conv1, in_stream, 1, 1, 1, 1);
    auto out_path2_conv2 = conv2d(model, "path2_conv2", kernel_size_path2_conv2, out_path2_conv1, 1, 1, 1, 1);
    auto out_path2_pooling = avgpool(out_path2_conv2, pooling_size, pooling_size);

    auto out_path3_conv1 = conv2d(model, "path3_conv1", kernel_size_path3_conv1, in_stream, 1, 1, 2, 2);
    auto out_path3_conv2 = conv2d(model, "path3_conv2", kernel_size_path3_conv2, out_path3_conv1, 1, 1, 2, 2);
    auto out_path3_pooling = avgpool(out_path3_conv2, pooling_size, pooling_size);

    std::vector<ImagePixelStream> list = {out_path1_pooling, out_path2_pooling, out_path3_pooling};

    auto out_conv = merge(list);

    auto out_flatten = flatten(out_conv);

    auto out_fc1 = linear(model, "fc_layer_1", in_size_fc1, out_size_fc1, out_flatten);
    auto out_fc2 = linear(model, "fc_layer_2", in_size_fc2, out_size_fc2, out_fc1);

    out_vector = out_fc2;

    // Compile the model
    model.compile();

    // Destroy the model
    model.destroy();

    return 0;
}