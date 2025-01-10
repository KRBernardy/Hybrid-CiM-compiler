#include <iostream>
#include <fstream>
#include "puma.h"
#include "conv.h"
#include "linear.h"

int main() {

    Model model = Model::create("LeNet_5");

    // Layer parameters
    unsigned int in_size_x = 32;
    unsigned int in_size_y = 32;
    unsigned int in_channels = 1;
    unsigned int kernel_size_x = 5;
    unsigned int kernel_size_y = 5;

    unsigned int in_channels_conv1 = in_channels;
    unsigned int out_channels_conv1 = 6;
    unsigned int in_channels_conv2 = out_channels_conv1;
    unsigned int out_channels_conv2 = 16;

    unsigned int in_size_fc1 = out_channels_conv2 * 5 * 5;
    unsigned int out_size_fc1 = 120;
    unsigned int in_size_fc2 = out_size_fc1;
    unsigned int out_size_fc2 = 84;
    unsigned int in_size_fc3 = out_size_fc2;
    unsigned int out_size_fc3 = 10;

    unsigned int out_size = out_size_fc3;

    unsigned int kernel_size_conv1[4] = {kernel_size_x, kernel_size_y, in_channels_conv1, out_channels_conv1};
    unsigned int kernel_size_conv2[4] = {kernel_size_x, kernel_size_y, in_channels_conv2, out_channels_conv2};


    // Input stream
    auto in_stream = InputImagePixelStream::create(model, "in_stream", in_size_x, in_size_y, in_channels);

    // Output stream
    auto out_vector = OutputVector::create(model, "out_stream", out_size);

    // Layer
    auto out_stream_conv1 = conv2d(model, "conv1", kernel_size_conv1, in_stream, 1, 1, 0, 0);
    auto out_stream_relu1 = relu(out_stream_conv1);
    auto out_avgpool1 = avgpool(out_stream_relu1, 2, 2);

    auto out_stream_conv2 = conv2d(model, "conv2", kernel_size_conv2, out_avgpool1, 1, 1, 0, 0);
    auto out_stream_relu2 = relu(out_stream_conv2);
    auto out_avgpool2 = avgpool(out_stream_relu2, 2, 2);

    auto out_flatten = flatten(out_avgpool2);

    auto out_fc1 = linear(model, "fc1", in_size_fc1, out_size_fc1, out_flatten);
    auto out_relu3 = relu(out_fc1);
    auto out_fc2 = linear(model, "fc2", in_size_fc2, out_size_fc2, out_relu3);
    auto out_relu4 = relu(out_fc2);
    auto out_fc3 = linear(model, "fc3", in_size_fc3, out_size_fc3, out_relu4);

    out_vector = out_fc3;

    // Compile
    model.compile();

    // Destroy model
    model.destroy();

    return 0;
}