#include <iostream>
#include <fstream>
#include "puma.h"
#include "linear.h"
#include "conv.h"

int main() {

    Model model = Model::create("ResNet_20");

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
    unsigned int out_size_fc = 100;

    // I/O stream
    auto in_stream = InputImagePixelStream::create(model, "in_stream", in_size_x, in_size_y, in_channels);
    auto out_vector = OutputVector::create(model, "out_vector", out_size_fc);

    // Model definition
    // First convolutional layer
    auto out_conv1 = conv2d(model, "conv1", kernel_size_conv1, in_stream, 1, 1, 1, 1);
    auto out_conv1_bn = batchnorm2d(model, "bn1", out_conv1, out_channels_conv1);
    auto out_conv1_activated = relu(out_conv1_bn);

    // Residual block 1
    auto out_layer1_block1_conv1 = conv2d(model, "layer1.0.conv1", kernel_size_layer1, out_conv1_activated, 1, 1, 1, 1);
    auto out_layer1_block1_conv1_bn = batchnorm2d(model, "layer1.0.bn1", out_layer1_block1_conv1, out_channels_layer1);
    auto out_layer1_block1_conv1_activated = relu(out_layer1_block1_conv1_bn);
    auto out_layer1_block1_conv2 = conv2d(model, "layer1.0.conv2", kernel_size_layer1, out_layer1_block1_conv1_activated, 1, 1, 1, 1);
    auto out_layer1_block1_conv2_bn = batchnorm2d(model, "layer1.0.bn2", out_layer1_block1_conv2, out_channels_layer1);
    auto out_layer1_block1 = relu(out_layer1_block1_conv2_bn);
    auto out_layer1_block2_conv1 = conv2d(model, "layer1.1.conv1", kernel_size_layer1, out_layer1_block1, 1, 1, 1, 1);
    auto out_layer1_block2_conv1_bn = batchnorm2d(model, "layer1.1.bn1", out_layer1_block2_conv1, out_channels_layer1);
    auto out_layer1_block2_conv1_activated = relu(out_layer1_block2_conv1_bn);
    auto out_layer1_block2_conv2 = conv2d(model, "layer1.1.conv2", kernel_size_layer1, out_layer1_block2_conv1_activated, 1, 1, 1, 1);
    auto out_layer1_block2_conv2_bn = batchnorm2d(model, "layer1.1.bn2", out_layer1_block2_conv2, out_channels_layer1);
    auto out_layer1_block2 = relu(out_layer1_block2_conv2_bn);
    auto out_layer1_block3_conv1 = conv2d(model, "layer1.2.conv1", kernel_size_layer1, out_layer1_block2, 1, 1, 1, 1);
    auto out_layer1_block3_conv1_bn = batchnorm2d(model, "layer1.2.bn1", out_layer1_block3_conv1, out_channels_layer1);
    auto out_layer1_block3_conv1_activated = relu(out_layer1_block3_conv1_bn);
    auto out_layer1_block3_conv2 = conv2d(model, "layer1.2.conv2", kernel_size_layer1, out_layer1_block3_conv1_activated, 1, 1, 1, 1);
    auto out_layer1_block3_conv2_bn = batchnorm2d(model, "layer1.2.bn2", out_layer1_block3_conv2, out_channels_layer1);
    auto out_layer1 = relu(out_layer1_block3_conv2_bn);

    // Residual block 2
    auto out_layer2_block1_conv1 = conv2d(model, "layer2.0.conv1", kernel_size_layer2_block1_conv1, out_layer1, 2, 2, 1, 1);
    auto out_layer2_block1_conv1_bn = batchnorm2d(model, "layer2.0.bn1", out_layer2_block1_conv1, out_channels_layer2);
    auto out_layer2_block1_conv1_activated = relu(out_layer2_block1_conv1_bn);
    auto out_layer2_block1_conv2 = conv2d(model, "layer2.0.conv2", kernel_size_layer2, out_layer2_block1_conv1_activated, 1, 1, 1, 1);
    auto out_layer2_block1_conv2_bn = batchnorm2d(model, "layer2.0.bn2", out_layer2_block1_conv2, out_channels_layer2);
    auto out_layer2_block1_side = conv2d(model, "layer2.0.shortcut.0", kernel_size_layer2_side, out_layer1, 2, 2, 0, 0);
    auto out_layer2_block1_side_bn = batchnorm2d(model, "layer2.0.shortcut.1", out_layer2_block1_side, out_channels_layer2);
    auto out_layer2_block1 = relu(out_layer2_block1_conv2_bn + out_layer2_block1_side_bn);
    auto out_layer2_block2_conv1 = conv2d(model, "layer2.1.conv1", kernel_size_layer2, out_layer2_block1, 1, 1, 1, 1);
    auto out_layer2_block2_conv1_bn = batchnorm2d(model, "layer2.1.bn1", out_layer2_block2_conv1, out_channels_layer2);
    auto out_layer2_block2_conv1_activated = relu(out_layer2_block2_conv1_bn);
    auto out_layer2_block2_conv2 = conv2d(model, "layer2.1.conv2", kernel_size_layer2, out_layer2_block2_conv1_activated, 1, 1, 1, 1);
    auto out_layer2_block2_conv2_bn = batchnorm2d(model, "layer2.1.bn2", out_layer2_block2_conv2, out_channels_layer2);
    auto out_layer2_block2 = relu(out_layer2_block2_conv2_bn);
    auto out_layer2_block3_conv1 = conv2d(model, "layer2.2.conv1", kernel_size_layer2, out_layer2_block2, 1, 1, 1, 1);
    auto out_layer2_block3_conv1_bn = batchnorm2d(model, "layer2.2.bn1", out_layer2_block3_conv1, out_channels_layer2);
    auto out_layer2_block3_conv1_activated = relu(out_layer2_block3_conv1_bn);
    auto out_layer2_block3_conv2 = conv2d(model, "layer2.2.conv2", kernel_size_layer2, out_layer2_block3_conv1_activated, 1, 1, 1, 1);
    auto out_layer2_block3_conv2_bn = batchnorm2d(model, "layer2.2.bn2", out_layer2_block3_conv2, out_channels_layer2);
    auto out_layer2 = relu(out_layer2_block3_conv2_bn);

    // Residual block 3
    auto out_layer3_block1_conv1 = conv2d(model, "layer3.0.conv1", kernel_size_layer3_block1_conv1, out_layer2, 2, 2, 1, 1);
    auto out_layer3_block1_conv1_bn = batchnorm2d(model, "layer3.0.bn1", out_layer3_block1_conv1, out_channels_layer3);
    auto out_layer3_block1_conv1_activated = relu(out_layer3_block1_conv1_bn);
    auto out_layer3_block1_conv2 = conv2d(model, "layer3.0.conv2", kernel_size_layer3, out_layer3_block1_conv1_activated, 1, 1, 1, 1);
    auto out_layer3_block1_conv2_bn = batchnorm2d(model, "layer3.0.bn2", out_layer3_block1_conv2, out_channels_layer3);
    auto out_layer3_block1_side = conv2d(model, "layer3.0.shortcut.0", kernel_size_layer3_side, out_layer2, 2, 2, 0, 0);
    auto out_layer3_block1_side_bn = batchnorm2d(model, "layer3.0.shortcut.1", out_layer3_block1_side, out_channels_layer3);
    auto out_layer3_block1 = relu(out_layer3_block1_conv2_bn + out_layer3_block1_side_bn);
    auto out_layer3_block2_conv1 = conv2d(model, "layer3.1.conv1", kernel_size_layer3, out_layer3_block1, 1, 1, 1, 1);
    auto out_layer3_block2_conv1_bn = batchnorm2d(model, "layer3.1.bn1", out_layer3_block2_conv1, out_channels_layer3);
    auto out_layer3_block2_conv1_activated = relu(out_layer3_block2_conv1_bn);
    auto out_layer3_block2_conv2 = conv2d(model, "layer3.1.conv2", kernel_size_layer3, out_layer3_block2_conv1_activated, 1, 1, 1, 1);
    auto out_layer3_block2_conv2_bn = batchnorm2d(model, "layer3.1.bn2", out_layer3_block2_conv2, out_channels_layer3);
    auto out_layer3_block2 = relu(out_layer3_block2_conv2_bn);
    auto out_layer3_block3_conv1 = conv2d(model, "layer3.2.conv1", kernel_size_layer3, out_layer3_block2, 1, 1, 1, 1);
    auto out_layer3_block3_conv1_bn = batchnorm2d(model, "layer3.2.bn1", out_layer3_block3_conv1, out_channels_layer3);
    auto out_layer3_block3_conv1_activated = relu(out_layer3_block3_conv1_bn);
    auto out_layer3_block3_conv2 = conv2d(model, "layer3.2.conv2", kernel_size_layer3, out_layer3_block3_conv1_activated, 1, 1, 1, 1);
    auto out_layer3_block3_conv2_bn = batchnorm2d(model, "layer3.2.bn2", out_layer3_block3_conv2, out_channels_layer3);
    auto out_layer3 = relu(out_layer3_block3_conv2_bn);

    // Average pooling
    auto out_avg_pool = maxpool(out_layer3, 8, 8);

    // Flatten
    auto out_flatten = flatten(out_avg_pool);

    // Fully connected layer
    out_vector = linear(model, "linear", in_size_fc, out_size_fc, out_flatten);

    // Compile the model
    model.compile();

    // Bind data
    ModelInstance modelInstance = ModelInstance::create(model);

    unsigned int* conv1Weights = new unsigned int[k_size_x * k_size_y * in_channels * out_channels_conv1];
    std::ifstream wf_conv1;
    wf_conv1.open("ResNet_20_weights/conv1_weights.txt");
    int i = 0;
    while (wf_conv1 >> conv1Weights[i]) {
        i++;
    }
    wf_conv1.close();
    modelInstance.bind("conv1_weights", conv1Weights);
    delete[] conv1Weights;

    float* bn1Scales = new float[out_channels_conv1];
    std::ifstream wf_bn1_scale;
    wf_bn1_scale.open("ResNet_20_weights/bn1_scale.txt");
    i = 0;
    while (wf_bn1_scale >> bn1Scales[i]) {
        i++;
    }
    wf_bn1_scale.close();
    modelInstance.bind("bn1_param_scales", bn1Scales);
    delete[] bn1Scales;

    float* bn1Shifts = new float[out_channels_conv1];
    std::ifstream wf_bn1_shift;
    wf_bn1_shift.open("ResNet_20_weights/bn1_shift.txt");
    i = 0;
    while (wf_bn1_shift >> bn1Shifts[i]) {
        i++;
    }
    wf_bn1_shift.close();
    modelInstance.bind("bn1_param_shifts", bn1Shifts);
    delete[] bn1Shifts;

    // Layer 1 parameters
    // layer1.0.conv1
    unsigned int* layer1_0_conv1_weights = new unsigned int[k_size_x * k_size_y * in_channels_layer1 * out_channels_layer1];
    std::ifstream wf_layer1_0_conv1;
    wf_layer1_0_conv1.open("ResNet_20_weights/layer1.0.conv1_weights.txt");
    i = 0;
    while (wf_layer1_0_conv1 >> layer1_0_conv1_weights[i]) {
        i++;
    }
    wf_layer1_0_conv1.close();
    modelInstance.bind("layer1.0.conv1_weights", layer1_0_conv1_weights);
    delete[] layer1_0_conv1_weights;

    // layer1.0.bn1
    float* layer1_0_bn1_scales = new float[out_channels_layer1];
    std::ifstream wf_layer1_0_bn1_scale;
    wf_layer1_0_bn1_scale.open("ResNet_20_weights/layer1.0.bn1_scale.txt");
    i = 0;
    while (wf_layer1_0_bn1_scale >> layer1_0_bn1_scales[i]) {
        i++;
    }
    wf_layer1_0_bn1_scale.close();
    modelInstance.bind("layer1.0.bn1_param_scales", layer1_0_bn1_scales);
    delete[] layer1_0_bn1_scales;

    float* layer1_0_bn1_shifts = new float[out_channels_layer1];
    std::ifstream wf_layer1_0_bn1_shift;
    wf_layer1_0_bn1_shift.open("ResNet_20_weights/layer1.0.bn1_shift.txt");
    i = 0;
    while (wf_layer1_0_bn1_shift >> layer1_0_bn1_shifts[i]) {
        i++;
    }
    wf_layer1_0_bn1_shift.close();
    modelInstance.bind("layer1.0.bn1_param_shifts", layer1_0_bn1_shifts);
    delete[] layer1_0_bn1_shifts;

    // layer1.0.conv2
    unsigned int* layer1_0_conv2_weights = new unsigned int[k_size_x * k_size_y * out_channels_layer1 * out_channels_layer1];
    std::ifstream wf_layer1_0_conv2;
    wf_layer1_0_conv2.open("ResNet_20_weights/layer1.0.conv2_weights.txt");
    i = 0;
    while (wf_layer1_0_conv2 >> layer1_0_conv2_weights[i]) {
        i++;
    }
    wf_layer1_0_conv2.close();
    modelInstance.bind("layer1.0.conv2_weights", layer1_0_conv2_weights);
    delete[] layer1_0_conv2_weights;

    // layer1.0.bn2
    float* layer1_0_bn2_scales = new float[out_channels_layer1];
    std::ifstream wf_layer1_0_bn2_scale;
    wf_layer1_0_bn2_scale.open("ResNet_20_weights/layer1.0.bn2_scale.txt");
    i = 0;
    while (wf_layer1_0_bn2_scale >> layer1_0_bn2_scales[i]) {
        i++;
    }
    wf_layer1_0_bn2_scale.close();
    modelInstance.bind("layer1.0.bn2_param_scales", layer1_0_bn2_scales);
    delete[] layer1_0_bn2_scales;

    float* layer1_0_bn2_shifts = new float[out_channels_layer1];
    std::ifstream wf_layer1_0_bn2_shift;
    wf_layer1_0_bn2_shift.open("ResNet_20_weights/layer1.0.bn2_shift.txt");
    i = 0;
    while (wf_layer1_0_bn2_shift >> layer1_0_bn2_shifts[i]) {
        i++;
    }
    wf_layer1_0_bn2_shift.close();
    modelInstance.bind("layer1.0.bn2_param_shifts", layer1_0_bn2_shifts);
    delete[] layer1_0_bn2_shifts;

    // layer1.1.conv1
    unsigned int* layer1_1_conv1_weights = new unsigned int[k_size_x * k_size_y * out_channels_layer1 * out_channels_layer1];
    std::ifstream wf_layer1_1_conv1;
    wf_layer1_1_conv1.open("ResNet_20_weights/layer1.1.conv1_weights.txt");
    i = 0;
    while (wf_layer1_1_conv1 >> layer1_1_conv1_weights[i]) {
        i++;
    }
    wf_layer1_1_conv1.close();
    modelInstance.bind("layer1.1.conv1_weights", layer1_1_conv1_weights);
    delete[] layer1_1_conv1_weights;

    // layer1.1.bn1
    float* layer1_1_bn1_scales = new float[out_channels_layer1];
    std::ifstream wf_layer1_1_bn1_scale;
    wf_layer1_1_bn1_scale.open("ResNet_20_weights/layer1.1.bn1_scale.txt");
    i = 0;
    while (wf_layer1_1_bn1_scale >> layer1_1_bn1_scales[i]) {
        i++;
    }
    wf_layer1_1_bn1_scale.close();
    modelInstance.bind("layer1.1.bn1_param_scales", layer1_1_bn1_scales);
    delete[] layer1_1_bn1_scales;

    float* layer1_1_bn1_shifts = new float[out_channels_layer1];
    std::ifstream wf_layer1_1_bn1_shift;
    wf_layer1_1_bn1_shift.open("ResNet_20_weights/layer1.1.bn1_shift.txt");
    i = 0;
    while (wf_layer1_1_bn1_shift >> layer1_1_bn1_shifts[i]) {
        i++;
    }
    wf_layer1_1_bn1_shift.close();
    modelInstance.bind("layer1.1.bn1_param_shifts", layer1_1_bn1_shifts);
    delete[] layer1_1_bn1_shifts;

    // layer1.1.conv2
    unsigned int* layer1_1_conv2_weights = new unsigned int[k_size_x * k_size_y * out_channels_layer1 * out_channels_layer1];
    std::ifstream wf_layer1_1_conv2;
    wf_layer1_1_conv2.open("ResNet_20_weights/layer1.1.conv2_weights.txt");
    i = 0;
    while (wf_layer1_1_conv2 >> layer1_1_conv2_weights[i]) {
        i++;
    }
    wf_layer1_1_conv2.close();
    modelInstance.bind("layer1.1.conv2_weights", layer1_1_conv2_weights);
    delete[] layer1_1_conv2_weights;

    // layer1.1.bn2
    float* layer1_1_bn2_scales = new float[out_channels_layer1];
    std::ifstream wf_layer1_1_bn2_scale;
    wf_layer1_1_bn2_scale.open("ResNet_20_weights/layer1.1.bn2_scale.txt");
    i = 0;
    while (wf_layer1_1_bn2_scale >> layer1_1_bn2_scales[i]) {
        i++;
    }
    wf_layer1_1_bn2_scale.close();
    modelInstance.bind("layer1.1.bn2_param_scales", layer1_1_bn2_scales);
    delete[] layer1_1_bn2_scales;

    float* layer1_1_bn2_shifts = new float[out_channels_layer1];
    std::ifstream wf_layer1_1_bn2_shift;
    wf_layer1_1_bn2_shift.open("ResNet_20_weights/layer1.1.bn2_shift.txt");
    i = 0;
    while (wf_layer1_1_bn2_shift >> layer1_1_bn2_shifts[i]) {
        i++;
    }
    wf_layer1_1_bn2_shift.close();
    modelInstance.bind("layer1.1.bn2_param_shifts", layer1_1_bn2_shifts);
    delete[] layer1_1_bn2_shifts;

    // layer1.2.conv1
    unsigned int* layer1_2_conv1_weights = new unsigned int[k_size_x * k_size_y * out_channels_layer1 * out_channels_layer1];
    std::ifstream wf_layer1_2_conv1;
    wf_layer1_2_conv1.open("ResNet_20_weights/layer1.2.conv1_weights.txt");
    i = 0;
    while (wf_layer1_2_conv1 >> layer1_2_conv1_weights[i]) {
        i++;
    }
    wf_layer1_2_conv1.close();
    modelInstance.bind("layer1.2.conv1_weights", layer1_2_conv1_weights);
    delete[] layer1_2_conv1_weights;

    // layer1.2.bn1
    float* layer1_2_bn1_scales = new float[out_channels_layer1];
    std::ifstream wf_layer1_2_bn1_scale;
    wf_layer1_2_bn1_scale.open("ResNet_20_weights/layer1.2.bn1_scale.txt");
    i = 0;
    while (wf_layer1_2_bn1_scale >> layer1_2_bn1_scales[i]) {
        i++;
    }
    wf_layer1_2_bn1_scale.close();
    modelInstance.bind("layer1.2.bn1_param_scales", layer1_2_bn1_scales);
    delete[] layer1_2_bn1_scales;

    float* layer1_2_bn1_shifts = new float[out_channels_layer1];
    std::ifstream wf_layer1_2_bn1_shift;
    wf_layer1_2_bn1_shift.open("ResNet_20_weights/layer1.2.bn1_shift.txt");
    i = 0;
    while (wf_layer1_2_bn1_shift >> layer1_2_bn1_shifts[i]) {
        i++;
    }
    wf_layer1_2_bn1_shift.close();
    modelInstance.bind("layer1.2.bn1_param_shifts", layer1_2_bn1_shifts);
    delete[] layer1_2_bn1_shifts;

    // layer1.2.conv2
    unsigned int* layer1_2_conv2_weights = new unsigned int[k_size_x * k_size_y * out_channels_layer1 * out_channels_layer1];
    std::ifstream wf_layer1_2_conv2;
    wf_layer1_2_conv2.open("ResNet_20_weights/layer1.2.conv2_weights.txt");
    i = 0;
    while (wf_layer1_2_conv2 >> layer1_2_conv2_weights[i]) {
        i++;
    }
    wf_layer1_2_conv2.close();
    modelInstance.bind("layer1.2.conv2_weights", layer1_2_conv2_weights);
    delete[] layer1_2_conv2_weights;

    // layer1.2.bn2
    float* layer1_2_bn2_scales = new float[out_channels_layer1];
    std::ifstream wf_layer1_2_bn2_scale;
    wf_layer1_2_bn2_scale.open("ResNet_20_weights/layer1.2.bn2_scale.txt");
    i = 0;
    while (wf_layer1_2_bn2_scale >> layer1_2_bn2_scales[i]) {
        i++;
    }
    wf_layer1_2_bn2_scale.close();
    modelInstance.bind("layer1.2.bn2_param_scales", layer1_2_bn2_scales);
    delete[] layer1_2_bn2_scales;

    float* layer1_2_bn2_shifts = new float[out_channels_layer1];
    std::ifstream wf_layer1_2_bn2_shift;
    wf_layer1_2_bn2_shift.open("ResNet_20_weights/layer1.2.bn2_shift.txt");
    i = 0;
    while (wf_layer1_2_bn2_shift >> layer1_2_bn2_shifts[i]) {
        i++;
    }
    wf_layer1_2_bn2_shift.close();
    modelInstance.bind("layer1.2.bn2_param_shifts", layer1_2_bn2_shifts);
    delete[] layer1_2_bn2_shifts;

    // Layer 2 parameters
    // layer2.0.conv1
    unsigned int* layer2_0_conv1_weights = new unsigned int[k_size_x * k_size_y * in_channels_layer2 * out_channels_layer2];
    std::ifstream wf_layer2_0_conv1;
    wf_layer2_0_conv1.open("ResNet_20_weights/layer2.0.conv1_weights.txt");
    i = 0;
    while (wf_layer2_0_conv1 >> layer2_0_conv1_weights[i]) {
        i++;
    }
    wf_layer2_0_conv1.close();
    modelInstance.bind("layer2.0.conv1_weights", layer2_0_conv1_weights);
    delete[] layer2_0_conv1_weights;

    // layer2.0.bn1
    float* layer2_0_bn1_scales = new float[out_channels_layer2];
    std::ifstream wf_layer2_0_bn1_scale;
    wf_layer2_0_bn1_scale.open("ResNet_20_weights/layer2.0.bn1_scale.txt");
    i = 0;
    while (wf_layer2_0_bn1_scale >> layer2_0_bn1_scales[i]) {
        i++;
    }
    wf_layer2_0_bn1_scale.close();
    modelInstance.bind("layer2.0.bn1_param_scales", layer2_0_bn1_scales);
    delete[] layer2_0_bn1_scales;

    float* layer2_0_bn1_shifts = new float[out_channels_layer2];
    std::ifstream wf_layer2_0_bn1_shift;
    wf_layer2_0_bn1_shift.open("ResNet_20_weights/layer2.0.bn1_shift.txt");
    i = 0;
    while (wf_layer2_0_bn1_shift >> layer2_0_bn1_shifts[i]) {
        i++;
    }
    wf_layer2_0_bn1_shift.close();
    modelInstance.bind("layer2.0.bn1_param_shifts", layer2_0_bn1_shifts);
    delete[] layer2_0_bn1_shifts;

    // layer2.0.conv2
    unsigned int* layer2_0_conv2_weights = new unsigned int[k_size_x * k_size_y * out_channels_layer2 * out_channels_layer2];
    std::ifstream wf_layer2_0_conv2;
    wf_layer2_0_conv2.open("ResNet_20_weights/layer2.0.conv2_weights.txt");
    i = 0;
    while (wf_layer2_0_conv2 >> layer2_0_conv2_weights[i]) {
        i++;
    }
    wf_layer2_0_conv2.close();
    modelInstance.bind("layer2.0.conv2_weights", layer2_0_conv2_weights);
    delete[] layer2_0_conv2_weights;

    // layer2.0.bn2
    float* layer2_0_bn2_scales = new float[out_channels_layer2];
    std::ifstream wf_layer2_0_bn2_scale;
    wf_layer2_0_bn2_scale.open("ResNet_20_weights/layer2.0.bn2_scale.txt");
    i = 0;
    while (wf_layer2_0_bn2_scale >> layer2_0_bn2_scales[i]) {
        i++;
    }
    wf_layer2_0_bn2_scale.close();
    modelInstance.bind("layer2.0.bn2_param_scales", layer2_0_bn2_scales);
    delete[] layer2_0_bn2_scales;

    float* layer2_0_bn2_shifts = new float[out_channels_layer2];
    std::ifstream wf_layer2_0_bn2_shift;
    wf_layer2_0_bn2_shift.open("ResNet_20_weights/layer2.0.bn2_shift.txt");
    i = 0;
    while (wf_layer2_0_bn2_shift >> layer2_0_bn2_shifts[i]) {
        i++;
    }
    wf_layer2_0_bn2_shift.close();
    modelInstance.bind("layer2.0.bn2_param_shifts", layer2_0_bn2_shifts);
    delete[] layer2_0_bn2_shifts;

    // layer2.0.shortcut.0
    unsigned int* layer2_0_shortcut_0_weights = new unsigned int[1 * 1 * in_channels_layer2 * out_channels_layer2];
    std::ifstream wf_layer2_0_shortcut_0;
    wf_layer2_0_shortcut_0.open("ResNet_20_weights/layer2.0.shortcut.0_weights.txt");
    i = 0;
    while (wf_layer2_0_shortcut_0 >> layer2_0_shortcut_0_weights[i]) {
        i++;
    }
    wf_layer2_0_shortcut_0.close();
    modelInstance.bind("layer2.0.shortcut.0_weights", layer2_0_shortcut_0_weights);
    delete[] layer2_0_shortcut_0_weights;

    // layer2.0.shortcut.1
    float* layer2_0_shortcut_1_scales = new float[out_channels_layer2];
    std::ifstream wf_layer2_0_shortcut_1_scale;
    wf_layer2_0_shortcut_1_scale.open("ResNet_20_weights/layer2.0.shortcut.1_scale.txt");
    i = 0;
    while (wf_layer2_0_shortcut_1_scale >> layer2_0_shortcut_1_scales[i]) {
        i++;
    }
    wf_layer2_0_shortcut_1_scale.close();
    modelInstance.bind("layer2.0.shortcut.1_param_scales", layer2_0_shortcut_1_scales);
    delete[] layer2_0_shortcut_1_scales;

    float* layer2_0_shortcut_1_shifts = new float[out_channels_layer2];
    std::ifstream wf_layer2_0_shortcut_1_shift;
    wf_layer2_0_shortcut_1_shift.open("ResNet_20_weights/layer2.0.shortcut.1_shift.txt");
    i = 0;
    while (wf_layer2_0_shortcut_1_shift >> layer2_0_shortcut_1_shifts[i]) {
        i++;
    }
    wf_layer2_0_shortcut_1_shift.close();
    modelInstance.bind("layer2.0.shortcut.1_param_shifts", layer2_0_shortcut_1_shifts);
    delete[] layer2_0_shortcut_1_shifts;

    // layer2.1.conv1
    unsigned int* layer2_1_conv1_weights = new unsigned int[k_size_x * k_size_y * out_channels_layer2 * out_channels_layer2];
    std::ifstream wf_layer2_1_conv1;
    wf_layer2_1_conv1.open("ResNet_20_weights/layer2.1.conv1_weights.txt");
    i = 0;
    while (wf_layer2_1_conv1 >> layer2_1_conv1_weights[i]) {
        i++;
    }
    wf_layer2_1_conv1.close();
    modelInstance.bind("layer2.1.conv1_weights", layer2_1_conv1_weights);
    delete[] layer2_1_conv1_weights;

    // layer2.1.bn1
    float* layer2_1_bn1_scales = new float[out_channels_layer2];
    std::ifstream wf_layer2_1_bn1_scale;
    wf_layer2_1_bn1_scale.open("ResNet_20_weights/layer2.1.bn1_scale.txt");
    i = 0;
    while (wf_layer2_1_bn1_scale >> layer2_1_bn1_scales[i]) {
        i++;
    }
    wf_layer2_1_bn1_scale.close();
    modelInstance.bind("layer2.1.bn1_param_scales", layer2_1_bn1_scales);
    delete[] layer2_1_bn1_scales;

    float* layer2_1_bn1_shifts = new float[out_channels_layer2];
    std::ifstream wf_layer2_1_bn1_shift;
    wf_layer2_1_bn1_shift.open("ResNet_20_weights/layer2.1.bn1_shift.txt");
    i = 0;
    while (wf_layer2_1_bn1_shift >> layer2_1_bn1_shifts[i]) {
        i++;
    }
    wf_layer2_1_bn1_shift.close();
    modelInstance.bind("layer2.1.bn1_param_shifts", layer2_1_bn1_shifts);
    delete[] layer2_1_bn1_shifts;

    // layer2.1.conv2
    unsigned int* layer2_1_conv2_weights = new unsigned int[k_size_x * k_size_y * out_channels_layer2 * out_channels_layer2];
    std::ifstream wf_layer2_1_conv2;
    wf_layer2_1_conv2.open("ResNet_20_weights/layer2.1.conv2_weights.txt");
    i = 0;
    while (wf_layer2_1_conv2 >> layer2_1_conv2_weights[i]) {
        i++;
    }
    wf_layer2_1_conv2.close();
    modelInstance.bind("layer2.1.conv2_weights", layer2_1_conv2_weights);
    delete[] layer2_1_conv2_weights;

    // layer2.1.bn2
    float* layer2_1_bn2_scales = new float[out_channels_layer2];
    std::ifstream wf_layer2_1_bn2_scale;
    wf_layer2_1_bn2_scale.open("ResNet_20_weights/layer2.1.bn2_scale.txt");
    i = 0;
    while (wf_layer2_1_bn2_scale >> layer2_1_bn2_scales[i]) {
        i++;
    }
    wf_layer2_1_bn2_scale.close();
    modelInstance.bind("layer2.1.bn2_param_scales", layer2_1_bn2_scales);
    delete[] layer2_1_bn2_scales;

    float* layer2_1_bn2_shifts = new float[out_channels_layer2];
    std::ifstream wf_layer2_1_bn2_shift;
    wf_layer2_1_bn2_shift.open("ResNet_20_weights/layer2.1.bn2_shift.txt");
    i = 0;
    while (wf_layer2_1_bn2_shift >> layer2_1_bn2_shifts[i]) {
        i++;
    }
    wf_layer2_1_bn2_shift.close();
    modelInstance.bind("layer2.1.bn2_param_shifts", layer2_1_bn2_shifts);
    delete[] layer2_1_bn2_shifts;

    // layer2.2.conv1
    unsigned int* layer2_2_conv1_weights = new unsigned int[k_size_x * k_size_y * out_channels_layer2 * out_channels_layer2];
    std::ifstream wf_layer2_2_conv1;
    wf_layer2_2_conv1.open("ResNet_20_weights/layer2.2.conv1_weights.txt");
    i = 0;
    while (wf_layer2_2_conv1 >> layer2_2_conv1_weights[i]) {
        i++;
    }
    wf_layer2_2_conv1.close();
    modelInstance.bind("layer2.2.conv1_weights", layer2_2_conv1_weights);
    delete[] layer2_2_conv1_weights;

    // layer2.2.bn1
    float* layer2_2_bn1_scales = new float[out_channels_layer2];
    std::ifstream wf_layer2_2_bn1_scale;
    wf_layer2_2_bn1_scale.open("ResNet_20_weights/layer2.2.bn1_scale.txt");
    i = 0;
    while (wf_layer2_2_bn1_scale >> layer2_2_bn1_scales[i]) {
        i++;
    }
    wf_layer2_2_bn1_scale.close();
    modelInstance.bind("layer2.2.bn1_param_scales", layer2_2_bn1_scales);
    delete[] layer2_2_bn1_scales;

    float* layer2_2_bn1_shifts = new float[out_channels_layer2];
    std::ifstream wf_layer2_2_bn1_shift;
    wf_layer2_2_bn1_shift.open("ResNet_20_weights/layer2.2.bn1_shift.txt");
    i = 0;
    while (wf_layer2_2_bn1_shift >> layer2_2_bn1_shifts[i]) {
        i++;
    }
    wf_layer2_2_bn1_shift.close();
    modelInstance.bind("layer2.2.bn1_param_shifts", layer2_2_bn1_shifts);
    delete[] layer2_2_bn1_shifts;

    // layer2.2.conv2
    unsigned int* layer2_2_conv2_weights = new unsigned int[k_size_x * k_size_y * out_channels_layer2 * out_channels_layer2];
    std::ifstream wf_layer2_2_conv2;
    wf_layer2_2_conv2.open("ResNet_20_weights/layer2.2.conv2_weights.txt");
    i = 0;
    while (wf_layer2_2_conv2 >> layer2_2_conv2_weights[i]) {
        i++;
    }
    wf_layer2_2_conv2.close();
    modelInstance.bind("layer2.2.conv2_weights", layer2_2_conv2_weights);
    delete[] layer2_2_conv2_weights;

    // layer2.2.bn2
    float* layer2_2_bn2_scales = new float[out_channels_layer2];
    std::ifstream wf_layer2_2_bn2_scale;
    wf_layer2_2_bn2_scale.open("ResNet_20_weights/layer2.2.bn2_scale.txt");
    i = 0;
    while (wf_layer2_2_bn2_scale >> layer2_2_bn2_scales[i]) {
        i++;
    }
    wf_layer2_2_bn2_scale.close();
    modelInstance.bind("layer2.2.bn2_param_scales", layer2_2_bn2_scales);
    delete[] layer2_2_bn2_scales;

    float* layer2_2_bn2_shifts = new float[out_channels_layer2];
    std::ifstream wf_layer2_2_bn2_shift;
    wf_layer2_2_bn2_shift.open("ResNet_20_weights/layer2.2.bn2_shift.txt");
    i = 0;
    while (wf_layer2_2_bn2_shift >> layer2_2_bn2_shifts[i]) {
        i++;
    }
    wf_layer2_2_bn2_shift.close();
    modelInstance.bind("layer2.2.bn2_param_shifts", layer2_2_bn2_shifts);
    delete[] layer2_2_bn2_shifts;

    // Layer 3 parameters
    // layer3.0.conv1
    unsigned int* layer3_0_conv1_weights = new unsigned int[k_size_x * k_size_y * in_channels_layer3 * out_channels_layer3];
    std::ifstream wf_layer3_0_conv1;
    wf_layer3_0_conv1.open("ResNet_20_weights/layer3.0.conv1_weights.txt");
    i = 0;
    while (wf_layer3_0_conv1 >> layer3_0_conv1_weights[i]) {
        i++;
    }
    wf_layer3_0_conv1.close();
    modelInstance.bind("layer3.0.conv1_weights", layer3_0_conv1_weights);
    delete[] layer3_0_conv1_weights;

    // layer3.0.bn1
    float* layer3_0_bn1_scales = new float[out_channels_layer3];
    std::ifstream wf_layer3_0_bn1_scale;
    wf_layer3_0_bn1_scale.open("ResNet_20_weights/layer3.0.bn1_scale.txt");
    i = 0;
    while (wf_layer3_0_bn1_scale >> layer3_0_bn1_scales[i]) {
        i++;
    }
    wf_layer3_0_bn1_scale.close();
    modelInstance.bind("layer3.0.bn1_param_scales", layer3_0_bn1_scales);
    delete[] layer3_0_bn1_scales;

    float* layer3_0_bn1_shifts = new float[out_channels_layer3];
    std::ifstream wf_layer3_0_bn1_shift;
    wf_layer3_0_bn1_shift.open("ResNet_20_weights/layer3.0.bn1_shift.txt");
    i = 0;
    while (wf_layer3_0_bn1_shift >> layer3_0_bn1_shifts[i]) {
        i++;
    }
    wf_layer3_0_bn1_shift.close();
    modelInstance.bind("layer3.0.bn1_param_shifts", layer3_0_bn1_shifts);
    delete[] layer3_0_bn1_shifts;

    // layer3.0.conv2
    unsigned int* layer3_0_conv2_weights = new unsigned int[k_size_x * k_size_y * out_channels_layer3 * out_channels_layer3];
    std::ifstream wf_layer3_0_conv2;
    wf_layer3_0_conv2.open("ResNet_20_weights/layer3.0.conv2_weights.txt");
    i = 0;
    while (wf_layer3_0_conv2 >> layer3_0_conv2_weights[i]) {
        i++;
    }
    wf_layer3_0_conv2.close();
    modelInstance.bind("layer3.0.conv2_weights", layer3_0_conv2_weights);
    delete[] layer3_0_conv2_weights;

    // layer3.0.bn2
    float* layer3_0_bn2_scales = new float[out_channels_layer3];
    std::ifstream wf_layer3_0_bn2_scale;
    wf_layer3_0_bn2_scale.open("ResNet_20_weights/layer3.0.bn2_scale.txt");
    i = 0;
    while (wf_layer3_0_bn2_scale >> layer3_0_bn2_scales[i]) {
        i++;
    }
    wf_layer3_0_bn2_scale.close();
    modelInstance.bind("layer3.0.bn2_param_scales", layer3_0_bn2_scales);
    delete[] layer3_0_bn2_scales;

    float* layer3_0_bn2_shifts = new float[out_channels_layer3];
    std::ifstream wf_layer3_0_bn2_shift;
    wf_layer3_0_bn2_shift.open("ResNet_20_weights/layer3.0.bn2_shift.txt");
    i = 0;
    while (wf_layer3_0_bn2_shift >> layer3_0_bn2_shifts[i]) {
        i++;
    }
    wf_layer3_0_bn2_shift.close();
    modelInstance.bind("layer3.0.bn2_param_shifts", layer3_0_bn2_shifts);
    delete[] layer3_0_bn2_shifts;

    // layer3.0.shortcut.0
    unsigned int* layer3_0_shortcut_0_weights = new unsigned int[1 * 1 * in_channels_layer3 * out_channels_layer3];
    std::ifstream wf_layer3_0_shortcut_0;
    wf_layer3_0_shortcut_0.open("ResNet_20_weights/layer3.0.shortcut.0_weights.txt");
    i = 0;
    while (wf_layer3_0_shortcut_0 >> layer3_0_shortcut_0_weights[i]) {
        i++;
    }
    wf_layer3_0_shortcut_0.close();
    modelInstance.bind("layer3.0.shortcut.0_weights", layer3_0_shortcut_0_weights);
    delete[] layer3_0_shortcut_0_weights;

    // layer3.0.shortcut.1
    float* layer3_0_shortcut_1_scales = new float[out_channels_layer3];
    std::ifstream wf_layer3_0_shortcut_1_scale;
    wf_layer3_0_shortcut_1_scale.open("ResNet_20_weights/layer3.0.shortcut.1_scale.txt");
    i = 0;
    while (wf_layer3_0_shortcut_1_scale >> layer3_0_shortcut_1_scales[i]) {
        i++;
    }
    wf_layer3_0_shortcut_1_scale.close();
    modelInstance.bind("layer3.0.shortcut.1_param_scales", layer3_0_shortcut_1_scales);
    delete[] layer3_0_shortcut_1_scales;

    float* layer3_0_shortcut_1_shifts = new float[out_channels_layer3];
    std::ifstream wf_layer3_0_shortcut_1_shift;
    wf_layer3_0_shortcut_1_shift.open("ResNet_20_weights/layer3.0.shortcut.1_shift.txt");
    i = 0;
    while (wf_layer3_0_shortcut_1_shift >> layer3_0_shortcut_1_shifts[i]) {
        i++;
    }
    wf_layer3_0_shortcut_1_shift.close();
    modelInstance.bind("layer3.0.shortcut.1_param_shifts", layer3_0_shortcut_1_shifts);
    delete[] layer3_0_shortcut_1_shifts;

    // layer3.1.conv1
    unsigned int* layer3_1_conv1_weights = new unsigned int[k_size_x * k_size_y * out_channels_layer3 * out_channels_layer3];
    std::ifstream wf_layer3_1_conv1;
    wf_layer3_1_conv1.open("ResNet_20_weights/layer3.1.conv1_weights.txt");
    i = 0;
    while (wf_layer3_1_conv1 >> layer3_1_conv1_weights[i]) {
        i++;
    }
    wf_layer3_1_conv1.close();
    modelInstance.bind("layer3.1.conv1_weights", layer3_1_conv1_weights);
    delete[] layer3_1_conv1_weights;

    // layer3.1.bn1
    float* layer3_1_bn1_scales = new float[out_channels_layer3];
    std::ifstream wf_layer3_1_bn1_scale;
    wf_layer3_1_bn1_scale.open("ResNet_20_weights/layer3.1.bn1_scale.txt");
    i = 0;
    while (wf_layer3_1_bn1_scale >> layer3_1_bn1_scales[i]) {
        i++;
    }
    wf_layer3_1_bn1_scale.close();
    modelInstance.bind("layer3.1.bn1_param_scales", layer3_1_bn1_scales);
    delete[] layer3_1_bn1_scales;

    float* layer3_1_bn1_shifts = new float[out_channels_layer3];
    std::ifstream wf_layer3_1_bn1_shift;
    wf_layer3_1_bn1_shift.open("ResNet_20_weights/layer3.1.bn1_shift.txt");
    i = 0;
    while (wf_layer3_1_bn1_shift >> layer3_1_bn1_shifts[i]) {
        i++;
    }
    wf_layer3_1_bn1_shift.close();
    modelInstance.bind("layer3.1.bn1_param_shifts", layer3_1_bn1_shifts);
    delete[] layer3_1_bn1_shifts;

    // layer3.1.conv2
    unsigned int* layer3_1_conv2_weights = new unsigned int[k_size_x * k_size_y * out_channels_layer3 * out_channels_layer3];
    std::ifstream wf_layer3_1_conv2;
    wf_layer3_1_conv2.open("ResNet_20_weights/layer3.1.conv2_weights.txt");
    i = 0;
    while (wf_layer3_1_conv2 >> layer3_1_conv2_weights[i]) {
        i++;
    }
    wf_layer3_1_conv2.close();
    modelInstance.bind("layer3.1.conv2_weights", layer3_1_conv2_weights);
    delete[] layer3_1_conv2_weights;

    // layer3.1.bn2
    float* layer3_1_bn2_scales = new float[out_channels_layer3];
    std::ifstream wf_layer3_1_bn2_scale;
    wf_layer3_1_bn2_scale.open("ResNet_20_weights/layer3.1.bn2_scale.txt");
    i = 0;
    while (wf_layer3_1_bn2_scale >> layer3_1_bn2_scales[i]) {
        i++;
    }
    wf_layer3_1_bn2_scale.close();
    modelInstance.bind("layer3.1.bn2_param_scales", layer3_1_bn2_scales);
    delete[] layer3_1_bn2_scales;

    float* layer3_1_bn2_shifts = new float[out_channels_layer3];
    std::ifstream wf_layer3_1_bn2_shift;
    wf_layer3_1_bn2_shift.open("ResNet_20_weights/layer3.1.bn2_shift.txt");
    i = 0;
    while (wf_layer3_1_bn2_shift >> layer3_1_bn2_shifts[i]) {
        i++;
    }
    wf_layer3_1_bn2_shift.close();
    modelInstance.bind("layer3.1.bn2_param_shifts", layer3_1_bn2_shifts);
    delete[] layer3_1_bn2_shifts;

    // layer3.2.conv1
    unsigned int* layer3_2_conv1_weights = new unsigned int[k_size_x * k_size_y * out_channels_layer3 * out_channels_layer3];
    std::ifstream wf_layer3_2_conv1;
    wf_layer3_2_conv1.open("ResNet_20_weights/layer3.2.conv1_weights.txt");
    i = 0;
    while (wf_layer3_2_conv1 >> layer3_2_conv1_weights[i]) {
        i++;
    }
    wf_layer3_2_conv1.close();
    modelInstance.bind("layer3.2.conv1_weights", layer3_2_conv1_weights);
    delete[] layer3_2_conv1_weights;

    // layer3.2.bn1
    float* layer3_2_bn1_scales = new float[out_channels_layer3];
    std::ifstream wf_layer3_2_bn1_scale;
    wf_layer3_2_bn1_scale.open("ResNet_20_weights/layer3.2.bn1_scale.txt");
    i = 0;
    while (wf_layer3_2_bn1_scale >> layer3_2_bn1_scales[i]) {
        i++;
    }
    wf_layer3_2_bn1_scale.close();
    modelInstance.bind("layer3.2.bn1_param_scales", layer3_2_bn1_scales);
    delete[] layer3_2_bn1_scales;

    float* layer3_2_bn1_shifts = new float[out_channels_layer3];
    std::ifstream wf_layer3_2_bn1_shift;
    wf_layer3_2_bn1_shift.open("ResNet_20_weights/layer3.2.bn1_shift.txt");
    i = 0;
    while (wf_layer3_2_bn1_shift >> layer3_2_bn1_shifts[i]) {
        i++;
    }
    wf_layer3_2_bn1_shift.close();
    modelInstance.bind("layer3.2.bn1_param_shifts", layer3_2_bn1_shifts);
    delete[] layer3_2_bn1_shifts;

    // layer3.2.conv2
    unsigned int* layer3_2_conv2_weights = new unsigned int[k_size_x * k_size_y * out_channels_layer3 * out_channels_layer3];
    std::ifstream wf_layer3_2_conv2;
    wf_layer3_2_conv2.open("ResNet_20_weights/layer3.2.conv2_weights.txt");
    i = 0;
    while (wf_layer3_2_conv2 >> layer3_2_conv2_weights[i]) {
        i++;
    }
    wf_layer3_2_conv2.close();
    modelInstance.bind("layer3.2.conv2_weights", layer3_2_conv2_weights);
    delete[] layer3_2_conv2_weights;

    // layer3.2.bn2
    float* layer3_2_bn2_scales = new float[out_channels_layer3];
    std::ifstream wf_layer3_2_bn2_scale;
    wf_layer3_2_bn2_scale.open("ResNet_20_weights/layer3.2.bn2_scale.txt");
    i = 0;
    while (wf_layer3_2_bn2_scale >> layer3_2_bn2_scales[i]) {
        i++;
    }
    wf_layer3_2_bn2_scale.close();
    modelInstance.bind("layer3.2.bn2_param_scales", layer3_2_bn2_scales);
    delete[] layer3_2_bn2_scales;

    float* layer3_2_bn2_shifts = new float[out_channels_layer3];
    std::ifstream wf_layer3_2_bn2_shift;
    wf_layer3_2_bn2_shift.open("ResNet_20_weights/layer3.2.bn2_shift.txt");
    i = 0;
    while (wf_layer3_2_bn2_shift >> layer3_2_bn2_shifts[i]) {
        i++;
    }
    wf_layer3_2_bn2_shift.close();
    modelInstance.bind("layer3.2.bn2_param_shifts", layer3_2_bn2_shifts);
    delete[] layer3_2_bn2_shifts;

    // Linear layer
    unsigned int* linear_weights = new unsigned int[in_size_fc * out_size_fc];
    std::ifstream wf_linear;
    wf_linear.open("ResNet_20_weights/linear_weights.txt");
    i = 0;
    while (wf_linear >> linear_weights[i]) {
        i++;
    }
    wf_linear.close();
    modelInstance.bind("linear_weights", linear_weights);
    delete[] linear_weights;

    float* linear_biases = new float[out_size_fc];
    std::ifstream wf_linear_b;
    wf_linear_b.open("ResNet_20_weights/linear_biases.txt");
    i = 0;
    while (wf_linear_b >> linear_biases[i]) {
        i++;
    }
    wf_linear_b.close();
    modelInstance.bind("linear_biases", linear_biases);
    delete[] linear_biases;

    // Generate data
    modelInstance.generateData();

    // Destroy the model
    model.destroy();

    return 0;
}