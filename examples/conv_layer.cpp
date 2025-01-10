#include <iostream>
#include <fstream>
#include "puma.h"
#include "conv.h"

int main() {

    Model model = Model::create("conv_layer");

    // Layer parameters
    unsigned int in_size_x = 6;
    unsigned int in_size_y = 6;
    unsigned int out_size_x = 4;
    unsigned int out_size_y = 4;
    unsigned int kernel_size[4] = {3, 3, 1, 1}; // kernel_size = [k_size_x, k_size_y, in_channels, out_channels]


    // Input stream
    auto in_stream = InputImagePixelStream::create(model, "in_stream", in_size_x, in_size_y, kernel_size[2]);

    // Output stream
    auto out_stream = OutputImagePixelStream::create(model, "out_stream", out_size_x, out_size_y, kernel_size[3]);

    // Layer
    out_stream = conv2d(model, "layer", kernel_size, in_stream, 1, 1, 0, 0);

    // Compile
    model.compile();

    // Load weights
    float* weights = new float[kernel_size[0] * kernel_size[1] * kernel_size[2] * kernel_size[3]];

    int i=0;
    std::ifstream wf;
    wf.open("conv_layer_weights/weight_1_1_3_3.txt");
    while(wf >> weights[i])
    { i++; }
    wf.close();

    ModelInstance modelInstance = ModelInstance::create(model);
    modelInstance.load("layer", weights);
    modelInstance.generateData();

    // Destroy model
    model.destroy();
    delete[] weights;

    return 0;
}