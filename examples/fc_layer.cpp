#include <iostream>
#include <fstream>
#include "puma.h"
#include "linear.h"

int main() {

    Model model = Model::create("fc_layer");

    unsigned int in_size = 256;
    auto in = InputVector::create(model, "in", in_size);

    unsigned int out_size = 128;
    auto out = OutputVector::create(model, "out", out_size);

    auto out1 = linear(model, "fcLayer", in_size, out_size, in);

    out = out1;

    model.compile();

    ModelInstance modelInstance = ModelInstance::create(model);

    float* fcLayerWeights = new float[in_size * out_size];

    std::ifstream wf;
    wf.open("fc_layer_weights/weight.txt");
    int i = 0;
    while(wf >> fcLayerWeights[i]) {
        i++;
    }
    wf.close();

    modelInstance.load("fcLayer", fcLayerWeights);

    modelInstance.generateData();

    model.destroy();

    delete[] fcLayerWeights;

    return 0;

}