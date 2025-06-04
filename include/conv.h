/*
 *  Copyright (c) 2019 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#ifndef _PUMA_TEST_CONV_LAYER_
#define _PUMA_TEST_CONV_LAYER_

#include "puma.h"

static ImagePixelStream conv2d(Model model, std::string layerName, const unsigned int* kernel_size, ImagePixelStream in_stream, unsigned int stride_x = 1, unsigned int stride_y = 1, unsigned int padding_x = 0, unsigned int padding_y = 0, unsigned int storageType = 1) {

    ConvolutionalConstantMatrix mat = ConvolutionalConstantMatrix::create(model, layerName + "mat", kernel_size[0], kernel_size[1], kernel_size[2], kernel_size[3], storageType);

    return conv2d_forward(mat, in_stream, stride_x, stride_y, padding_x, padding_y);

}

static ImagePixelStream batchnorm2d(Model model, std::string layerName, ImagePixelStream in_stream, unsigned int n_channels) {

    BatchNormParam param = BatchNormParam::create(model, layerName + "param", n_channels);

    return batchnorm(in_stream, param);

}

#endif

