/*
 *  Copyright (c) 2019 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#ifndef _PUMA_TEST_FULLY_CONNECTED_LAYER_
#define _PUMA_TEST_FULLY_CONNECTED_LAYER_

#include "puma.h"

static Vector linear(Model model, std::string layerName, unsigned int in_size, unsigned int out_size, Vector in, bool has_bias = true, unsigned int storageType = 1, float activation_scale=1.0f, float weights_scale=1.0f, int activation_zero_point=0, int weights_zero_point=0) {

    ConstantMatrix weight = ConstantMatrix::create(model, layerName + "_weight", in_size, out_size, storageType, activation_scale, weights_scale, activation_zero_point, weights_zero_point);
    if (has_bias) {
        ConstantVector bias = ConstantVector::create(model, layerName + "_bias", out_size);
        return weight * in + bias;
    }
    else {
        return weight * in;
    }
}

#endif

