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

static Vector linear(Model model, std::string layerName, unsigned int in_size, unsigned int out_size, Vector in, bool has_bias = true, unsigned int storageType = 1) {

    ConstantMatrix weight = ConstantMatrix::create(model, layerName + "weight", in_size, out_size, storageType);
    if (has_bias) {
        ConstantVector bias = ConstantVector::create(model, layerName + "bias", out_size);
        return weight * in + bias;
    }
    else {
        return weight * in;
    }
}

#endif

