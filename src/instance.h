/*
 *  Copyright (c) 2019 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#include <map>

#include "common.h"

class ModelInstanceImpl {

    private:

        ModelImpl* model_;
        Placer* placer_;
        RegisterAllocator* registerAllocator_;
        std::map<std::string, unsigned int*> matrixData_;
        std::map<std::string, float*> vectorData_;

    public:

        ModelInstanceImpl(ModelImpl* model, Placer* placer, RegisterAllocator* registerAllocator);

        void bind(std::string vectorName, float* data);
        void bind(std::string matrixName, unsigned int* data);
        void generateData();

};

