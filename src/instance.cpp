/*
 *  Copyright (c) 2019 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#include <assert.h>
#include <fstream>
#include <sstream>

#include "instance.h"
#include "model.h"
#include "placer.h"
#include "regalloc.h"
#include "tensors.h"

#include <nlohmann/json.hpp>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <errno.h>
#endif

using json = nlohmann::json;

ModelInstance ModelInstance::create(Model model) {
    ModelInstance instance;
    instance.impl_ = model.unwrap()->createInstance();
    return instance;
}

void ModelInstance::bind(std::string tensorName, float* data) {
    impl_->bind(tensorName, data);
}

void ModelInstance::load(std::string layerName, float* weights) {
    impl_->load(layerName, weights);
}

void ModelInstance::generateData() {
    impl_->generateData();
}

ModelInstanceImpl* ModelInstance::unwrap() {
    return impl_;
}

ModelInstanceImpl::ModelInstanceImpl(ModelImpl* model, Placer* placer, RegisterAllocator* registerAllocator)
    : model_(model), placer_(placer), registerAllocator_(registerAllocator) 
{ }

void ModelInstanceImpl::bind(std::string tensorName, float* data) {
    tensorData_[tensorName] = data;
}

void ModelInstanceImpl::load(std::string layerName, float* weights) {
    bind(layerName + "_weight", weights);
}

void ModelInstanceImpl::generateData() {

    // TODO: Define ABI for laying out the data

    std::cout << "Generating data files... " << std::flush;

    // Generate a directory with model name
    std::string dirName = model_->getName();

    #ifdef _WIN32
    _mkdir(dirName.c_str()); // Windows
    #else
    mkdir(dirName.c_str(), 0777); // Linux/Unix
    #endif

    json js = json::array();

    for(auto m = model_->const_mat_begin(); m != model_->const_mat_end(); ++m) {
        ConstantMatrixImpl* mat = *m;
        std::string matName = mat->name();
        assert(tensorData_.count(matName) && "No data provided for matrix");
        float* matData = tensorData_[matName];
        for(unsigned int h = 0; h < mat->nHeightTiles(); ++h) {
            for(unsigned int w = 0; w < mat->nWidthTiles(); ++w) {
                ConstantMatrixTile* matTile = mat->getTile(h, w);
                unsigned int pTile = placer_->getPTile(matTile);
                unsigned int pCore = placer_->getPCore(matTile);
                unsigned int pMVMU = placer_->getPMVMU(matTile);
                json temp;
                temp["tile"] = pTile;
                temp["core"] = pCore;
                temp["mvmu"] = pMVMU;
                temp["value"] = json::array();
                //std::stringstream fileName;
                //fileName << dirName << "/" << model_->getName() << "-tile" << pTile << "-core" << pCore << "-mvmu" << pMVMU << ".weights";
                //std::ofstream mvmuData;
                //mvmuData.open(fileName.str());
                for(unsigned int row = 0; row < MVMU_DIM; ++row) {
                    for(unsigned int col = 0; col < MVMU_DIM; ++col) {
                        if(row < matTile->height() && col < matTile->width()) {
                            //mvmuData << matData[(h*MVMU_DIM + row)*mat->width() + w*MVMU_DIM + col] << " ";
                            float data = matData[(h*MVMU_DIM + row)*mat->width() + w*MVMU_DIM + col];
                            temp["value"].push_back(data);
                        } else {
                            //mvmuData << "0.0 ";
                            temp["value"].push_back(0.0);
                        }
                    }
                }
                //mvmuData.close();
                js.push_back(temp);
            }
        }
    }
    for(auto m = model_->conv_mat_begin(); m != model_->conv_mat_end(); ++m) {
        ConvolutionalConstantMatrixImpl* mat = *m;
        std::string matName = mat->name();
        assert(tensorData_.count(matName) && "No data provided for matrix");
        float* matData = tensorData_[matName];
        for(unsigned int kh = 0; kh < mat->getKernelHeight(); ++kh) {
            for(unsigned int kw = 0; kw < mat->getKernelWidth(); ++kw) {
                for(unsigned int h = 0; h < mat->getNOutChannelTiles(); ++h) {
                    for(unsigned int w = 0; w < mat->getNInChannelTiles(); ++w) {
                        ConstantMatrixTile* matTile = mat->getTile(kh, kw, h, w);
                        unsigned int pTile = placer_->getPTile(matTile);
                        unsigned int pCore = placer_->getPCore(matTile);
                        unsigned int pMVMU = placer_->getPMVMU(matTile);
                        json temp;
                        temp["tile"] = pTile;
                        temp["core"] = pCore;
                        temp["mvmu"] = pMVMU;
                        temp["value"] = json::array();
                        //std::stringstream fileName;
                        //fileName << dirName << "/" << model_->getName() << "-tile" << pTile << "-core" << pCore << "-mvmu" << pMVMU << ".weights";
                        //std::ofstream mvmuData;
                        //mvmuData.open(fileName.str());
                        for(unsigned int row = 0; row < MVMU_DIM; ++row) {
                            for(unsigned int col = 0; col < MVMU_DIM; ++col) {
                                if(row < matTile->height() && col < matTile->width()) {
                                    //mvmuData << matData[((kh*mat->getKernelWidth() + kw)*mat->getNOutChannels() + h*MVMU_DIM + row)*mat->getNInChannels() + w*MVMU_DIM + col] << " ";
                                    float data = matData[((kh*mat->getKernelWidth() + kw)*mat->getNOutChannels() + h*MVMU_DIM + row)*mat->getNInChannels() + w*MVMU_DIM + col];
                                    temp["value"].push_back(data);
                                } else {
                                    //mvmuData << "0.0 ";
                                    temp["value"].push_back(0.0);
                                }
                            }
                        }
                        //mvmuData.close();
                        js.push_back(temp);
                    }
                }
            }
        }
    }
    for (auto v = model_->const_vec_begin(); v != model_->const_vec_end(); ++v) {
        ConstantVectorImpl* vec = *v;
        std::string vecName = vec->name();
        assert(tensorData_.count(vecName) && "No data provided for vector");
        float* vecData = tensorData_[vecName];
        for (unsigned int t = 0; t < vec->nTiles(); ++t) {
            ConstantVectorTile* tile = vec->getTile(t);
            unsigned int pTile = placer_->getPTile(tile);
            unsigned int pCore = placer_->getPCore(tile);
            unsigned int pMVMU = placer_->getPMVMU(tile);
            unsigned int startAddr = registerAllocator_->getRegister(tile);
            json temp;
            temp["tile"] = pTile;
            temp["core"] = pCore;
            temp["mvmu"] = pMVMU;
            temp["reg"] = startAddr;
            temp["value"] = json::array();
            for (unsigned int i = 0; i < tile->length(); ++i) {
                float data = vecData[t * MVMU_DIM + i];
                temp["value"].push_back(data);
            }
            js.push_back(temp);
        }
    }

    std::ofstream jsonFile(dirName + "/weights.json");
    if (!jsonFile.is_open()) {
        std::cerr << "Error opening JSON file for writing." << std::endl;
        return;
    }
    jsonFile << js.dump();
    jsonFile.close();

    std::cout << "done." << std::endl;

}

