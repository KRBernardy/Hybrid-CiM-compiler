#include <assert.h>
#include <iostream>
#include <fstream>
#include <sstream>

#include "puma.h"
#include <nlohmann/json.hpp>

#include "model.h"
#include "placer.h"
#include "configgen.h"

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <errno.h>
#endif

using json = nlohmann::json;

ConfigGenerator::ConfigGenerator(ModelImpl* model, Placer* placer)
    : model_(model), placer_(placer) {
    // Generate a directory with model name
    std::string dirName = model_->getName();

    #ifdef _WIN32
        _mkdir(dirName.c_str()); // Windows
    #else
        mkdir(dirName.c_str(), 0777); // Linux/Unix
    #endif

    json j = configGen();
    std::ofstream jsonFile(dirName + "/config.json");
    if (!jsonFile.is_open()) {
        std::cerr << "Error opening JSON file for writing." << std::endl;
        return;
    }
    jsonFile << j.dump(4);
    jsonFile.close();
}

json ConfigGenerator::configGen() {
    json j;
    j["model_name"] = model_->getName();
    j["num_tiles_per_node"] = placer_->getNPTiles();
    j["num_cores_per_tile"] = N_CORES_PER_TILE;
    j["num_mvmus_per_core"] = model_->getModelType() == ModelImpl::INFERENCE
                                  ? N_CONSTANT_MVMUS_PER_CORE
                                  : N_TRAINING_MVMUS_PER_CORE;

    json core_config;
    core_config["dataMem_size"] = REGISTER_FILE_SIZE;
    core_config["storageMem_size"] = N_STORAGE_REGISTERS;

    json core_type_2d = json::array();
    for (unsigned int pTile = 0; pTile < placer_->getNPTiles(); ++pTile) {
        json tile_core_types = json::array();
        unsigned int nCoresOfTile = placer_->getNCoresOfTile(pTile);
        for (unsigned int pCore = 0; pCore < nCoresOfTile; ++pCore) {
            tile_core_types.push_back(placer_->getType(pTile, pCore));
        }
        for (unsigned int i = tile_core_types.size(); i < N_CORES_PER_TILE; ++i) {
            tile_core_types.push_back(0); // Fill with 0 if fewer cores than expected
        }
        core_type_2d.push_back(tile_core_types);
    }
    core_config["core_type"] = core_type_2d;

    j["core_config"] = core_config;

    return j;
}