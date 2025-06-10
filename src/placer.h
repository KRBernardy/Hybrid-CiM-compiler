/*
 *  Copyright (c) 2019 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#include "common.h"

class Placer {

    private:

        ModelImpl* model_;
        Partitioner* partitioner_;

        unsigned int nPTiles_;
        unsigned int nPCores_;
        unsigned int nPMVMUs_;

        std::vector<unsigned int> vtile2ptile_;
        std::vector<unsigned int> vcore2pcore_;
        std::vector<unsigned int> vmvmu2pmvmu_;

        std::vector<std::vector<std::vector<unsigned int>>> pMVMUType_;
        std::vector<std::vector<unsigned int>> pCoreType_;
        std::vector<unsigned int> pTileType_;

        void assignPTiles();
        void assignPCores();
        void assignPMVMUs();

    public:

        Placer(ModelImpl* model, Partitioner* partitioner);

        unsigned int getNPMVMUs() { return nPMVMUs_; }
        unsigned int getNPCores() { return nPCores_; }
        unsigned int getNPTiles() { return nPTiles_; }
        unsigned int getPMVMU(ConstantMatrixTile* tile);
        unsigned int getPTile(ConstantMatrixTile* tile);
        unsigned int getPCore(ConstantMatrixTile* tile);
        unsigned int getPMVMU(TrainingMatrixTile* tile);
        unsigned int getPTile(TrainingMatrixTile* tile);
        unsigned int getPCore(TrainingMatrixTile* tile);
        unsigned int getPMVMU(Operation* op);
        unsigned int getPTile(Operation* op);
        unsigned int getPCore(Operation* op);
        unsigned int getType(unsigned int pTile, unsigned int pCore, unsigned int pMVMU);
        unsigned int getType(unsigned int pTile, unsigned int pCore);
        unsigned int getType(unsigned int pTile);
        unsigned int getNCoresOfTile(unsigned int pTile);
        unsigned int getNMVMVUsOfCore(unsigned int pTile, unsigned int pCore);

        std::string printAssignment(Operation* op);

};

