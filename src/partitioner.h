/*
 *  Copyright (c) 2019 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#ifndef PARTITIONER_H
#define PARTITIONER_H

#include "common.h"
#include "model.h"

class ConstantMatrixTile;
class TrainingMatrixTile;
class Operation;
class ProducerOperation;
class ConsumerOperation;

class Partitioner {

    public:

        Partitioner(ModelImpl* model, CompilerOptions::GraphPartitioningScheme gp);

        unsigned int getVMVMU(Operation* op);
        unsigned int getVCore(Operation* op);
        unsigned int getVTile(Operation* op);
        unsigned int getVMVMU(ConstantMatrixTile* tile);
        unsigned int getVCore(ConstantMatrixTile* tile);
        unsigned int getVTile(ConstantMatrixTile* tile);
        unsigned int getVMVMU(TrainingMatrixTile* tile);
        unsigned int getVCore(TrainingMatrixTile* tile);
        unsigned int getVTile(TrainingMatrixTile* tile);
        unsigned int getVCore(unsigned int vMVMU);
        unsigned int getVTile(unsigned int vCore);
        unsigned int getVMVMUType(unsigned int vMVMU);
        unsigned int getVCoreType(unsigned int vCore);
        unsigned int getVTileType(unsigned int vTile);
        unsigned int getNVMVMUs() { return nVMVMUs_; }
        unsigned int getNVCores() { return nVCores_; }
        unsigned int getNVTiles() { return nVTiles_; }

        void cloneAssignment(Operation* cloneFrom, Operation* cloneTo);

        std::string printAssignment(Operation* op);
        void printReport(std::ofstream& report);

    private:

        ModelImpl* model_;
        CompilerOptions::GraphPartitioningScheme gp_;

        unsigned int nVMVMUs_;
        unsigned int nVCores_;
        unsigned int nVTiles_;

        std::vector<ConstantMatrixTile*> cmatTiles_;
        std::vector<TrainingMatrixTile*> tmatTiles_;
        
        // Core assignment mappings
        std::map<Operation*, unsigned int> op2vcore_;  // Direct operation to core mapping
        std::map<ConstantMatrixTile*, unsigned int> cmat2vmvmu_;
        std::map<TrainingMatrixTile*, unsigned int> tmat2vmvmu_;
        std::map<unsigned int, unsigned int> vmvmu2vcore_;  // MVMU to core mapping
        std::map<unsigned int, unsigned int> vcore2vtile_;  // Core to tile mapping
        
        // Type tracking
        std::vector<unsigned int> vmvmuType_;
        std::vector<unsigned int> vcoreType_;
        std::vector<unsigned int> vtileType_;
        
        // Load balancing
        std::vector<unsigned int> coreWeights_;  // Track computational load per core
        std::vector<std::set<unsigned int>> coreMVMUs_;  // Track MVMUs assigned to each core

        // Assignment functions
        void CreateMatListInRowMajor();
        void CreateMatListInColMajor();
        void CreateMatListRandomly();
        void assignMatsToVMVMUs();
        void assignMVMUsToVCores();
        void assignOperationsToVCores();
        void assignVTilesInVCoreOrder();
        void assignVTilesWithKaHIP();
        
        // Helper functions
        bool isVCoreAssigned(Operation* op);
        void assignVCore(Operation* op, unsigned int vCore);
        unsigned int calculateCommCost(Operation* op, unsigned int vCore);
        unsigned int findBestCoreForOperation(Operation* op);
        void unlink(Operation* op);

        // Data movement insertion
        void insertLoadsAndStores();
        void insertSendsAndRecives();
        void insertInputAndOutput();
        void insertCopies();

        // Statistics
        unsigned int numLoads_ = 0;
        unsigned int numStores_ = 0;
        unsigned int numSends_ = 0;
        unsigned int numReceives_ = 0;

};

#endif