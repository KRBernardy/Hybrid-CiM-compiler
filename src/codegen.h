/*
 *  Copyright (c) 2019 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#include "common.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class CodeGenerator {

    private:

        ModelImpl* model_;
        Placer* placer_;
        MemoryAllocator* memoryAllocator_;
        Coalescer* coalescer_;
        Linearizer* linearizer_;
        RegisterAllocator* registerAllocator_;

        void codegen();
        json jsonGen();
        std::string codegen(CoalescedMVMSet* coalescedMVMSet);
        std::string codegen(CoalescedTrainingOperationSet* coalescedTrainingOperationSet);
        std::string codegen(MVMOperation* mvm);
        std::string codegen(TrainingMatrixOperation* trainOp);
        std::string codegen(ALUVectorOperation* aluOp);
        std::string codegen(SetImmediateOperation* seti);
        std::string codegen(CopyOperation* copy);
        std::string codegen(LoadOperation* load);
        std::string codegen(StoreOperation* store);
        std::string codegen(SendOperation* send);
        std::string codegen(ReceiveOperation* recv);
        std::string codegen(WriteInputOperation* write);
        std::string codegen(ReadOutputOperation* read);
        std::string codegen(VectorRebuildOperation* rebuild);
        json jsonGen(CoalescedMVMSet *coalescedMVMSet, int tileID, int coreID);
        json jsonGen(CoalescedTrainingOperationSet *coalescedTrainingOperationSet, int tileID, int coreID);
        json jsonGen(MVMOperation *mvm, int tileID, int coreID);
        json jsonGen(TrainingMatrixOperation *trainOp, int tileID, int coreID);
        json jsonGen(ALUVectorOperation *aluOp, int tileID, int coreID);
        json jsonGen(SetImmediateOperation *seti, int tileID, int coreID);
        json jsonGen(CopyOperation *copy, int tileID, int coreID);
        json jsonGen(LoadOperation *load, int tileID, int coreID);
        json jsonGen(StoreOperation *store, int tileID, int coreID);
        json jsonGen(SendOperation *send, int tileID);
        json jsonGen(ReceiveOperation *recv, int tileID);
        json jsonGen(WriteInputOperation *write, int tileID);
        json jsonGen(ReadOutputOperation *read, int tileID);
        json jsonGen(VectorRebuildOperation *rebuild, int tileID, int coreID);

    public:

        CodeGenerator(ModelImpl* model, Placer* placer, MemoryAllocator* memoryAllocator, Coalescer* coalescer, Linearizer* linearizer, RegisterAllocator* registerAllocator);

};

