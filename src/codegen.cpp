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

#include "puma.h"
#include <nlohmann/json.hpp>

#include "coalescer.h"
#include "codegen.h"
#include "linearizer.h"
#include "memalloc.h"
#include "model.h"
#include "operations.h"
#include "placer.h"
#include "regalloc.h"

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <errno.h>
#endif

CodeGenerator::CodeGenerator(ModelImpl* model, Placer* placer, MemoryAllocator* memoryAllocator, Coalescer* coalescer, Linearizer* linearizer, RegisterAllocator* registerAllocator)
    : model_(model), placer_(placer), memoryAllocator_(memoryAllocator), coalescer_(coalescer), linearizer_(linearizer), registerAllocator_(registerAllocator)
{
    //codegen();
    json j = jsonGen();
    std::ofstream jsonFile(model_->getName() + ".json");
    if (!jsonFile.is_open()) {
        std::cerr << "Error opening JSON file for writing." << std::endl;
        return;
    }
    jsonFile << j.dump(4);
    jsonFile.close();
}

using json = nlohmann::json;

void CodeGenerator::codegen() {

    // TODO: Define ABI for laying out the binary
    // Generate a directory with model name
    std::string dirName = model_->getName();

    #ifdef _WIN32
    _mkdir(dirName.c_str()); // Windows
    #else
    mkdir(dirName.c_str(), 0777); // Linux/Unix
    #endif

    for(unsigned int pTile = 0; pTile < placer_->getNPTiles(); ++pTile) {

        // Generate code for the tile
        std::stringstream fileName;
        fileName << dirName << "/" << model_->getName() << "-tile" << pTile << ".puma";
        std::ofstream tileCode;
        tileCode.open(fileName.str());
        std::list<TileOperation*>& tileOperationList = linearizer_->getTileOperationList(pTile);
        for(TileOperation* tileOp : tileOperationList) {
            if(SendOperation* send = dynamic_cast<SendOperation*>(tileOp)) {
                tileCode << codegen(send);
            } else if(ReceiveOperation* recv = dynamic_cast<ReceiveOperation*>(tileOp)) {
                tileCode << codegen(recv);
            } else if(WriteInputOperation* write = dynamic_cast<WriteInputOperation*>(tileOp)) {
                tileCode << codegen(write);
            } else if(ReadOutputOperation* read = dynamic_cast<ReadOutputOperation*>(tileOp)) {
                tileCode << codegen(read);
            } else {
                assert(0 && "Unsupported operation for code generation!");
            }
        }
        tileCode << "halt()" << std::endl;
        tileCode.close();

        // Generate code for each core in the tile
        for(unsigned int pCore = 0; pCore < N_CORES_PER_TILE; ++pCore) {
            std::stringstream fileName;
            fileName << dirName << "/" << model_->getName() << "-tile" << pTile << "-core" << pCore << ".puma";
            std::ofstream coreCode;
            coreCode.open(fileName.str());
            std::list<CoreOperation*>& coreOperationList = linearizer_->getCoreOperationList(pTile, pCore);
            for(CoreOperation* coreOp : coreOperationList) {
                if(MVMOperation* mvm = dynamic_cast<MVMOperation*>(coreOp)) {
                    coreCode << codegen(mvm);
                } else if(TrainingMatrixOperation* trainOp = dynamic_cast<TrainingMatrixOperation*>(coreOp)) {
                    coreCode << codegen(trainOp);
                } else if(ALUVectorOperation* aluOp = dynamic_cast<ALUVectorOperation*>(coreOp)) {
                    coreCode << codegen(aluOp);
                } else if(SetImmediateOperation* seti = dynamic_cast<SetImmediateOperation*>(coreOp)) {
                    coreCode << codegen(seti);
                } else if(CopyOperation* copy = dynamic_cast<CopyOperation*>(coreOp)) {
                    coreCode << codegen(copy);
                } else if(LoadOperation* load = dynamic_cast<LoadOperation*>(coreOp)) {
                    coreCode << codegen(load);
                } else if(StoreOperation* store = dynamic_cast<StoreOperation*>(coreOp)) {
                    coreCode << codegen(store);
                } else if(VectorRebuildOperation* rebuild = dynamic_cast<VectorRebuildOperation*>(coreOp)) {
                    coreCode << codegen(rebuild);
                } else {
                    assert(0 && "Unsupported operation for code generation!");
                }
            }
            coreCode << "hlt()" << std::endl;
            coreCode.close();
        }

    }

}

json CodeGenerator::jsonGen() {
    json j = json::array();
    for (unsigned int pTile = 0; pTile < placer_->getNPTiles(); ++pTile) {
        // Generate JSON for the tile
        std::list<TileOperation *> &tileOperationList = linearizer_->getTileOperationList(pTile);
        for (TileOperation *tileOp : tileOperationList) {
            json temp;
            if (SendOperation *send = dynamic_cast<SendOperation *>(tileOp))
                temp = jsonGen(send, pTile);
            else if (ReceiveOperation *recv = dynamic_cast<ReceiveOperation *>(tileOp))
                temp = jsonGen(recv, pTile);
            else if (WriteInputOperation *write = dynamic_cast<WriteInputOperation *>(tileOp))
                temp = jsonGen(write, pTile);
            else if (ReadOutputOperation *read = dynamic_cast<ReadOutputOperation *>(tileOp))
                temp = jsonGen(read, pTile);
            else
                assert(0 && "Unsupported operation for JSON generation!");
            if (temp != json()) {
                j.push_back(temp);
            }
        }
        json halt;
        halt["type"] = "halt";
        halt["tile"] = pTile;
        j.push_back(halt);

        // Generate json for each core in the tile
        for (unsigned int pCore = 0; pCore < N_CORES_PER_TILE; ++pCore) {
            std::list<CoreOperation *> &coreOperationList = linearizer_->getCoreOperationList(pTile, pCore);
            for (CoreOperation *coreOp : coreOperationList) {
                json temp;
                if (MVMOperation *mvm = dynamic_cast<MVMOperation *>(coreOp))
                    temp = jsonGen(mvm, pTile, pCore);
                else if (TrainingMatrixOperation *trainOp = dynamic_cast<TrainingMatrixOperation *>(coreOp))
                    temp = jsonGen(trainOp, pTile, pCore);
                else if (ALUVectorOperation *aluOp = dynamic_cast<ALUVectorOperation *>(coreOp))
                    temp = jsonGen(aluOp, pTile, pCore);
                else if (SetImmediateOperation *seti = dynamic_cast<SetImmediateOperation *>(coreOp))
                    temp = jsonGen(seti, pTile, pCore);
                else if (CopyOperation *copy = dynamic_cast<CopyOperation *>(coreOp))
                    temp = jsonGen(copy, pTile, pCore);
                else if (LoadOperation *load = dynamic_cast<LoadOperation *>(coreOp))
                    temp = jsonGen(load, pTile, pCore);
                else if (StoreOperation *store = dynamic_cast<StoreOperation *>(coreOp))
                    temp = jsonGen(store, pTile, pCore);
                else if (VectorRebuildOperation *rebuild = dynamic_cast<VectorRebuildOperation *>(coreOp))
                    temp = jsonGen(rebuild, pTile, pCore);
                else
                    assert(0 && "Unsupported operation for JSON generation!");
                if (temp != json()) {
                    j.push_back(temp);
                }
            }
            json hlt;
            hlt["type"] = "hlt";
            hlt["tile"] = pTile;
            hlt["core"] = pCore;
            j.push_back(hlt);
        }
    }
    return j;
}

std::string CodeGenerator::codegen(CoalescedMVMSet *coalescedMVMSet) {
    std::stringstream ss;
    ss << "mvm(['";
    for(unsigned int i = 0; i < N_CONSTANT_MVMUS_PER_CORE; ++i) {
        if(coalescedMVMSet->usesPMVMU(i)) {
            ss << 1;
        } else {
            ss << 0;
        }
    }
    ss << "'])\n";
    return ss.str();
}

json CodeGenerator::jsonGen(CoalescedMVMSet* coalescedMVMSet, int tileID, int coreID) {
    json j;
    j["type"] = "mvm";
    j["tile"] = tileID;
    j["core"] = coreID;
    j["xbar"] = json::array();
    for(unsigned int i = 0; i < N_CONSTANT_MVMUS_PER_CORE; ++i) {
        if (coalescedMVMSet->usesPMVMU(i)) {
            j["xbar"].push_back(i);
        }
    }
    return j;
}

std::string CodeGenerator::codegen(CoalescedTrainingOperationSet* coalescedTrainingOperationSet) {
    std::stringstream ss;
    ss << "train([";
    for(unsigned int pMVMU = 0; pMVMU < N_TRAINING_MVMUS_PER_CORE; ++pMVMU) {
        ss << "'";
        for(unsigned int t = 0; t < N_TRAINING_OPERATIONS; ++t) {
            TrainingMatrixOperation::OpType opType = (TrainingMatrixOperation::OpType)t;
            if(coalescedTrainingOperationSet->usesPMVMUForOp(pMVMU, opType)) {
                ss << 1;
            } else {
                ss << 0;
            }
        }
        ss << "'";
    }
    ss << "])\n";
    return ss.str();
}

std::string CodeGenerator::codegen(MVMOperation* mvm) {
    CoalescedMVMSet* coalescedMVMSet = mvm->getCoalescedSet();
    if(coalescedMVMSet != NULL) {
        if(coalescedMVMSet->isSetLeader(mvm)) { // Only one MVM in a coalesced set does code generation on behalf of the others
            return codegen(coalescedMVMSet);
        } else {
            return "";
        }
    } else {
        std::stringstream ss;
        ss << "mvm(['";
        for(unsigned int i = 0; i < N_CONSTANT_MVMUS_PER_CORE; ++i) {
            if(i == placer_->getPMVMU(mvm)) {
                ss << 1;
            } else {
                ss << 0;
            }
        }
        ss << "'])\n";
        return ss.str();
    }
}

json CodeGenerator::jsonGen(MVMOperation *mvm, int tileID, int coreID) {
    CoalescedMVMSet *coalescedMVMSet = mvm->getCoalescedSet();
    if (coalescedMVMSet != NULL) {
        if (coalescedMVMSet->isSetLeader(mvm)) { // Only one MVM in a coalesced set does code generation on behalf of the others
            return codegen(coalescedMVMSet);
        }
        else {
            return json();
        }
    }
    json j;
    j["type"] = "mvm";
    j["tile"] = tileID;
    j["core"] = coreID;
    j["xbar"] = json::array();
    for (unsigned int i = 0; i < N_CONSTANT_MVMUS_PER_CORE; ++i) {
        if (i == placer_->getPMVMU(mvm)) {
            j["xbar"].push_back(i);
        }
    }
    return j;
}

std::string CodeGenerator::codegen(TrainingMatrixOperation* trainOp) {
    CoalescedTrainingOperationSet* coalescedTrainingOperationSet = trainOp->getCoalescedSet();
    if(coalescedTrainingOperationSet != NULL) {
        if(coalescedTrainingOperationSet->isSetLeader(trainOp)) { // Only one training operation in a coalesced set does code generation on behalf of the others
            return codegen(coalescedTrainingOperationSet);
        } else {
            return "";
        }
    } else {
        std::stringstream ss;
        ss << "train([";
        for(unsigned int pMVMU = 0; pMVMU < N_TRAINING_MVMUS_PER_CORE; ++pMVMU) {
            ss << "'";
            for(unsigned int t = 0; t < N_TRAINING_OPERATIONS; ++t) {
                TrainingMatrixOperation::OpType opType = (TrainingMatrixOperation::OpType)t;
                if(pMVMU == placer_->getPMVMU(trainOp) && opType == trainOp->getOpType()) {
                    ss << 1;
                } else {
                    ss << 0;
                }
            }
        }
        ss << "])\n";
        return ss.str();
    }
}

std::string CodeGenerator::codegen(ALUVectorOperation* aluOp) {
    std::stringstream ss;
    ss << "alu";
    switch(aluOp->getOpCode()) {
        case ALUVectorOperation::MULI:
            ss << "i";
    }
    ss << "('";
    switch(aluOp->getOpCode()) {
        case ALUVectorOperation::ADD: ss << "add"; break;
        case ALUVectorOperation::SUB: ss << "sub"; break;
        case ALUVectorOperation::MUL:
        case ALUVectorOperation::MULI: ss << "mul"; break;
        case ALUVectorOperation::DIV: ss << "div"; break;
        case ALUVectorOperation::AND: ss << "and"; break;
        case ALUVectorOperation::OR: ss << "or"; break;
        case ALUVectorOperation::NOT: ss << "not"; break;
        case ALUVectorOperation::EQ: ss << "eq"; break;
        case ALUVectorOperation::NEQ: ss << "neq"; break;
        case ALUVectorOperation::LT: ss << "lt"; break;
        case ALUVectorOperation::LEQ: ss << "leq"; break;
        case ALUVectorOperation::GT: ss << "gt"; break;
        case ALUVectorOperation::GEQ: ss << "geq"; break;
        case ALUVectorOperation::MIN: ss << "min"; break;
        case ALUVectorOperation::MAX: ss << "max"; break;
        case ALUVectorOperation::MSE: ss << "mse"; break;
        case ALUVectorOperation::SIG: ss << "sig"; break;
        case ALUVectorOperation::TANH: ss << "tanh"; break;
        case ALUVectorOperation::EXP: ss << "exp"; break;
        case ALUVectorOperation::LOG: ss << "log"; break;
        case ALUVectorOperation::RELU: ss << "relu"; break;
        case ALUVectorOperation::RELUD: ss << "relud"; break;
        case ALUVectorOperation::LOG_SOFTMAX: ss << "log_softmax"; break;
        case ALUVectorOperation::LOG_SOFTMAXD: ss << "log_softmaxd"; break;
        case ALUVectorOperation::RNDCMP: ss << "rndcmp"; break;
    }
    ss << "', "
       << "d1=" << registerAllocator_->getRegister(aluOp) << ", "
       << "r1=" << registerAllocator_->getRegister(aluOp->getOperand(0)) << ", ";
    if(aluOp->numOperands() > 1) {
        ss << "r2=" << registerAllocator_->getRegister(aluOp->getOperand(1)) << ", ";
    }
    if(aluOp->isImmediate()) {
        ss << "imm=" << aluOp->getImmediate() << ", ";
    }
    ss << "vec=" << aluOp->length()
       << ")\n";
    return ss.str();
}

json CodeGenerator::jsonGen(ALUVectorOperation *aluOp, int tileID, int coreID) {
    json j;
    j["type"] = "alu";
    j["tile"] = tileID;
    j["core"] = coreID;
    switch (aluOp->getOpCode()) {
    case ALUVectorOperation::ADD: j["opcode"] = "add"; break;
    case ALUVectorOperation::SUB: j["opcode"] = "sub"; break;
    case ALUVectorOperation::MUL:
    case ALUVectorOperation::MULI: j["opcode"] = "mul"; break;
    case ALUVectorOperation::DIV: j["opcode"] = "div"; break;
    case ALUVectorOperation::AND: j["opcode"] = "and"; break;
    case ALUVectorOperation::OR: j["opcode"] = "or"; break;
    case ALUVectorOperation::NOT: j["opcode"] = "not"; break;
    case ALUVectorOperation::EQ: j["opcode"] = "eq"; break;
    case ALUVectorOperation::NEQ: j["opcode"] = "neq"; break;
    case ALUVectorOperation::LT: j["opcode"] = "lt"; break;
    case ALUVectorOperation::LEQ: j["opcode"] = "leq"; break;
    case ALUVectorOperation::GT: j["opcode"] = "gt"; break;
    case ALUVectorOperation::GEQ: j["opcode"] = "geq"; break;
    case ALUVectorOperation::MIN: j["opcode"] = "min"; break;
    case ALUVectorOperation::MAX: j["opcode"] = "max"; break;
    case ALUVectorOperation::MSE: j["opcode"] = "mse"; break;
    case ALUVectorOperation::SIG: j["opcode"] = "sig"; break;
    case ALUVectorOperation::TANH: j["opcode"] = "tanh"; break;
    case ALUVectorOperation::EXP: j["opcode"] = "exp"; break;
    case ALUVectorOperation::LOG: j["opcode"] = "log"; break;
    case ALUVectorOperation::RELU: j["opcode"] = "relu"; break;
    case ALUVectorOperation::RELUD: j["opcode"] = "relud"; break;
    case ALUVectorOperation::LOG_SOFTMAX: j["opcode"] = "log_softmax"; break;
    case ALUVectorOperation::LOG_SOFTMAXD: j["opcode"] = "log_softmaxd"; break;
    case ALUVectorOperation::RNDCMP: j["opcode"] = "rndcmp"; break;
    default: assert(0 && "Unsupported ALU operation!"); break;
    }
    j["dest"] = registerAllocator_->getRegister(aluOp);
    j["read_1"] = registerAllocator_->getRegister(aluOp->getOperand(0));
    if (aluOp->numOperands() > 1) {
        j["read_2"] = registerAllocator_->getRegister(aluOp->getOperand(1));
    }
    if (aluOp->isImmediate()) {
        j["imm"] = aluOp->getImmediate();
    }
    j["vec"] = aluOp->length();
    return j;
}

std::string CodeGenerator::codegen(SetImmediateOperation* seti) {
    std::stringstream ss;
    ss << "set("
       << "d1=" << registerAllocator_->getRegister(seti) << ", "
       << "imm=" << seti->getImmediate() << ", "
       << "vec=" << seti->length() << ", "
       << "is_address=" << seti->isAddress()
       << ")\n";
    return ss.str();
}

json CodeGenerator::jsonGen(SetImmediateOperation *seti, int tileID, int coreID) {
    json j;
    j["type"] = "set";
    j["tile"] = tileID;
    j["core"] = coreID;
    j["dest"] = registerAllocator_->getRegister(seti);
    j["imm"] = seti->getImmediate();
    j["vec"] = seti->length();
    j["is_address"] = seti->isAddress();
    return j;
}

std::string CodeGenerator::codegen(CopyOperation *copy) {
    std::stringstream ss;
    ss << "copy("
       << "d1=" << registerAllocator_->getRegister(copy) << ", "
       << "r1=" << registerAllocator_->getRegister(copy->getOperand(0)) << ", "
       << "vec=" << copy->length() << ", "
       << "src_type=" << 1
       << ")\n";
    return ss.str();
}

json CodeGenerator::jsonGen(CopyOperation *copy, int tileID, int coreID) {
    json j;
    j["type"] = "copy";
    j["tile"] = tileID;
    j["core"] = coreID;
    j["dest"] = registerAllocator_->getRegister(copy);
    j["read"] = registerAllocator_->getRegister(copy->getOperand(0));
    j["vec"] = copy->length();
    return j;
}

std::string CodeGenerator::codegen(LoadOperation *load) {
    std::stringstream ss;
    unsigned int loadWidth;
    for(loadWidth = MAX_LOAD_STORE_WIDTH; !(load->getDataLength()%loadWidth == 0); --loadWidth);
    ss << "load("
       << "d1=" << registerAllocator_->getRegister(load) << ", "
       << "r1=" << registerAllocator_->getRegister(load->getOperand(0)) << ", "
       << "load_width=" << loadWidth << ", "
       << "vec=" << load->getDataLength()/loadWidth
       << ")\n";
    return ss.str();
}

json CodeGenerator::jsonGen(LoadOperation *load, int tileID, int coreID) {
    json j;
    unsigned int loadWidth;
    for (loadWidth = MAX_LOAD_STORE_WIDTH; !(load->getDataLength() % loadWidth == 0); --loadWidth);
    j["type"] = "load";
    j["tile"] = tileID;
    j["core"] = coreID;
    j["dest"] = registerAllocator_->getRegister(load);
    j["read"] = registerAllocator_->getRegister(load->getOperand(0));
    j["width"] = loadWidth;
    j["vec"] = load->getDataLength() / loadWidth;
    return j;
}

std::string CodeGenerator::codegen(StoreOperation *store) {
    std::stringstream ss;
    unsigned int storeWidth;
    for(storeWidth = MAX_LOAD_STORE_WIDTH; !(store->length()%storeWidth == 0); --storeWidth);
    ss << "store(d1=" << registerAllocator_->getRegister(store->getOperand(1)) << ", "
       << "r1=" << registerAllocator_->getRegister(store->getOperand(0)) << ", "
       << "counter=" << store->numUsers() << ", "
       << "store_width=" << storeWidth << ", "
       << "vec=" << store->length()/storeWidth
       << ")\n";
    return ss.str();
}

json CodeGenerator::jsonGen(StoreOperation *store, int tileID, int coreID) {
    json j;
    unsigned int storeWidth;
    for (storeWidth = MAX_LOAD_STORE_WIDTH; !(store->length() % storeWidth == 0); --storeWidth);
    j["type"] = "store";
    j["tile"] = tileID;
    j["core"] = coreID;
    j["dest"] = registerAllocator_->getRegister(store->getOperand(1));
    j["read"] = registerAllocator_->getRegister(store->getOperand(0));
    j["width"] = storeWidth;
    j["vec"] = store->length() / storeWidth;
    return j;
}

std::string CodeGenerator::codegen(SendOperation* send) {
    std::stringstream ss;
    unsigned int sendWidth;
    for(sendWidth = MAX_SEND_RECV_WIDTH; !(send->length()%sendWidth == 0); --sendWidth);
    ss << "send("
       << "mem_addr=" << memoryAllocator_->getTileMemoryAddress(send->getSrc(0)) << ", "
       << "vtile_id=" << placer_->getPTile(send) << ", " // FIXME: Assign sender IDs
       << "send_width=" << sendWidth << ", "
       << "target_addr=" << placer_->getPTile(send->getDst()) << ", "
       << "vec=" << send->length()/sendWidth
       << ")\n";
    return ss.str();
}

json CodeGenerator::jsonGen(SendOperation *send, int tileID) {
    json j;
    unsigned int sendWidth;
    for (sendWidth = MAX_SEND_RECV_WIDTH; !(send->length() % sendWidth == 0); --sendWidth);
    j["type"] = "send";
    j["tile"] = tileID;
    j["mem_addr"] = memoryAllocator_->getTileMemoryAddress(send->getSrc(0));
    j["target_tile"] = placer_->getPTile(send->getDst());
    j["width"] = sendWidth;
    j["vec"] = send->length() / sendWidth;
    return j;
}

std::string CodeGenerator::codegen(ReceiveOperation* recv) {
    std::stringstream ss;
    unsigned int recvWidth;
    for(recvWidth = MAX_SEND_RECV_WIDTH; !(recv->length()%recvWidth == 0); --recvWidth);
    ss << "receive(mem_addr=" << memoryAllocator_->getTileMemoryAddress(recv) << ", "
       << "vtile_id=" << placer_->getPTile(recv->getSrc()) << ", " // FIXME: Assign sender IDs
       << "receive_width=" << recvWidth << ", "
       << "counter=" << recv->numUsers() << ", "
       << "vec=" << recv->length()/recvWidth
       << ")\n";
    return ss.str();
}

json CodeGenerator::jsonGen(ReceiveOperation *recv, int tileID) {
    json j;
    unsigned int recvWidth;
    for (recvWidth = MAX_SEND_RECV_WIDTH; !(recv->length() % recvWidth == 0); --recvWidth);
    j["type"] = "receive";
    j["tile"] = tileID;
    j["mem_addr"] = memoryAllocator_->getTileMemoryAddress(recv);
    j["source_tile"] = placer_->getPTile(recv->getSrc());
    j["width"] = recvWidth;
    j["vec"] = recv->length() / recvWidth;
    return j;
}

std::string CodeGenerator::codegen(WriteInputOperation* write) {
    return "";
}

json CodeGenerator::jsonGen(WriteInputOperation *write, int tileID) {
    json j;
    return j;
}

std::string CodeGenerator::codegen(ReadOutputOperation* read) {
    return "";
}

json CodeGenerator::jsonGen(ReadOutputOperation *read, int tileID) {
    json j;
    return j;
}

std::string CodeGenerator::codegen(VectorRebuildOperation* rebuild) {
    std::stringstream ss;
    for (int i = 0; i < rebuild->numOperands(); ++i) {
        ProducerOperation* src = rebuild->getOperand(i);
        for (auto j = rebuild->getIndexBegin(src), k = rebuild->getPlaceBegin(src); j != rebuild->getIndexEnd(src); ++j, ++k) {
            unsigned int index = *j, place = *k;
            ss << "copy("
               << "d1=" << registerAllocator_->getRegister(rebuild) + place << ", "
               << "r1=" << registerAllocator_->getRegister(src) + index << ", "
               << "vec=" << 1 << ", "
               << "src_type=" << 1
               << ")\n";
        }
    }
    return ss.str();
}

json CodeGenerator::jsonGen(VectorRebuildOperation *rebuild, int tileID, int coreID) {
    json js = json::array();
    for (int i = 0; i < rebuild->numOperands(); ++i) {
        ProducerOperation* src = rebuild->getOperand(i);
        for (auto j = rebuild->getIndexBegin(src), k = rebuild->getPlaceBegin(src); j != rebuild->getIndexEnd(src); ++j, ++k) {
            unsigned int index = *j, place = *k;
            json copy;
            copy["type"] = "copy";
            copy["tile"] = tileID;
            copy["core"] = coreID;
            copy["dest"] = registerAllocator_->getRegister(rebuild) + place;
            copy["read"] = registerAllocator_->getRegister(src) + index;
            copy["vec"] = 1;
            js.push_back(copy);
        }
    }
    return js;
}