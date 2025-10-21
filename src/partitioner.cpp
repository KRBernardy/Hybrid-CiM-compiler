/*
 *  Copyright (c) 2019 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#include "partitioner.h"

#include <assert.h>
#include <algorithm>
#include <limits>
#include <climits>
#include <fstream>
#include <sstream>
#include <random>
#include <vector>
#include <map>
#include <queue>

#include "model.h"
#include "operations.h"
#include "puma.h"
#include "tensors.h"

Partitioner::Partitioner(ModelImpl *model, CompilerOptions::GraphPartitioningScheme gp)
    : model_(model), gp_(gp) {
    
    // Step 1: Create matrix tile list and assign to virtual MVMUs
    if(gp_ == CompilerOptions::GP_ROW_MAJOR) {
        CreateMatListInRowMajor();
    } else if(gp_ == CompilerOptions::GP_COL_MAJOR) {
        CreateMatListInColMajor();
    } else if(gp_ == CompilerOptions::GP_RANDOM) {
        CreateMatListRandomly();
    }
    assignMatsToVMVMUs();
    
    // Step 2: Assign MVMUs to cores
    assignMVMUsToVCores();
    
    // Step 3: Assign all operations to cores
    assignOperationsToVCores();
    
    // Step 4: Assign cores to tiles
    if(gp_ == CompilerOptions::GP_ROW_MAJOR || 
       gp_ == CompilerOptions::GP_COL_MAJOR || 
       gp_ == CompilerOptions::GP_RANDOM) {
        assignVTilesInVCoreOrder();
    } else if(gp_ == CompilerOptions::GP_KAHIP) {
        assignVTilesWithKaHIP(); // Not implemented yet
    }
    
    // Step 5: Insert necessary data movement operations
    insertLoadsAndStores();
    insertSendsAndRecives();
    insertInputAndOutput();
    insertCopies();
}

bool Partitioner::isVCoreAssigned(Operation* op) {
    return op2vcore_.count(op) > 0;
}

void Partitioner::assignVCore(Operation* op, unsigned int vCore) {
    assert(!isVCoreAssigned(op) && "Cannot reassign virtual core!");
    op2vcore_[op] = vCore;
    
    // Update core weight for load balancing
    if(vCore < coreWeights_.size()) {
        unsigned int weight = 0;
        if (dynamic_cast<MVMOperation*>(op)) {
            weight = OP_WEIGHT_MVM[getVCoreType(vCore)];
        } else if (dynamic_cast<TrainingMatrixOperation*>(op)) {
            weight = OP_WEIGHT_TRAINING_MVM;
        } else if (dynamic_cast<ALUVectorOperation*>(op)) {
            weight = OP_WEIGHT_ALU;
        } else if (dynamic_cast<LoadOperation*>(op)) {
            weight = OP_WEIGHT_LOAD;
        } else if (dynamic_cast<StoreOperation*>(op)) {
            weight = OP_WEIGHT_STORE;
        } else if (dynamic_cast<SendOperation*>(op)) {
            weight = OP_WEIGHT_SEND;
        } else if (dynamic_cast<ReceiveOperation*>(op)) {
            weight = OP_WEIGHT_RECV;
        } else if (dynamic_cast<CopyOperation*>(op)) {
            weight = OP_WEIGHT_COPY;
        } else if (dynamic_cast<SetImmediateOperation*>(op)) {
            weight = OP_WEIGHT_SETI;
        } else if (dynamic_cast<PseudoInputOperation*>(op)) {
            weight = OP_WEIGHT_INPUT;
        } else if (dynamic_cast<PseudoOutputOperation*>(op)) {
            weight = OP_WEIGHT_OUTPUT;
        } else {
            weight = op->length(); // Fallback to length if type is unknown
        }
        coreWeights_[vCore] += weight;
    }
}

unsigned int Partitioner::getVMVMU(Operation* op) {
    if (MVMOperation* mvmOp = dynamic_cast<MVMOperation*>(op)) {
        return getVMVMU(mvmOp->getMatrix());
    }
    else if (TrainingMatrixOperation* tmatOp = dynamic_cast<TrainingMatrixOperation*>(op)) {
        return getVMVMU(tmatOp->getMatrix());
    }
    assert(false && "Only MVMOperation and TrainingMatrixOperation have VMVMU assignments!");
}

unsigned int Partitioner::getVCore(Operation* op) {
    assert(isVCoreAssigned(op) && "Virtual core not assigned!");
    return op2vcore_[op];
}

unsigned int Partitioner::getVTile(Operation* op) {
    return vcore2vtile_[getVCore(op)];
}

unsigned int Partitioner::getVMVMU(ConstantMatrixTile* tile) {
    assert(cmat2vmvmu_.count(tile) && "Virtual MVMU not assigned!");
    return cmat2vmvmu_[tile];
}

unsigned int Partitioner::getVCore(ConstantMatrixTile* tile) {
    return vmvmu2vcore_[getVMVMU(tile)];
}

unsigned int Partitioner::getVTile(ConstantMatrixTile* tile) {
    return vcore2vtile_[getVCore(tile)];
}

unsigned int Partitioner::getVMVMU(TrainingMatrixTile* tile) {
    assert(tmat2vmvmu_.count(tile) && "Virtual MVMU not assigned!");
    return tmat2vmvmu_[tile];
}

unsigned int Partitioner::getVCore(TrainingMatrixTile* tile) {
    return vmvmu2vcore_[getVMVMU(tile)];
}

unsigned int Partitioner::getVTile(TrainingMatrixTile* tile) {
    return vcore2vtile_[getVCore(tile)];
}

unsigned int Partitioner::getVCore(unsigned int vMVMU) {
    assert(vmvmu2vcore_.count(vMVMU) && "Virtual core not assigned!");
    return vmvmu2vcore_[vMVMU];
}

unsigned int Partitioner::getVTile(unsigned int vCore) {
    assert(vcore2vtile_.count(vCore) && "Virtual tile not assigned!");
    return vcore2vtile_[vCore];
}

void Partitioner::assignMVMUsToVCores() {
    // Initialize core tracking
    unsigned int nMVMUSPerCore = (model_->getModelType() == ModelImpl::INFERENCE) ? 
                                  N_CONSTANT_MVMUS_PER_CORE : N_TRAINING_MVMUS_PER_CORE;
    
    // Reserve virtual cores 0 and 1 for input and output
    nVCores_ = 2;
    vcoreType_.resize(2);
    vcoreType_[0] = vcoreType_[1] = 0;
    
    // Initialize MVMU to core mapping
    vmvmu2vcore_[0] = 0;  // Input MVMU
    vmvmu2vcore_[1] = 1;  // Output MVMU
    
    // Initialize core MVMUs tracking
    coreMVMUs_.resize(N_MAX_CORES);
    coreMVMUs_[0].insert(0);
    coreMVMUs_[1].insert(1);
    
    // Group MVMUs by type
    std::map<unsigned int, std::vector<unsigned int>> mvmusByType;
    for(unsigned int vMVMU = 2; vMVMU < nVMVMUs_; ++vMVMU) {
        mvmusByType[getVMVMUType(vMVMU)].push_back(vMVMU);
    }
    
    // Assign MVMUs to cores, ensuring one type per core
    for(auto& [type, mvmus] : mvmusByType) {
        unsigned int mvmusAssigned = 0;
        while(mvmusAssigned < mvmus.size()) {
            unsigned int vCore = nVCores_++;
            vcoreType_.push_back(type);
            
            // Assign up to nMVMUSPerCore MVMUs to this core
            for(unsigned int i = 0; i < nMVMUSPerCore && mvmusAssigned < mvmus.size(); ++i) {
                unsigned int vMVMU = mvmus[mvmusAssigned];
                vmvmu2vcore_[vMVMU] = vCore;
                coreMVMUs_[vCore].insert(vMVMU);
                mvmusAssigned++;
            }
        }
    }
    
    // Initialize core weights for load balancing
    coreWeights_.resize(nVCores_, 0);
}

unsigned int Partitioner::calculateCommCost(Operation* op, unsigned int vCore) {
    unsigned int commCost = 0;
    
    // Calculate communication cost from operands
    if(ConsumerOperation* consumer = dynamic_cast<ConsumerOperation*>(op)) {
        for(unsigned int o = 0; o < consumer->numOperands(); ++o) {
            ProducerOperation* operand = consumer->getOperand(o);
            if(isVCoreAssigned(operand) && getVCore(operand) != vCore) {
                commCost += operand->length();
            }
        }
    }
    
    // Calculate communication cost to users
    if(ProducerOperation* producer = dynamic_cast<ProducerOperation*>(op)) {
        for(auto u = producer->user_begin(); u != producer->user_end(); ++u) {
            ConsumerOperation* user = *u;
            if(isVCoreAssigned(user) && getVCore(user) != vCore) {
                commCost += producer->length();
            }
        }
    }
    
    return commCost;
}

unsigned int Partitioner::findBestCoreForOperation(Operation* op) {
    unsigned int bestCore = 2;  // Start from first non-reserved core
    int bestScore = INT_MAX;
    
    // This function should only be called for non-matrix operations,
    // as matrix operations are assigned in the first pass.
    assert(dynamic_cast<MVMOperation*>(op) == nullptr && "MVM ops should be pre-assigned!");
    assert(dynamic_cast<TrainingMatrixOperation*>(op) == nullptr && "Training ops should be pre-assigned!");

    unsigned int weight = 0;
    if (dynamic_cast<ALUVectorOperation*>(op)) {
        weight = OP_WEIGHT_ALU;
    } else if (dynamic_cast<LoadOperation*>(op)) {
        weight = OP_WEIGHT_LOAD;
    } else if (dynamic_cast<StoreOperation*>(op)) {
        weight = OP_WEIGHT_STORE;
    } else if (dynamic_cast<SendOperation*>(op)) {
        weight = OP_WEIGHT_SEND;
    } else if (dynamic_cast<ReceiveOperation*>(op)) {
        weight = OP_WEIGHT_RECV;
    } else if (dynamic_cast<CopyOperation*>(op)) {
        weight = OP_WEIGHT_COPY;
    } else if (dynamic_cast<SetImmediateOperation*>(op)) {
        weight = OP_WEIGHT_SETI;
    } else {
        weight = op->length(); // Fallback
    }

    for(unsigned int vCore = 2; vCore < nVCores_; ++vCore) {
        // Calculate score: current load + operation weight
        int score = coreWeights_[vCore] + weight;
        
        // Subtract communication benefit (operations on same core don't need communication)
        unsigned int localCommBenefit = 0;
        
        // Check communication with operands
        if(ConsumerOperation* consumer = dynamic_cast<ConsumerOperation*>(op)) {
            for(unsigned int o = 0; o < consumer->numOperands(); ++o) {
                ProducerOperation* operand = consumer->getOperand(o);
                if(isVCoreAssigned(operand) && getVCore(operand) == vCore) {
                    localCommBenefit += operand->length() * 2;  // Weight local communication benefit
                }
            }
        }
        
        // Check communication with users
        if(ProducerOperation* producer = dynamic_cast<ProducerOperation*>(op)) {
            for(auto u = producer->user_begin(); u != producer->user_end(); ++u) {
                ConsumerOperation* user = *u;
                if(isVCoreAssigned(user) && getVCore(user) == vCore) {
                    localCommBenefit += producer->length() * 2;  // Weight local communication benefit
                }
            }
        }
        
        score -= localCommBenefit;
        
        // Add penalty for remote communication
        score += calculateCommCost(op, vCore);
        
        if(score < bestScore) {
            bestScore = score;
            bestCore = vCore;
        }
    }
    
    return bestCore;
}

void Partitioner::assignOperationsToVCores() {
    // First pass: Assign MVM operations to cores based on their MVMU placement
    for(auto it = model_->op_begin(); it != model_->op_end(); ++it) {
        Operation* op = *it;
        
        if(MVMOperation* mvm = dynamic_cast<MVMOperation*>(op)) {
            ConstantMatrixTile* tile = mvm->getMatrix();
            unsigned int vMVMU = getVMVMU(tile);
            unsigned int vCore = vmvmu2vcore_[vMVMU];
            assignVCore(mvm, vCore);
        } else if(TrainingMatrixOperation* trainOp = dynamic_cast<TrainingMatrixOperation*>(op)) {
            TrainingMatrixTile* tile = trainOp->getMatrix();
            unsigned int vMVMU = getVMVMU(tile);
            unsigned int vCore = vmvmu2vcore_[vMVMU];
            assignVCore(trainOp, vCore);
        }
    }
    
    // Second pass: Use a priority queue to assign remaining operations
    // Using indexed priority queue with decrease-key support
    struct PQEntry {
        int priority;
        int counter;
        Operation* op;
        bool valid;
        
        bool operator<(const PQEntry& other) const {
            if(priority != other.priority) return priority < other.priority;
            return counter > other.counter;  // Tie-breaking by insertion order
        }
    };
    
    std::priority_queue<PQEntry> pq;
    std::set<Operation*> in_queue;
    std::map<Operation*, PQEntry*> entry_finder;
    std::map<Operation*, int> connectivity;
    int counter = 0;

    // Initialize priority queue with all unassigned operations
    for(auto it = model_->op_begin(); it != model_->op_end(); ++it) {
        Operation* op = *it;
        if(!isVCoreAssigned(op)) {
            int conn = 0;
            if(ConsumerOperation* cons = dynamic_cast<ConsumerOperation*>(op)) {
                for(unsigned int o = 0; o < cons->numOperands(); ++o) {
                    if(isVCoreAssigned(cons->getOperand(o))) conn++;
                }
            }
            if(ProducerOperation* prod = dynamic_cast<ProducerOperation*>(op)) {
                for(auto u = prod->user_begin(); u != prod->user_end(); ++u) {
                    if(isVCoreAssigned(*u)) conn++;
                }
            }
            connectivity[op] = conn;
            PQEntry entry = {conn, counter++, op, true};
            pq.push(entry);
            in_queue.insert(*it);
            // Note: We can't store pointer to heap entry, so we track validity differently
        }
    }

    // Process operations from the priority queue
    while(!pq.empty()) {
        PQEntry entry = pq.top();
        pq.pop();
        
        Operation* op = entry.op;

        // Skip if already assigned (stale entry in queue)
        if(isVCoreAssigned(op) || in_queue.find(op) == in_queue.end()) {
            continue;
        }

        // Assign the operation to the best core
        unsigned int bestCore = findBestCoreForOperation(op);
        assignVCore(op, bestCore);
        in_queue.erase(op);

        // Update connectivity and re-insert neighbors with updated priorities
        std::set<Operation*> to_update;
        
        // Collect neighbors to update
        if(ConsumerOperation* cons = dynamic_cast<ConsumerOperation*>(op)) {
            for(unsigned int o = 0; o < cons->numOperands(); ++o) {
                ProducerOperation* operand = cons->getOperand(o);
                if(ProducerOperation* producer = dynamic_cast<ProducerOperation*>(operand)) {
                    if(!isVCoreAssigned(producer) && in_queue.find(producer) != in_queue.end()) {
                        to_update.insert(producer);
                    }
                }
            }
        }
        if(ProducerOperation* prod = dynamic_cast<ProducerOperation*>(op)) {
            for(auto u = prod->user_begin(); u != prod->user_end(); ++u) {
                if(ConsumerOperation* consumer = dynamic_cast<ConsumerOperation*>(*u)) {
                     if(!isVCoreAssigned(consumer) && in_queue.find(consumer) != in_queue.end()) {
                        to_update.insert(consumer);
                    }
                }
            }
        }
        
        // Update priorities (decrease-key by invalidating old and inserting new)
        for(Operation* neighbor : to_update) {
            connectivity[neighbor]++;
            PQEntry new_entry = {connectivity[neighbor], counter++, neighbor, true};
            pq.push(new_entry);
        }
    }
    
}

void Partitioner::assignVTilesInVCoreOrder() {
    for(unsigned int vCore = 0; vCore < nVCores_; ++vCore) {
        vcore2vtile_[vCore] = vCore;
    }
    nVTiles_ = nVCores_;
    
}

void Partitioner::insertLoadsAndStores() {
    // Insert loads and stores for inter-core communication
    for(auto it = model_->op_begin(); it != model_->op_end(); ++it) {
        Operation* op = *it;
        if(ProducerOperation* producer = dynamic_cast<ProducerOperation*>(op)) {
            StoreOperation* store = nullptr;
            std::map<unsigned int, LoadOperation*> loads;
            
            for(auto u = producer->user_begin(); u != producer->user_end(); ) {
                ConsumerOperation* consumer = *u;
                ++u;  // Increment before potential modification
                
                if(getVCore(producer) != getVCore(consumer)) {
                    // Need inter-core communication
                    if(store == nullptr) {
                        store = new StoreOperation(model_, producer);
                        numStores_ += store->length();
                        assignVCore(store, getVCore(producer));
                    }
                    
                    unsigned int consumerCore = getVCore(consumer);
                    if(loads[consumerCore] == nullptr) {
                        loads[consumerCore] = new LoadOperation(model_, store);
                        numLoads_ += loads[consumerCore]->length();
                        assignVCore(loads[consumerCore], consumerCore);
                    }
                    
                    consumer->replaceOperand(producer, loads[consumerCore]);
                }
            }
        }
    }
}

void Partitioner::insertSendsAndRecives() {
    // Insert sends and receives for inter-tile communication
    for(auto it = model_->op_begin(); it != model_->op_end(); ++it) {
        Operation* op = *it;
        if(StoreOperation* store = dynamic_cast<StoreOperation*>(op)) {
            SendOperation* send = nullptr;
            std::map<unsigned int, ReceiveOperation*> receives;
            
            for(auto u = store->user_begin(); u != store->user_end(); ) {
                TileMemoryReadOperation* read = *u;
                ++u;
                if (getVTile(store) != getVTile(read)) {
                    if (receives[getVTile(read)] == NULL) {
                        SendOperation* send = new SendOperation(model_, store);
                        numSends_ += send->length();
                        cloneAssignment(store, send);
                        ReceiveOperation* recv = new ReceiveOperation(model_, send);
                        numReceives_ += recv->length();
                        cloneAssignment(read, recv);
                        receives[getVTile(read)] = recv;
                    }
                    read->replaceSrc(store, receives[getVTile(read)]);
                }
            }
        }
    }
}

void Partitioner::insertInputAndOutput() {
    // Replace pseudo input and output operations
    std::map<InputVectorTile*, std::map<unsigned int, LoadOperation*>> loads;
    std::map<InputVectorTile*, std::map<unsigned int, ReceiveOperation*>> recvs;
    std::map<InputVectorTile*, WriteInputOperation*> inputs;
    for (auto it = model_->op_begin(); it != model_->op_end();) {
        Operation* op = *it;
        ++it;  // op might get removed from the graph
        if (PseudoInputOperation* pseudoInput = dynamic_cast<PseudoInputOperation*>(op)) {
            InputVectorTile* src = pseudoInput->getSrc();
            for (auto u = pseudoInput->user_begin(); u != pseudoInput->user_end();) {
                ConsumerOperation* consumer = *u;
                ++u;  // replaceOperand may remove consumer from pseudoInput's users
                if (loads[src][getVCore(consumer)] == NULL) {
                    if (recvs[src][getVTile(consumer)] == NULL) {
                        if (inputs[src] == NULL) {
                            WriteInputOperation* input = new WriteInputOperation(model_, src);
                            assignVCore(input, 0);
                            inputs[src] = input;
                        }
                        SendOperation* send = new SendOperation(model_, inputs[src]);
                        numSends_ += send->length();
                        cloneAssignment(inputs[src], send);
                        ReceiveOperation* recv = new ReceiveOperation(model_, send);
                        numReceives_ += recv->length();
                        cloneAssignment(consumer, recv);
                        recvs[src][getVTile(consumer)] = recv;
                    }
                    LoadOperation* load = new LoadOperation(model_, recvs[src][getVTile(consumer)]);
                    numLoads_ += load->length();
                    cloneAssignment(consumer, load);
                    loads[src][getVCore(consumer)] = load;
                }
                consumer->replaceOperand(pseudoInput, loads[src][getVCore(consumer)]);
            }
            unlink(pseudoInput);
        } else if (PseudoOutputOperation* pseudoOutput = dynamic_cast<PseudoOutputOperation*>(op)) {
            OutputVectorTile* dst = pseudoOutput->getDst();
            for (unsigned int o = 0; o < pseudoOutput->numOperands(); ++o) {
                ProducerOperation* producer = pseudoOutput->getOperand(o);
                StoreOperation* store = new StoreOperation(model_, producer);
                numStores_ += store->length();
                cloneAssignment(pseudoOutput, store);
                SendOperation* send = new SendOperation(model_, store);
                numSends_ += send->length();
                cloneAssignment(pseudoOutput, send);
                ReceiveOperation* recv = new ReceiveOperation(model_, send);
                numReceives_ += recv->length();
                assignVCore(recv, 1);
                ReadOutputOperation* output = new ReadOutputOperation(model_, recv, dst);
                cloneAssignment(recv, output);
                producer->removeUser(pseudoOutput);
            }
            unlink(pseudoOutput);
        }
    }
}

void Partitioner::insertCopies() {
    // Insert copy operations across producers and consumers that use different
    // register spaces
    for (auto it = model_->op_begin(); it != model_->op_end(); ++it) {
        Operation* op = *it;
        if (ConsumerOperation* consumer = dynamic_cast<ConsumerOperation*>(op)) {
            bool isMatrixOperation = (dynamic_cast<MVMOperation*>(consumer) != NULL) ||
                                     (dynamic_cast<TrainingMatrixOperation*>(consumer) != NULL);
            if (isMatrixOperation) {
                for (unsigned int o = 0; o < consumer->numOperands(); ++o) {
                    ProducerOperation* producer = consumer->getOperand(o);
                    bool producerIsMatrixOperation =
                        (dynamic_cast<MVMOperation*>(producer) != NULL) ||
                        (dynamic_cast<TrainingMatrixOperation*>(producer) != NULL);
                    bool producerHasMultipleUsers = (producer->numUsers() > 1);
                    if (producerIsMatrixOperation || producerHasMultipleUsers) {
                        /*
                         * producerIsMatrixOperation:
                         * Matrix operations write their outputs to reserved output
                         * registers which cannot be read by other matrix operations which
                         * read from reserved input registers. Therefore if a matrix
                         * operation feeds another matrix operation, a copy must be inserted
                         * between them.
                         *
                         * producerHasMultipleUsers:
                         * If a producer has multiple users, it can't be assigned to the
                         * reserved input registers of one matrix operation because the
                         * other users also need to access it. Therefore, a copy is inserted
                         * to the matix operation.
                         */
                        CopyOperation* copy = new CopyOperation(model_, producer);
                        cloneAssignment(consumer, copy);
                        consumer->replaceOperand(producer, copy);
                    }
                }
            }
        }
    }
}

void Partitioner::cloneAssignment(Operation* cloneFrom, Operation* cloneTo) {
    if(isVCoreAssigned(cloneFrom)) {
        assignVCore(cloneTo, getVCore(cloneFrom));
    }
}

void Partitioner::unlink(Operation* op) {
    op2vcore_.erase(op);
    model_->unlink(op);
}

std::string Partitioner::printAssignment(Operation* op) {
    std::stringstream ss;
    if(isVCoreAssigned(op)) {
        ss << "\nvCore = " << getVCore(op);
        ss << ", vTile = " << getVTile(op);
    }
    return ss.str();
}

void Partitioner::printReport(std::ofstream& report) {
    report << "Partitioner Report:" << std::endl;
    report << "  Number of virtual MVMUs: " << nVMVMUs_ << std::endl;
    report << "  Number of virtual cores: " << nVCores_ << std::endl;
    report << "  Number of virtual tiles: " << nVTiles_ << std::endl;
    report << "  Number of loads inserted: " << numLoads_ << std::endl;
    report << "  Number of stores inserted: " << numStores_ << std::endl;
    report << "  Number of sends inserted: " << numSends_ << std::endl;
    report << "  Number of receives inserted: " << numReceives_ << std::endl;
    
    // Print core load distribution
    report << "\nCore Load Distribution:" << std::endl;
    for(unsigned int vCore = 0; vCore < nVCores_; ++vCore) {
        report << "  Core " << vCore << ": " << coreWeights_[vCore] << " operations";
        if(!coreMVMUs_[vCore].empty()) {
            report << " (MVMUs:";
            for(unsigned int mvmu : coreMVMUs_[vCore]) {
                report << " " << mvmu;
            }
            report << ")";
        }
        report << std::endl;
    }
}

// Implement the MVMU assignment functions
void Partitioner::CreateMatListInRowMajor() {
    // Extract all matrix tiles in row major order
    if (model_->getModelType() == ModelImpl::INFERENCE) {
        for (auto m = model_->const_mat_begin(); m != model_->const_mat_end(); ++m) {
            ConstantMatrixImpl* mat = *m;
            for (unsigned int h = 0; h < mat->nHeightTiles(); ++h) {
                for (unsigned int w = 0; w < mat->nWidthTiles(); ++w) {
                    cmatTiles_.push_back(mat->getTile(h, w));
                }
            }
        }
        for (auto m = model_->conv_mat_begin(); m != model_->conv_mat_end(); ++m) {
            ConvolutionalConstantMatrixImpl* mat = *m;
            for (unsigned int kh = 0; kh < mat->getKernelHeight(); ++kh) {
                for (unsigned int kw = 0; kw < mat->getKernelWidth(); ++kw) {
                    for (unsigned int h = 0; h < mat->getNOutChannelTiles(); ++h) {
                        for (unsigned int w = 0; w < mat->getNInChannelTiles(); ++w) {
                            cmatTiles_.push_back(mat->getTile(kh, kw, h, w));
                        }
                    }
                }
            }
        }
        vmvmuType_.resize(cmatTiles_.size() + 2);
    } else if (model_->getModelType() == ModelImpl::TRAINING) {
        for (auto m = model_->train_mat_begin(); m != model_->train_mat_end(); ++m) {
            TrainingMatrixImpl* mat = *m;
            for (unsigned int h = 0; h < mat->nHeightTiles(); ++h) {
                for (unsigned int w = 0; w < mat->nWidthTiles(); ++w) {
                    tmatTiles_.push_back(mat->getTile(h, w));
                }
            }
        }
        vmvmuType_.resize(tmatTiles_.size() + 2);
    }
}

void Partitioner::CreateMatListInColMajor() {
    // Extract all matrix tiles in column major order
    if (model_->getModelType() == ModelImpl::INFERENCE) {
        for (auto m = model_->const_mat_begin(); m != model_->const_mat_end(); ++m) {
            ConstantMatrixImpl* mat = *m;
            for (unsigned int w = 0; w < mat->nWidthTiles(); ++w) {
                for (unsigned int h = 0; h < mat->nHeightTiles(); ++h) {
                    cmatTiles_.push_back(mat->getTile(h, w));
                }
            }
        }
        for (auto m = model_->conv_mat_begin(); m != model_->conv_mat_end(); ++m) {
            ConvolutionalConstantMatrixImpl* mat = *m;
            for (unsigned int kw = 0; kw < mat->getKernelHeight(); ++kw) {
                for (unsigned int kh = 0; kh < mat->getKernelWidth(); ++kh) {
                    for (unsigned int w = 0; w < mat->getNInChannelTiles(); ++w) {
                        for (unsigned int h = 0; h < mat->getNOutChannelTiles(); ++h) {
                            cmatTiles_.push_back(mat->getTile(kh, kw, h, w));
                        }
                    }
                }
            }
        }
        vmvmuType_.resize(cmatTiles_.size() + 2);
    } else if (model_->getModelType() == ModelImpl::TRAINING) {
        for (auto m = model_->train_mat_begin(); m != model_->train_mat_end(); ++m) {
            TrainingMatrixImpl* mat = *m;
            for (unsigned int w = 0; w < mat->nWidthTiles(); ++w) {
                for (unsigned int h = 0; h < mat->nHeightTiles(); ++h) {
                    tmatTiles_.push_back(mat->getTile(h, w));
                }
            }
        }
        vmvmuType_.resize(tmatTiles_.size() + 2);
    }
}

void Partitioner::CreateMatListRandomly() {
    // Extract all matrix tiles in row major order
    if (model_->getModelType() == ModelImpl::INFERENCE) {
        for (auto m = model_->const_mat_begin(); m != model_->const_mat_end(); ++m) {
            ConstantMatrixImpl* mat = *m;
            for (unsigned int h = 0; h < mat->nHeightTiles(); ++h) {
                for (unsigned int w = 0; w < mat->nWidthTiles(); ++w) {
                    cmatTiles_.push_back(mat->getTile(h, w));
                }
            }
        }
        for (auto m = model_->conv_mat_begin(); m != model_->conv_mat_end(); ++m) {
            ConvolutionalConstantMatrixImpl* mat = *m;
            for (unsigned int kh = 0; kh < mat->getKernelHeight(); ++kh) {
                for (unsigned int kw = 0; kw < mat->getKernelWidth(); ++kw) {
                    for (unsigned int h = 0; h < mat->getNOutChannelTiles(); ++h) {
                        for (unsigned int w = 0; w < mat->getNInChannelTiles(); ++w) {
                            cmatTiles_.push_back(mat->getTile(kh, kw, h, w));
                        }
                    }
                }
            }
        }
    } else if (model_->getModelType() == ModelImpl::TRAINING) {
        for (auto m = model_->train_mat_begin(); m != model_->train_mat_end(); ++m) {
            TrainingMatrixImpl* mat = *m;
            for (unsigned int h = 0; h < mat->nHeightTiles(); ++h) {
                for (unsigned int w = 0; w < mat->nWidthTiles(); ++w) {
                    tmatTiles_.push_back(mat->getTile(h, w));
                }
            }
        }
    }

    // Shuffle them randomly
    if (model_->getModelType() == ModelImpl::INFERENCE) {
        std::random_shuffle(cmatTiles_.begin(), cmatTiles_.end());
    } else if (model_->getModelType() == ModelImpl::TRAINING) {
        std::random_shuffle(tmatTiles_.begin(), tmatTiles_.end());
    }
}

void Partitioner::assignMatsToVMVMUs() {
    // Assign constant matrix tiles to virtual MVMUs
    nVMVMUs_ = 2;  // Reserve 0 and 1 for input and output
    vmvmuType_.resize(nVMVMUs_ + 2); // +2 for input and output
    vmvmuType_[0] = vmvmuType_[1] = 0;  // Input and output MVMUs are type 0
    if (model_->getModelType() == ModelImpl::INFERENCE) {
        cmat2vmvmu_.clear();
        for (ConstantMatrixTile* tile : cmatTiles_) {
            unsigned int vMVMU = nVMVMUs_++;
            cmat2vmvmu_[tile] = vMVMU;
            vmvmuType_[vMVMU] = tile->getStorageType();
        }
    } else if (model_->getModelType() == ModelImpl::TRAINING) {
        tmat2vmvmu_.clear();
        for (TrainingMatrixTile* tile : tmatTiles_) {
            unsigned int vMVMU = nVMVMUs_++;
            tmat2vmvmu_[tile] = vMVMU;
        }
    }
}

void Partitioner::assignVTilesWithKaHIP() {
    // Implementation for KaHIP-based tile assignment
    // ... (implementation details)
}

unsigned int Partitioner::getVMVMUType(unsigned int vMVMU) {
    assert(vMVMU < vmvmuType_.size() && "Invalid virtual MVMU!");
    return vmvmuType_[vMVMU];
}

unsigned int Partitioner::getVCoreType(unsigned int vCore) {
    assert(vCore < vcoreType_.size() && "Invalid virtual core!");
    return vcoreType_[vCore];
}

unsigned int Partitioner::getVTileType(unsigned int vTile) {
    assert(vTile < vtileType_.size() && "Invalid virtual tile!");
    return vtileType_[vTile];
}