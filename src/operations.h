/*
 *  Copyright (c) 2019 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#include <iostream>
#include <set>
#include <string>
#include <vector>
#include <map>

#include "common.h"

class Operation {

    protected:

        ModelImpl* model_;
        unsigned int length_;

        Operation() { }

        Operation(ModelImpl* model, unsigned int length);

    public:

        virtual ~Operation() { }

        ModelImpl* getModel() const { return model_; }
        unsigned int length() const { return length_; }

        std::string printNodeName();
        virtual std::string printNodeStyle();
        virtual std::string printOperationType()=0;
        virtual void printNodeAndEdges(std::ostream& fout);

};

class ProducerOperation : public virtual Operation {

    protected:

        std::set<ConsumerOperation*> users_;

        ProducerOperation() { }

    public:

        void addUser(ConsumerOperation* user) { users_.insert(user); }
        void removeUser(ConsumerOperation* user) { users_.erase(user); }

        typedef std::set<ConsumerOperation*>::iterator user_iterator;
        user_iterator user_begin() { return users_.begin(); }
        user_iterator user_end() { return users_.end(); }
        unsigned int numUsers() { return users_.size(); }

        void printNodeAndEdges(std::ostream& fout);

};

class ConsumerOperation : public virtual Operation {

    protected:

        std::vector<ProducerOperation*> operands_;

        ConsumerOperation(ProducerOperation* op1=NULL, ProducerOperation* op2=NULL);

    public:

        unsigned int numOperands() { return operands_.size(); }
        ProducerOperation* getOperand(unsigned int i) { return operands_[i]; }
        void addOperand(ProducerOperation* op);
        void removeOperand(ProducerOperation* op);
        bool uses(ProducerOperation* op);
        void replaceOperand(ProducerOperation* op, ProducerOperation* replacement);

};

class DataMovementOperation : public virtual Operation {

    protected:

        bool partial_;
        unsigned int start_;
        unsigned int dataLength_;

        DataMovementOperation(bool partial, unsigned int start, unsigned int dataLength);
        DataMovementOperation() : DataMovementOperation(false, 0, 0) { }

    public:
    
        bool isPartial() { return partial_; }
        unsigned int getStart() { return start_; }
        unsigned int getDataLength() { return dataLength_; }

};

class TileMemoryWriteOperation : public virtual DataMovementOperation {

    protected:

        std::set<TileMemoryReadOperation*> users_;

        TileMemoryWriteOperation() { }

    public:

        unsigned int numUsers() { return users_.size(); }
        void addUser(TileMemoryReadOperation* user) { users_.insert(user); }
        void removeUser(TileMemoryReadOperation* user) { users_.erase(user); }

        typedef std::set<TileMemoryReadOperation*>::iterator user_iterator;
        user_iterator user_begin() { return users_.begin(); }
        user_iterator user_end() { return users_.end(); }

        void printNodeAndEdges(std::ostream& fout);

};

class TileMemoryReadOperation : public virtual DataMovementOperation {

    protected:

        std::vector<TileMemoryWriteOperation*> srcs_;

        TileMemoryReadOperation(TileMemoryWriteOperation* src1 = NULL, TileMemoryWriteOperation* src2 = NULL);

    public:

        unsigned int numSrcs() { return srcs_.size(); }
        TileMemoryWriteOperation* getSrc(unsigned int i) { return srcs_[i]; }
        void addSrc(TileMemoryWriteOperation* src);
        void removeSrc(TileMemoryWriteOperation* src);
        void replaceSrc(TileMemoryWriteOperation* old, TileMemoryWriteOperation* replacement);

};

class InputOperation : public virtual Operation {

    protected:

        InputVectorTile* src_;

        InputOperation(InputVectorTile* src);

    public:

        InputVectorTile* getSrc() { return src_; }

        void printNodeAndEdges(std::ostream& fout);

};

class OutputOperation : public virtual Operation {

    protected:

        OutputVectorTile* dst_;

        OutputOperation(OutputVectorTile* dst);

    public:

        OutputVectorTile* getDst() { return dst_; }

        void printNodeAndEdges(std::ostream& fout);

};

class CoreOperation : public virtual Operation {

};

class TileOperation : public virtual Operation {

};

class MVMOperation : public ProducerOperation, public ConsumerOperation, public CoreOperation {

    protected:

        ConstantMatrixTile* mat_;
        CoalescedMVMSet* coalescedSet_;

    public:

        MVMOperation(ModelImpl* model, ConstantMatrixTile* mat, ProducerOperation* src);

        void setCoalescedSet(CoalescedMVMSet* coalescedSet);
        void resetCoalescedSet();
        CoalescedMVMSet* getCoalescedSet() { return coalescedSet_; }

        std::string printNodeStyle();
        std::string printOperationType();
        void printNodeAndEdges(std::ostream& fout) { ProducerOperation::printNodeAndEdges(fout); }

};

class CoalescedMVMSet {

    private:

        std::vector<MVMOperation*> mvms_;

    public:

        CoalescedMVMSet() : mvms_(N_CONSTANT_MVMUS_PER_CORE) { }

        std::vector<MVMOperation*>::iterator begin() { return mvms_.begin(); }
        std::vector<MVMOperation*>::iterator end() { return mvms_.end(); }

        bool usesPMVMU(unsigned int pMVMU) { return mvms_[pMVMU] != NULL; }
        void add(MVMOperation* mvm, unsigned int pMVMU);
        void removeAll();
        bool isSetLeader(MVMOperation* mvm);
        bool isComplete();

};

class TrainingMatrixOperation : public ProducerOperation, public ConsumerOperation, public CoreOperation {

    public:

        enum OpType {
            MVM = 0,        /* Forward pass MVM operation */
            MVM_TRANSPOSE,  /* Backward pass MVM operation with transpose */
            OUTER_PRODUCT   /* Outer product operation for error updates */
        };

    protected:

        TrainingMatrixTile* mat_;
        OpType opType_;
        CoalescedTrainingOperationSet* coalescedSet_;

    public:

        TrainingMatrixOperation(ModelImpl* model, TrainingMatrixTile* mat, OpType opType, ProducerOperation* src1, ProducerOperation* src2=NULL);

        OpType getOpType() { return opType_; }

        void setCoalescedSet(CoalescedTrainingOperationSet* coalescedSet);
        void resetCoalescedSet();
        CoalescedTrainingOperationSet* getCoalescedSet() { return coalescedSet_; }

        std::string printNodeStyle();
        std::string printOperationType();
        void printNodeAndEdges(std::ostream& fout) { ProducerOperation::printNodeAndEdges(fout); }

};

class CoalescedTrainingOperationSet {

    private:

        std::vector<TrainingMatrixOperation*> trainOps_;

    public:

        CoalescedTrainingOperationSet() : trainOps_(N_TRAINING_MVMUS_PER_CORE*N_TRAINING_OPERATIONS) { }

        std::vector<TrainingMatrixOperation*>::iterator begin() { return trainOps_.begin(); }
        std::vector<TrainingMatrixOperation*>::iterator end() { return trainOps_.end(); }

        bool usesPMVMUForOp(unsigned int pMVMU, TrainingMatrixOperation::OpType opType) { return trainOps_[pMVMU*N_TRAINING_OPERATIONS + opType] != NULL; }
        void add(TrainingMatrixOperation* trainOp, unsigned int pMVMU);
        void removeAll();
        bool isSetLeader(TrainingMatrixOperation* trainOp);
        bool isComplete();

};

class ALUVectorOperation : public ProducerOperation, public ConsumerOperation, public CoreOperation {

    public:

        enum OpCode {
            ADD, SUB, MUL, DIV,                                                 /* Arithmetic */
            ADDI, SUBI, MULI, DIVI,                                             /* Arithmetic immediate */
            AND, OR, NOT,                                                       /* Logical */
            EQ, NEQ, LT, LEQ, GT, GEQ,                                          /* Comparison */
            MIN, MAX,                                                           /* Min/Max */
            MSE,                                                                /* Other binary */
            SIG, TANH, EXP, LOG, RELU, RELUD, LOG_SOFTMAX, LOG_SOFTMAXD, RNDCMP /* Nonlinear */
        };

    protected:

        OpCode opCode_;
        float imm_;

    public:

        ALUVectorOperation(ModelImpl* model, OpCode opCode, ProducerOperation* src1=NULL, ProducerOperation* src2=NULL);
        ALUVectorOperation(ModelImpl* model, OpCode opCode, ProducerOperation* src1, float imm);

        OpCode getOpCode() { return opCode_; }
        bool isImmediate() { return opCode_ == ADDI || opCode_ == SUBI || opCode_ == MULI || opCode_ == DIVI; }
        float getImmediate() { return imm_; }

        std::string printNodeStyle();
        std::string printOperationType();
        void printNodeAndEdges(std::ostream& fout) { ProducerOperation::printNodeAndEdges(fout); }

};

class SetImmediateOperation : public ProducerOperation, public CoreOperation {

    protected:

        unsigned int imm_;
        bool isAddress_;

    public:

        SetImmediateOperation(ModelImpl* model, unsigned int imm, unsigned int length=1, bool address=false);

        unsigned int getImmediate() { return imm_; }
        bool isAddress() { return isAddress_; }

        std::string printOperationType();
        void printNodeAndEdges(std::ostream& fout) { ProducerOperation::printNodeAndEdges(fout); }

};

class CopyOperation : public ProducerOperation, public ConsumerOperation, public DataMovementOperation, public CoreOperation {

    public:

        CopyOperation(ModelImpl* model, ProducerOperation* src, bool partial = false, unsigned int start = 0, unsigned int dataLength = 1);

        std::string printOperationType();
        void printNodeAndEdges(std::ostream& fout) { ProducerOperation::printNodeAndEdges(fout); }

};

class LoadOperation : public ProducerOperation, public ConsumerOperation, public TileMemoryReadOperation, public CoreOperation {

    public:

        LoadOperation(ModelImpl* model, TileMemoryWriteOperation* src, bool partial = false, unsigned int start = 0, unsigned int dataLength = 1);

        void addTileMemoryAddressOperand(ProducerOperation* address);

        std::string printNodeStyle();
        std::string printOperationType();
        void printNodeAndEdges(std::ostream& fout) { ProducerOperation::printNodeAndEdges(fout); }

};

class StoreOperation : public ConsumerOperation, public TileMemoryWriteOperation, public CoreOperation {

    public:

        StoreOperation(ModelImpl* model, ProducerOperation* src, bool partial = false, unsigned int start = 0, unsigned int dataLength = 1);

        void addTileMemoryAddressOperand(ProducerOperation* address);

        std::string printNodeStyle();
        std::string printOperationType();
        void printNodeAndEdges(std::ostream& fout) { TileMemoryWriteOperation::printNodeAndEdges(fout); }

};

class SendOperation : public TileMemoryReadOperation, public TileOperation {

    protected:

        ReceiveOperation* dst_;

    public:

        SendOperation(ModelImpl* model, TileMemoryWriteOperation* src, bool partial = false, unsigned int start = 0, unsigned int dataLength = 1);

        ReceiveOperation* getDst() { return dst_; }
        void setDst(ReceiveOperation* dst);

        std::string printNodeStyle();
        std::string printOperationType();
        void printNodeAndEdges(std::ostream& fout);

};

class ReceiveOperation : public TileMemoryWriteOperation, public TileOperation {

    protected:

        SendOperation* src_;

    public:

        ReceiveOperation(ModelImpl* model, SendOperation* src, bool partial = false, unsigned int start = 0, unsigned int dataLength = 1);

        SendOperation* getSrc() { return src_; }

        std::string printNodeStyle();
        std::string printOperationType();
        void printNodeAndEdges(std::ostream& fout) { TileMemoryWriteOperation::printNodeAndEdges(fout); }

};

class WriteInputOperation : public InputOperation, public TileMemoryWriteOperation, public TileOperation {

    public:

        WriteInputOperation(ModelImpl* model, InputVectorTile* src);

        std::string printOperationType();
        void printNodeAndEdges(std::ostream& fout);

};

class ReadOutputOperation : public OutputOperation, public TileMemoryReadOperation, public TileOperation {

    public:

        ReadOutputOperation(ModelImpl* model, TileMemoryWriteOperation* src, OutputVectorTile* dst);

        std::string printOperationType();
        void printNodeAndEdges(std::ostream& fout);

};

class VectorRebuildOperation : public ProducerOperation, public ConsumerOperation, public CoreOperation {

    protected:

        std::set<ProducerOperation*> operandSet_;
        std::map<ProducerOperation*, std::vector<unsigned int>> places_;
        std::map<ProducerOperation*, std::vector<unsigned int>> indices_;

    public:

        VectorRebuildOperation(ModelImpl* model, std::vector<ProducerOperation*>& srcs, std::vector<unsigned int>& indices);

        std::vector<unsigned int>::iterator getPlaceBegin(ProducerOperation* i) { return places_[i].begin(); }
        std::vector<unsigned int>::iterator getPlaceEnd(ProducerOperation* i) { return places_[i].end(); }
        std::vector<unsigned int>::iterator getIndexBegin(ProducerOperation* i) { return indices_[i].begin(); }
        std::vector<unsigned int>::iterator getIndexEnd(ProducerOperation* i) { return indices_[i].end(); }
        void updatePlaceAndIndex(ProducerOperation* producer, LoadOperation* load);

        std::string printOperationType();
        void printNodeAndEdges(std::ostream& fout);

};

/* Psudeo-operations: Not real operations. Will be replaced before code generation. */

class PseudoInputOperation : public InputOperation, public ProducerOperation {

    public:

        PseudoInputOperation(ModelImpl* model, InputVectorTile* src);

        std::string printOperationType();
        void printNodeAndEdges(std::ostream& fout);

};

class PseudoOutputOperation : public OutputOperation, public ConsumerOperation {

    public:

        PseudoOutputOperation(ModelImpl* model, ProducerOperation* src, OutputVectorTile* dst);

        std::string printOperationType();
        void printNodeAndEdges(std::ostream& fout);

};

class ConstantVectorOperation : public ProducerOperation, public CoreOperation {

    protected:

        ConstantVectorTile* vec_;

    public:

        ConstantVectorOperation(ModelImpl* model, ConstantVectorTile* vec);
        ConstantVectorTile* getVector() { return vec_; }

        std::string printOperationType();
        void printNodeAndEdges(std::ostream& fout);

};