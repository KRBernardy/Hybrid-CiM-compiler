/*
 *  Copyright (c) 2019 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#ifndef _COMMON_H_
#define _COMMON_H_

#include "puma.h"

/* Constants */
#define MVMU_DIM                        128
#define N_CONSTANT_MVMUS_PER_CORE       6
#define N_TRAINING_MVMUS_PER_CORE       0
#define N_CORES_PER_TILE                8
#define N_MAX_TILES                     32
#define N_MAX_CORES                     (N_CORES_PER_TILE * N_MAX_TILES)
#define MAX_LOAD_STORE_WIDTH            16
#define MAX_SEND_RECV_WIDTH             16
#define N_TRAINING_OPERATIONS           3
#define N_INPUT_REGISTERS               (MVMU_DIM*((N_CONSTANT_MVMUS_PER_CORE >= N_TRAINING_OPERATIONS*N_TRAINING_MVMUS_PER_CORE)?N_CONSTANT_MVMUS_PER_CORE:(N_TRAINING_OPERATIONS*N_TRAINING_MVMUS_PER_CORE)))
#define N_OUTPUT_REGISTERS              N_INPUT_REGISTERS
#define INPUT_REGISTERS_START_ADDRESS   0
#define OUTPUT_REGISTERS_START_ADDRESS  (INPUT_REGISTERS_START_ADDRESS + N_INPUT_REGISTERS)
#define REGISTER_FILE_START_ADDRESS     (OUTPUT_REGISTERS_START_ADDRESS + N_OUTPUT_REGISTERS)
#define REGISTER_FILE_SIZE              8192
#define REGISTERS_PER_CORE              (N_INPUT_REGISTERS + N_OUTPUT_REGISTERS + REGISTER_FILE_SIZE)
#define N_STORAGE_REGISTERS             1024
#define STORAGE_REGISTERS_START_ADDRESS (REGISTER_FILE_START_ADDRESS + REGISTER_FILE_SIZE)

/* Operation Weights for Load Balancing */
#define N_STORAGE_TYPES 10
const unsigned int OP_WEIGHT_MVM[] = {0, 384, 770, 770, 962, 514, 770, 450, 0, 0}; // Weights for MVM op for each storage type
#define OP_WEIGHT_TRAINING_MVM 12
#define OP_WEIGHT_TRAINING_MVM_TRANSPOSE 12
#define OP_WEIGHT_TRAINING_OUTER_PRODUCT 15
#define OP_WEIGHT_ALU 10
#define OP_WEIGHT_LOAD 2
#define OP_WEIGHT_STORE 1
#define OP_WEIGHT_SEND 8
#define OP_WEIGHT_RECV 10
#define OP_WEIGHT_COPY 1
#define OP_WEIGHT_SETI 1
#define OP_WEIGHT_INPUT 0
#define OP_WEIGHT_OUTPUT 0

/* tensors.h */
class AbstractTensor;
class AbstractVector;
class AbstractMatrix;
class AbstractImagePixelStream;
class InputVectorTile;
class InputVectorImpl;
class InputImagePixelStreamImpl;
class VectorImpl;
class ConstantVectorTile;
class ConstantVectorImpl;
class ImagePixelStreamImpl;
class OutputVectorTile;
class OutputVectorImpl;
class OutputImagePixelStreamImpl;
class ConstantMatrixTile;
class ConstantMatrixImpl;
class ConvolutionalConstantMatrixImpl;
class TrainingMatrixTile;
class TrainingMatrixImpl;
class BatchNormParamImpl;

/* operations.h */
class Operation;
class ProducerOperation;
class ConsumerOperation;
class TileMemoryWriteOperation;
class TileMemoryReadOperation;
class InputOperation;
class OutputOperation;
class CoreOperation;
class TileOperation;
class MVMOperation;
class CoalescedMVMSet;
class TrainingMatrixOperation;
class CoalescedTrainingOperationSet;
class ALUVectorOperation;
class SetImmediateOperation;
class CopyOperation;
class LoadOperation;
class StoreOperation;
class SendOperation;
class ReceiveOperation;
class WriteInputOperation;
class ReadOutputOperation;
class VectorRebuildOperation;
class PseudoInputOperation;
class PseudoOutputOperation;
class ConstantVectorOperation;

/* allocator.h */
class CoreAllocator;
class SpillTracker;

/* model.h */
class ModelImpl;

/* partitioner.h */
class Partitioner;

/* placer.h */
class Placer;

/* memalloc.h */
class MemoryAllocator;

/* coalescer.h */
class Coalescer;

/* linearizer.h */
class Linearizer;

/* regalloc.h */
class RegisterAllocator;

/* codegen.h */
class CodeGenerator;

/* configgen.h */
class ConfigGenerator;

/* instance.h */
class ModelInstanceImpl;

#endif

