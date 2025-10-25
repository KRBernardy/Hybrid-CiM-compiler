/*
 *  Copyright (c) 2019 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#include <iostream>
#include <string>
#include <vector>

#include "common.h"

class AbstractTensor {

    protected:

        ModelImpl* model_;
        std::string name_;

        AbstractTensor(ModelImpl* model, std::string name) : model_(model), name_(name) {}

    public:

        ModelImpl* getModel() const { return model_; }
        std::string name() { return name_; }

        std::string printNodeName();
        virtual std::string printNodeStyle();
        virtual std::string printTensorType()=0;

};

class AbstractVector : public AbstractTensor {

    protected:

        unsigned int length_;

        AbstractVector(ModelImpl* model, std::string name, unsigned int length)
            : AbstractTensor(model, name), length_(length) {}

    public:

        void checkCompatibility(AbstractVector* v);

        unsigned int length() const { return length_; }
        unsigned int nTiles() { return (length_ - 1)/MVMU_DIM + 1; }

};


class AbstractMatrix : public AbstractTensor {

    protected:

        unsigned int width_;
        unsigned int height_;

        AbstractMatrix(ModelImpl* model, std::string name, unsigned int width, unsigned int height)
            : AbstractTensor(model, name), width_(width), height_(height) {}

    public:

        unsigned int width() const { return width_; }
        unsigned int height() const { return height_; }

};

class AbstractImagePixelStream : public AbstractTensor {

    protected:

        unsigned int imageWidth_;
        unsigned int imageHeight_;
        unsigned int nChannels_;

    public:

        AbstractImagePixelStream(ModelImpl* model, std::string name, unsigned int imageWidth, unsigned int imageHeight, unsigned int nChannels)
            : AbstractTensor(model, name), imageWidth_(imageWidth), imageHeight_(imageHeight), nChannels_(nChannels) {}

        unsigned int imageWidth() { return imageWidth_; }
        unsigned int imageHeight() { return imageHeight_; }
        unsigned int nChannels() { return nChannels_; }
        unsigned int nTiles() { return (nChannels() - 1)/MVMU_DIM + 1; }

        void checkCompatibility(AbstractImagePixelStream* vs);

        std::string printNodeName();
        virtual std::string printTensorType()=0;

};

class InputVectorTile : public AbstractVector {

    public:

        InputVectorTile(ModelImpl* model, std::string name, unsigned int length) : AbstractVector(model, name, length) { }

        std::string printNodeStyle();
        std::string printTensorType();

};

class InputVectorImpl : public AbstractVector {

    protected:

        std::vector<InputVectorTile*> tiles_;

    public:

        InputVectorImpl(ModelImpl* model, std::string name, unsigned int length);
        ~InputVectorImpl();

        unsigned int nTiles() { return (length_ - 1)/MVMU_DIM + 1; }
        InputVectorTile* getTile(unsigned int t);

        std::string printNodeStyle();
        std::string printTensorType();
        void printNodeAndEdges(std::ostream& fout);

};

class InputImagePixelStreamTile : public AbstractImagePixelStream {

    protected:

        std::vector< std::vector<InputVectorTile*> > stream_;

    public:

        InputImagePixelStreamTile(ModelImpl* model, std::string name, unsigned int imageWidth, unsigned int imageHeight, unsigned int nChannels);
        ~InputImagePixelStreamTile();

        InputVectorTile* get(unsigned int h, unsigned int w);

        std::string printNodeStyle();
        std::string printTensorType();

};

class InputImagePixelStreamImpl : public AbstractImagePixelStream {

    protected:

        std::vector<InputImagePixelStreamTile*> tiles_;

    public:

        InputImagePixelStreamImpl(ModelImpl* model, std::string name, unsigned int imageWidth, unsigned int imageHeight, unsigned int nChannels);
        ~InputImagePixelStreamImpl();

        unsigned int nTiles() { return (nChannels() - 1)/MVMU_DIM + 1; }
        InputImagePixelStreamTile* getTile(unsigned int t);

        std::string printNodeStyle();
        std::string printTensorType();
        void printNodeAndEdges(std::ostream& fout);

};

class VectorImpl : public AbstractVector {

    protected:

        std::vector<ProducerOperation*> tiles_;

    public:

        VectorImpl(ModelImpl* model, unsigned int length);

        unsigned int nTiles() { return (length_ - 1)/MVMU_DIM + 1; }
        ProducerOperation* getTile(unsigned int t);
        void setTile(unsigned int t, ProducerOperation* producer);

        std::string printTensorType();

};

class ConstantVectorTile : public AbstractVector {

    protected:

        ConstantVectorOperation *op_;

    public:

        ConstantVectorTile(ModelImpl* model, unsigned int length);
        ConstantVectorOperation* getOp() { return op_; }

        std::string printTensorType();

};

class ConstantVectorImpl : public AbstractVector {

    protected:
        std::vector<ConstantVectorTile*> tiles_;

    public:
        ConstantVectorImpl(ModelImpl *model, std::string name, unsigned int length);

        unsigned int nTiles() { return (length_ - 1) / MVMU_DIM + 1; }
        ConstantVectorTile *getTile(unsigned int t) { return tiles_[t]; }

        std::string printTensorType();
};

class ImagePixelStreamTile : public AbstractImagePixelStream {

    protected:

        std::vector< std::vector<ProducerOperation*> > stream_;

    public:

        ImagePixelStreamTile(ModelImpl* model, unsigned int imageWidth, unsigned int imageHeight, unsigned int nChannels);

        void add(unsigned int h, unsigned int w, ProducerOperation* vec);
        ProducerOperation* get(unsigned int h, unsigned int w);

        std::string printTensorType(); 

};

class ImagePixelStreamImpl : public AbstractImagePixelStream {

    protected:

        std::vector<ImagePixelStreamTile*> tiles_;

    public:

        ImagePixelStreamImpl(ModelImpl* model, unsigned int imageWidth, unsigned int imageHeight, unsigned int nChannels);
        ~ImagePixelStreamImpl();

        unsigned int nTiles() { return (nChannels() - 1)/MVMU_DIM + 1; }
        ImagePixelStreamTile* getTile(unsigned int t);

        std::string printTensorType();

};

class OutputVectorTile : public AbstractVector {

    public:

        OutputVectorTile(ModelImpl* model, std::string name, unsigned int length) : AbstractVector(model, name, length) { }

        std::string printNodeStyle();
        std::string printTensorType();

};

class OutputVectorImpl : public AbstractVector {

    protected:

        std::vector<OutputVectorTile*> tiles_;

    public:

        OutputVectorImpl(ModelImpl* model, std::string name, unsigned int length);
        ~OutputVectorImpl();

        unsigned int nTiles() { return (length_ - 1)/MVMU_DIM + 1; }
        OutputVectorTile* getTile(unsigned int t);

        std::string printNodeStyle();
        std::string printTensorType();
        void printNodeAndEdges(std::ostream& fout);

};

class OutputImagePixelStreamTile : public AbstractImagePixelStream {

    protected:

        std::vector< std::vector<OutputVectorTile*> > stream_;

    public:

        OutputImagePixelStreamTile(ModelImpl* model, std::string name, unsigned int imageWidth, unsigned int imageHeight, unsigned int nChannels);
        ~OutputImagePixelStreamTile();

        OutputVectorTile* get(unsigned int h, unsigned int w);

        std::string printNodeStyle();
        std::string printTensorType();

};

class OutputImagePixelStreamImpl : public AbstractImagePixelStream {

    protected:

        std::vector<OutputImagePixelStreamTile*> tiles_;

    public:

        OutputImagePixelStreamImpl(ModelImpl* model, std::string name, unsigned int imageWidth, unsigned int imageHeight, unsigned int nChannels);
        ~OutputImagePixelStreamImpl();

        unsigned int nTiles() { return (nChannels() - 1)/MVMU_DIM + 1; }
        OutputImagePixelStreamTile* getTile(unsigned int t);

        std::string printNodeStyle();
        std::string printTensorType();
        void printNodeAndEdges(std::ostream& fout);

};

class ConstantMatrixTile : public AbstractMatrix {

    protected:

        std::vector<MVMOperation*> users_;
        unsigned int storage_type_; // using this to differentiate between different storage types

    public:

        ConstantMatrixTile(ModelImpl* model, std::string name, unsigned int width, unsigned int height, unsigned int storageType) : AbstractMatrix(model, name, width, height), storage_type_(storageType) { }

        void addUser(MVMOperation* user) { users_.push_back(user); }
        unsigned int numUsers() { return users_.size(); }
        MVMOperation* getUser(unsigned int i) { return users_[i]; }
        unsigned int getStorageType() { return storage_type_; }

        std::string printTensorType();

};

class ConstantMatrixImpl : public AbstractMatrix {

    protected:

        std::vector< std::vector<ConstantMatrixTile*> > tiles_;
        unsigned int storage_type_; // using this to differentiate between different storage types
        float input_scale_;
        float output_scale_;
        int input_zero_point_;
        int output_zero_point_;

       public:
        ConstantMatrixImpl(ModelImpl *model, std::string name, unsigned int width, unsigned int height, unsigned int storageType, float activation_scale=1.0f, float weights_scale=1.0f, int activation_zero_point=0, int weights_zero_point=0);
        ~ConstantMatrixImpl();

        unsigned int nHeightTiles() { return (height_ - 1)/MVMU_DIM + 1; }
        unsigned int nWidthTiles() { return (width_ - 1)/MVMU_DIM + 1; }
        ConstantMatrixTile* getTile(unsigned int h, unsigned int w);
        unsigned int getStorageType() { return storage_type_; }
        float getInputScale() { return input_scale_; }
        float getOutputScale() { return output_scale_; }
        int getInputZeroPoint() { return input_zero_point_; }
        int getOutputZeroPoint() { return output_zero_point_; }

        std::string printTensorType();

        void checkCompatibilityForMVM(AbstractVector* v);

};

class ConvolutionalConstantMatrixImpl : public AbstractTensor {

    protected:

        unsigned int kernelWidth_;
        unsigned int kernelHeight_;
        unsigned int nInChannels_;
        unsigned int nOutChannels_;
        unsigned int storage_type_; // using this to differentiate between different storage types
        std::vector< std::vector< std::vector< std::vector<ConstantMatrixTile*> > > > tiles_;
        float input_scale_;
        float output_scale_;
        int input_zero_point_;
        int output_zero_point_;

       public:
        ConvolutionalConstantMatrixImpl(ModelImpl* model, std::string name,
                                        unsigned int kernelWidth, unsigned int kernelHeight,
                                        unsigned int nInChannels, unsigned int nOutChannels,
                                        unsigned int storageType, float activation_scale=1.0f,
                                        float weights_scale=1.0f, int activation_zero_point=0,
                                        int weights_zero_point=0);
        ~ConvolutionalConstantMatrixImpl();

        unsigned int getKernelWidth() { return kernelWidth_; }
        unsigned int getKernelHeight() { return kernelHeight_; }
        unsigned int getNInChannels() { return nInChannels_; }
        unsigned int getNOutChannels() { return nOutChannels_; }
        unsigned int getNInChannelTiles() { return (nInChannels_ - 1)/MVMU_DIM + 1; }
        unsigned int getNOutChannelTiles() { return (nOutChannels_ - 1)/MVMU_DIM + 1; }
        ConstantMatrixTile* getTile(unsigned int kh, unsigned int kw, unsigned int h, unsigned int w);
        void checkCompatibility(AbstractImagePixelStream* vs);
        unsigned int getStorageType() { return storage_type_; }
        float getInputScale() { return input_scale_; }
        float getOutputScale() { return output_scale_; }
        int getInputZeroPoint() { return input_zero_point_; }
        int getOutputZeroPoint() { return output_zero_point_; }

        std::string printTensorType();

};

class TrainingMatrixTile : public AbstractMatrix {

    protected:

        std::vector<TrainingMatrixOperation*> users_;

    public:

        TrainingMatrixTile(ModelImpl* model, std::string name, unsigned int width, unsigned int height) : AbstractMatrix(model, name, width, height) { }

        void addUser(TrainingMatrixOperation* user) { users_.push_back(user); }
        unsigned int numUsers() { return users_.size(); }
        TrainingMatrixOperation* getUser(unsigned int i) { return users_[i]; }

        std::string printTensorType();

};

class TrainingMatrixImpl : public AbstractMatrix {

    protected:

        std::vector< std::vector<TrainingMatrixTile*> > tiles_;

    public:

        TrainingMatrixImpl(ModelImpl* model, std::string name, unsigned int width, unsigned int height);
        ~TrainingMatrixImpl();

        unsigned int nHeightTiles() { return (height_ - 1)/MVMU_DIM + 1; }
        unsigned int nWidthTiles() { return (width_ - 1)/MVMU_DIM + 1; }
        TrainingMatrixTile* getTile(unsigned int h, unsigned int w);

        std::string printTensorType();

        void checkCompatibilityForMVM(AbstractVector* v);
        void checkCompatibilityForMVMTranspose(AbstractVector* v);
        void checkCompatibilityForOuterProductAccumulate(AbstractVector* v1, AbstractVector* v2);

};

class BatchNormParamImpl : public AbstractTensor {

    protected:

        unsigned int nChannels_;
        ConstantVectorImpl* weights_;
        ConstantVectorImpl* biases_;
        ConstantVectorImpl* means_;
        ConstantVectorImpl* variances_;

    public:

        BatchNormParamImpl(ModelImpl* model, std::string name, unsigned int nChannels);
        ~BatchNormParamImpl();

        unsigned int nChannels() { return nChannels_; }
        ConstantVectorImpl *getWeight() { return weights_; }
        ConstantVectorImpl *getBias() { return biases_; }
        ConstantVectorImpl *getMean() { return means_; }
        ConstantVectorImpl *getVariance() { return variances_; }

        void checkCompatibility(AbstractImagePixelStream* vs);

        std::string printTensorType();

};