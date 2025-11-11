/*
 *  Copyright (c) 2019 IMPACT Research Group, University of Illinois.
 *  All rights reserved.
 *
 *  This file is covered by the LICENSE.txt license file in the root directory.
 *
 */

#include <assert.h>
#include <sstream>

#include "model.h"
#include "operations.h"
#include "tensors.h"

void OutputVector::operator=(Vector xparam) {
    VectorImpl* x = xparam.unwrap();
    OutputVectorImpl* y = impl_;
    y->checkCompatibility(x);
    for(unsigned int t = 0; t < x->nTiles(); ++t) {
        ProducerOperation* producer = x->getTile(t);
        OutputVectorTile* output = y->getTile(t);
        new PseudoOutputOperation(producer->getModel(), producer, output);
    }
}

void OutputImagePixelStream::operator=(ImagePixelStream xsparam) {
    ImagePixelStreamImpl* xs = xsparam.unwrap();
    OutputImagePixelStreamImpl* ys = impl_;
    ys->checkCompatibility(xs);
    for(unsigned int t = 0; t < xs->nTiles(); ++t) {
        ImagePixelStreamTile* xsTile = xs->getTile(t);
        OutputImagePixelStreamTile* ysTile = ys->getTile(t);
        // TODO: Convert the following into a single operation with codegened loops
        for(unsigned int h = 0; h < xs->imageHeight(); ++h) {
            for(unsigned int w = 0; w < xs->imageWidth(); ++w) {
                ProducerOperation* x = xsTile->get(h, w);
                OutputVectorTile* y = ysTile->get(h, w);
                new PseudoOutputOperation(x->getModel(), x, y);
            }
        }
    }
}

Vector::Vector(InputVector xparam) {
    InputVectorImpl* x = xparam.unwrap();
    VectorImpl* y = new VectorImpl(x->getModel(), x->length());
    y->checkCompatibility(x);
    for(unsigned int t = 0; t < x->nTiles(); ++t) {
        ProducerOperation* producer = new PseudoInputOperation(x->getModel(), x->getTile(t));
        y->setTile(t, producer);
    }
    impl_ = y;
}

Vector::Vector(ConstantVector xparam) {
    ConstantVectorImpl* x = xparam.unwrap();
    VectorImpl* y = new VectorImpl(x->getModel(), x->length());
    y->checkCompatibility(x);
    for(unsigned int t = 0; t < x->nTiles(); ++t) {
        ProducerOperation* producer = x->getTile(t)->getOp();
        y->setTile(t, producer);
    }
    impl_ = y;
}

ImagePixelStream::ImagePixelStream(InputImagePixelStream xsparam) {
    InputImagePixelStreamImpl* xs = xsparam.unwrap();
    ImagePixelStreamImpl* ys = new ImagePixelStreamImpl(xs->getModel(), xs->imageWidth(), xs->imageHeight(), xs->nChannels());
    ys->checkCompatibility(xs);
    for(unsigned int t = 0; t < xs->nTiles(); ++t) {
        InputImagePixelStreamTile* xsTile = xs->getTile(t);
        ImagePixelStreamTile* ysTile = ys->getTile(t);
        // TODO: Convert the following into a single operation with codegened loops
        for(unsigned int h = 0; h < xs->imageHeight(); ++ h) {
            for(unsigned int w = 0; w < xs->imageWidth(); ++ w) {
                InputVectorTile* x = xsTile->get(h, w);
                ProducerOperation* y = new PseudoInputOperation(x->getModel(), x);
                ysTile->add(h, w, y);
            }
        }
    }
    impl_ = ys;
}

Vector unaryOp(Vector xparam, ALUVectorOperation::OpCode op) {
    VectorImpl* x = xparam.unwrap();
    VectorImpl* y = new VectorImpl(x->getModel(), x->length());
    y->checkCompatibility(x);
    for(unsigned int t = 0; t < x->nTiles(); ++t) {
        ProducerOperation* producer = new ALUVectorOperation(x->getModel(), op, x->getTile(t));
        y->setTile(t, producer);
    }
    return Vector(y);
}

Vector operator~(Vector x) {
    return unaryOp(x, ALUVectorOperation::NOT);
}

Vector sig(Vector x) {
    return unaryOp(x, ALUVectorOperation::SIG);
}

Vector tanh(Vector x) {
    return unaryOp(x, ALUVectorOperation::TANH);
}

Vector exp(Vector x) {
    return unaryOp(x, ALUVectorOperation::EXP);
}

Vector log(Vector x) {
    return unaryOp(x, ALUVectorOperation::LOG);
}

Vector relu(Vector x) {
    return unaryOp(x, ALUVectorOperation::RELU);
}

Vector relud(Vector x) {
    return unaryOp(x, ALUVectorOperation::RELUD);
}

Vector log_softmax(Vector x) {
    return unaryOp(x, ALUVectorOperation::LOG_SOFTMAX);
}

Vector log_softmaxd(Vector x) {
    return unaryOp(x, ALUVectorOperation::LOG_SOFTMAXD);
}

Vector rndcmp(Vector x) {
    return unaryOp(x, ALUVectorOperation::RNDCMP);
}

Vector quant(Vector xparam, float scale, int zero_point = 0) {
    VectorImpl* x = xparam.unwrap();
    VectorImpl* y = new VectorImpl(x->getModel(), x->length());
    y->checkCompatibility(x);
    for(unsigned int t = 0; t < x->nTiles(); ++t) {
        ProducerOperation* scaled = new ALUVectorOperation(x->getModel(), ALUVectorOperation::DIVI, x->getTile(t), scale);
        ProducerOperation* rounded;
        if (zero_point != 0) {
            assert(false && "Quantization with non-zero zero_point is not supported yet.");
        }
        else {
            rounded = new ALUVectorOperation(x->getModel(), ALUVectorOperation::FPtoINT, scaled);
        }
        y->setTile(t, rounded);
    }
    return Vector(y);
}

Vector dequant(Vector xparam, float scale, int zero_point = 0) {
    VectorImpl* x = xparam.unwrap();
    VectorImpl* y = new VectorImpl(x->getModel(), x->length());
    y->checkCompatibility(x);
    for(unsigned int t = 0; t < x->nTiles(); ++t) {
        ProducerOperation* intToFp = new ALUVectorOperation(x->getModel(), ALUVectorOperation::INTtoFP, x->getTile(t));
        ProducerOperation* scaled;
        if (zero_point != 0) {
            assert(false && "Dequantization with non-zero zero_point is not supported yet.");
        }
        else {
            scaled = new ALUVectorOperation(x->getModel(), ALUVectorOperation::MULI, intToFp, scale);
        }
        y->setTile(t, scaled);
    }
    return Vector(y);
}

Vector binaryOp(Vector x1param, Vector x2param, ALUVectorOperation::OpCode op) {
    VectorImpl* x1 = x1param.unwrap();
    VectorImpl* x2 = x2param.unwrap();
    VectorImpl* y = new VectorImpl(x1->getModel(), x1->length());
    y->checkCompatibility(x1);
    y->checkCompatibility(x2);
    for(unsigned int t = 0; t < x1->nTiles(); ++t) {
        ProducerOperation* producer = new ALUVectorOperation(x1->getModel(), op, x1->getTile(t), x2->getTile(t));
        y->setTile(t, producer);
    }
    return Vector(y);
}

Vector operator+(Vector x1, Vector x2) {
    return binaryOp(x1, x2, ALUVectorOperation::FADD);
}

Vector operator-(Vector x1, Vector x2) {
    return binaryOp(x1, x2, ALUVectorOperation::FSUB);
}

Vector operator*(Vector x1, Vector x2) {
    return binaryOp(x1, x2, ALUVectorOperation::FMUL);
}

Vector operator/(Vector x1, Vector x2) {
    return binaryOp(x1, x2, ALUVectorOperation::FDIV);
}

Vector operator&(Vector x1, Vector x2) {
    return binaryOp(x1, x2, ALUVectorOperation::AND);
}

Vector operator|(Vector x1, Vector x2) {
    return binaryOp(x1, x2, ALUVectorOperation::OR);
}

Vector operator==(Vector x1, Vector x2) {
    return binaryOp(x1, x2, ALUVectorOperation::EQ);
}

Vector operator!=(Vector x1, Vector x2) {
    return binaryOp(x1, x2, ALUVectorOperation::NEQ);
}

Vector operator<(Vector x1, Vector x2) {
    return binaryOp(x1, x2, ALUVectorOperation::LT);
}

Vector operator<=(Vector x1, Vector x2) {
    return binaryOp(x1, x2, ALUVectorOperation::LEQ);
}

Vector operator>(Vector x1, Vector x2) {
    return binaryOp(x1, x2, ALUVectorOperation::GT);
}

Vector operator>=(Vector x1, Vector x2) {
    return binaryOp(x1, x2, ALUVectorOperation::GEQ);
}

Vector min(Vector x1, Vector x2) {
    return binaryOp(x1, x2, ALUVectorOperation::FMIN);
}

Vector max(Vector x1, Vector x2) {
    return binaryOp(x1, x2, ALUVectorOperation::FMAX);
}

Vector immediateOp(Vector xparam, float imm, ALUVectorOperation::OpCode op) {
    VectorImpl* x = xparam.unwrap();
    VectorImpl* y = new VectorImpl(x->getModel(), x->length());
    y->checkCompatibility(x);
    for(unsigned int t = 0; t < x->nTiles(); ++t) {
        ProducerOperation* producer = new ALUVectorOperation(x->getModel(), op, x->getTile(t), imm);
        y->setTile(t, producer);
    }
    return Vector(y);
}

Vector operator*(float imm, Vector x) {
    return immediateOp(x, imm, ALUVectorOperation::MULI);
}

ImagePixelStream sig(ImagePixelStream xsparam) {
    ImagePixelStreamImpl* xs = xsparam.unwrap();
    ImagePixelStreamImpl* ys = new ImagePixelStreamImpl(xs->getModel(), xs->imageWidth(), xs->imageHeight(), xs->nChannels());
    ys->checkCompatibility(xs);
    for(unsigned int t = 0; t < xs->nTiles(); ++t) {
        ImagePixelStreamTile* xsTile = xs->getTile(t);
        ImagePixelStreamTile* ysTile = ys->getTile(t);
        // TODO: Convert the following into a single operation with codegened loops
        for(unsigned int h = 0; h < xs->imageHeight(); ++h) {
            for(unsigned int w = 0; w < xs->imageWidth(); ++w) {
                ProducerOperation* x = xsTile->get(h, w);
                ProducerOperation* y = new ALUVectorOperation(x->getModel(), ALUVectorOperation::SIG, x);
                ysTile->add(h, w, y);
            }
        }
    }
    return ImagePixelStream(ys);
}

ImagePixelStream relu(ImagePixelStream xsparam) {
    ImagePixelStreamImpl* xs = xsparam.unwrap();
    ImagePixelStreamImpl* ys = new ImagePixelStreamImpl(xs->getModel(), xs->imageWidth(), xs->imageHeight(), xs->nChannels());
    ys->checkCompatibility(xs);
    for(unsigned int t = 0; t < xs->nTiles(); ++t) {
        ImagePixelStreamTile* xsTile = xs->getTile(t);
        ImagePixelStreamTile* ysTile = ys->getTile(t);
        // TODO: Convert the following into a single operation with codegened loops
        for(unsigned int h = 0; h < xs->imageHeight(); ++h) {
            for(unsigned int w = 0; w < xs->imageWidth(); ++w) {
                ProducerOperation* x = xsTile->get(h, w);
                ProducerOperation* y = new ALUVectorOperation(x->getModel(), ALUVectorOperation::RELU, x);
                ysTile->add(h, w, y);
            }
        }
    }
    return ImagePixelStream(ys);
}

ImagePixelStream maxpool(ImagePixelStream xsparam, unsigned int hspan, unsigned int wspan) {
    ImagePixelStreamImpl* xs = xsparam.unwrap();
    unsigned int ysWidth = (xs->imageWidth() - 1)/wspan + 1;
    unsigned int ysHeight = (xs->imageHeight() - 1)/hspan + 1;
    ImagePixelStreamImpl* ys = new ImagePixelStreamImpl(xs->getModel(), ysWidth, ysHeight, xs->nChannels());
    for(unsigned int t = 0; t < xs->nTiles(); ++t) {
        ImagePixelStreamTile* xsTile = xs->getTile(t);
        ImagePixelStreamTile* ysTile = ys->getTile(t);
        // TODO: Convert the following into a single operation with codegened loops
        ProducerOperation* accum[ysHeight][ysWidth][hspan*wspan];
        for(unsigned int hi = 0; hi < xs->imageHeight(); ++hi) {
            for(unsigned int wi = 0; wi < xs->imageWidth(); ++wi) {
                ProducerOperation* xTile = xsTile->get(hi, wi);
                unsigned int ho = hi/hspan;
                unsigned int hh = hi%hspan;
                unsigned int wo = wi/wspan;
                unsigned int ww = wi%wspan;
                unsigned int accumIdx = hh*wspan + ww;
                if(accumIdx == 0) {
                    accum[ho][wo][accumIdx] = xTile;
                } else {
                    accum[ho][wo][accumIdx] = new ALUVectorOperation(accum[ho][wo][accumIdx - 1]->getModel(), ALUVectorOperation::FMAX, accum[ho][wo][accumIdx - 1], xTile);
                }
                if((hh == hspan - 1 || hi == xs->imageHeight() - 1) && (ww == wspan - 1 || wi == xs->imageWidth() - 1)) {
                    ysTile->add(ho, wo, accum[ho][wo][accumIdx]);
                }
            }
        }
    }
    return ImagePixelStream(ys);
}

ImagePixelStream avgpool(ImagePixelStream xsparam, unsigned int hspan, unsigned int wspan) {
    ImagePixelStreamImpl* xs = xsparam.unwrap();
    unsigned int ysWidth = (xs->imageWidth() - 1)/wspan + 1;
    unsigned int ysHeight = (xs->imageHeight() - 1)/hspan + 1;
    ImagePixelStreamImpl* ys = new ImagePixelStreamImpl(xs->getModel(), ysWidth, ysHeight, xs->nChannels());
    for(unsigned int t = 0; t < xs->nTiles(); ++t) {
        ImagePixelStreamTile* xsTile = xs->getTile(t);
        ImagePixelStreamTile* ysTile = ys->getTile(t);
        // TODO: Convert the following into a single operation with codegened loops
        ProducerOperation* accum[ysHeight][ysWidth][hspan*wspan];
        for(unsigned int hi = 0; hi < xs->imageHeight(); ++hi) {
            for(unsigned int wi = 0; wi < xs->imageWidth(); ++wi) {
                ProducerOperation* xTile = xsTile->get(hi, wi);
                unsigned int ho = hi/hspan;
                unsigned int hh = hi%hspan;
                unsigned int wo = wi/wspan;
                unsigned int ww = wi%wspan;
                unsigned int accumIdx = hh*wspan + ww;
                if(accumIdx == 0) {
                    accum[ho][wo][accumIdx] = xTile;
                } else {
                    accum[ho][wo][accumIdx] = new ALUVectorOperation(accum[ho][wo][accumIdx - 1]->getModel(), ALUVectorOperation::FADD, accum[ho][wo][accumIdx - 1], xTile);
                }
                if((hh == hspan - 1 || hi == xs->imageHeight() - 1) && (ww == wspan - 1 || wi == xs->imageWidth() - 1)) {
                    ysTile->add(ho, wo, new ALUVectorOperation(accum[ho][wo][accumIdx]->getModel(), ALUVectorOperation::MULI, accum[ho][wo][accumIdx], 1.0 / ((hh + 1) * (ww + 1))));
                }
            }
        }
    }
    return ImagePixelStream(ys);
}

ImagePixelStream quant(ImagePixelStream xsparam, float scale, int zero_point = 0) {
    ImagePixelStreamImpl* xs = xsparam.unwrap();
    ImagePixelStreamImpl* ys = new ImagePixelStreamImpl(xs->getModel(), xs->imageWidth(), xs->imageHeight(), xs->nChannels());
    ys->checkCompatibility(xs);
    for(unsigned int t = 0; t < xs->nTiles(); ++t) {
        ImagePixelStreamTile* xsTile = xs->getTile(t);
        ImagePixelStreamTile* ysTile = ys->getTile(t);

        for(unsigned int h = 0; h < xsTile->imageHeight(); ++h) {
            for(unsigned int w = 0; w < xsTile->imageWidth(); ++w) {
                ProducerOperation* xPixel = xsTile->get(h, w);
                ProducerOperation* scaled = new ALUVectorOperation(ysTile->getModel(), ALUVectorOperation::DIVI, xPixel, scale);
                ProducerOperation* yPixel;
                if (zero_point != 0) {
                    assert(false && "Quantization with non-zero zero_point is not supported yet.");
                }
                else {
                    yPixel = new ALUVectorOperation(ysTile->getModel(), ALUVectorOperation::FPtoINT, scaled);
                }
                ysTile->add(h, w, yPixel);
            }
        }
    }
    return ImagePixelStream(ys);
}

ImagePixelStream dequant(ImagePixelStream xsparam, float scale, int zero_point = 0) {
    ImagePixelStreamImpl* xs = xsparam.unwrap();
    ImagePixelStreamImpl* ys = new ImagePixelStreamImpl(xs->getModel(), xs->imageWidth(), xs->imageHeight(), xs->nChannels());
    ys->checkCompatibility(xs);
    for(unsigned int t = 0; t < xs->nTiles(); ++t) {
        ImagePixelStreamTile* xsTile = xs->getTile(t);
        ImagePixelStreamTile* ysTile = ys->getTile(t);

        for(unsigned int h = 0; h < xsTile->imageHeight(); ++h) {
            for(unsigned int w = 0; w < xsTile->imageWidth(); ++w) {
                ProducerOperation* xPixel = xsTile->get(h, w);
                ProducerOperation* intToFp = new ALUVectorOperation(ysTile->getModel(), ALUVectorOperation::INTtoFP, xPixel);
                ProducerOperation* yPixel;
                if (zero_point != 0) {
                    assert(false && "Dequantization with non-zero zero_point is not supported yet.");
                }
                else {
                    yPixel = new ALUVectorOperation(ysTile->getModel(), ALUVectorOperation::MULI, intToFp, scale);
                }
                
                ysTile->add(h, w, yPixel);
            }
        }
    }
    return ImagePixelStream(ys);
}

Vector operator*(ConstantMatrix Mparam, Vector xparam) {
    ConstantMatrixImpl* M = Mparam.unwrap();
    ModelImpl* model = M->getModel();
    Vector quanted_x = quant(xparam, M->getInputScale(), M->getInputZeroPoint());
    VectorImpl* x = quanted_x.unwrap();
    VectorImpl* y = new VectorImpl(model, M->height());
    M->checkCompatibilityForMVM(x);
    std::set<MVMOperation*>* coalesceableMVMSet = new std::set<MVMOperation*>();
    for(unsigned int h = 0; h < y->nTiles(); ++h) {
        ProducerOperation* accum[x->nTiles()]; // TODO: The following implements a sequential reduction; a tree reduction would be more efficient
        for(unsigned int w = 0; w < x->nTiles(); ++w) {
            MVMOperation* mvm = new MVMOperation(model, M->getTile(h, w), x->getTile(w));
            coalesceableMVMSet->insert(mvm);
            if(w == 0) {
                accum[w] = mvm;
            } else {
                accum[w] = new ALUVectorOperation(model, ALUVectorOperation::IADD, mvm, accum[w - 1]);
            }
        }
        y->setTile(h, accum[x->nTiles() - 1]);
    }
    model->addCoalesceableMVMSet(coalesceableMVMSet);
    return dequant(Vector(y), M->getOutputScale(), M->getOutputZeroPoint());
}

ImagePixelStream operator*(ConvolutionalConstantMatrix Mparam, ImagePixelStream xsparam) {
    return conv2d_forward(Mparam, xsparam, 1, 1, 0, 0);
}

ImagePixelStream operator+(ImagePixelStream x1param, ImagePixelStream x2param) {
    ImagePixelStreamImpl* x1 = x1param.unwrap();
    ImagePixelStreamImpl* x2 = x2param.unwrap();
    x1->checkCompatibility(x2);
    ModelImpl* model = x1->getModel();
    ImagePixelStreamImpl* y = new ImagePixelStreamImpl(model, x1->imageWidth(), x1->imageHeight(), x1->nChannels());
    y->checkCompatibility(x1);
    y->checkCompatibility(x2);
    for(unsigned int t = 0; t < x1->nTiles(); ++t) {
        ImagePixelStreamTile* x1Tile = x1->getTile(t);
        ImagePixelStreamTile* x2Tile = x2->getTile(t);
        ImagePixelStreamTile* yTile = y->getTile(t);
        // TODO: Convert the following into a single operation with codegened loops
        for(unsigned int h = 0; h < x1->imageHeight(); ++h) {
            for(unsigned int w = 0; w < x1->imageWidth(); ++w) {
                ProducerOperation* x1Pixel = x1Tile->get(h, w);
                ProducerOperation* x2Pixel = x2Tile->get(h, w);
                ProducerOperation* yPixel = new ALUVectorOperation(model, ALUVectorOperation::FADD, x1Pixel, x2Pixel);
                yTile->add(h, w, yPixel);
            }
        }
    }
    return ImagePixelStream(y);
}

ImagePixelStream conv2d_forward(ConvolutionalConstantMatrix Mparam, ImagePixelStream xsparam, unsigned int stride_x, unsigned int stride_y, unsigned int padding_x, unsigned int padding_y) {
    ConvolutionalConstantMatrixImpl* M = Mparam.unwrap();
    ImagePixelStream quanted_xs = quant(xsparam, M->getInputScale(), M->getInputZeroPoint());
    ImagePixelStreamImpl* xs = quanted_xs.unwrap();
    M->checkCompatibility(xs);
    ModelImpl* model = M->getModel();
    int kernelWidth = M->getKernelWidth();
    int kernelHeight = M->getKernelHeight();
    int nInChannelTiles = M->getNInChannelTiles();
    int nOutChannelTiles = M->getNOutChannelTiles();
    int imageWidth = xs->imageWidth();
    int imageHeight = xs->imageHeight();
    int outImageWidth = (imageWidth + 2 * padding_x - kernelWidth) / stride_x + 1;
    int outImageHeight = (imageHeight + 2 * padding_y - kernelHeight) / stride_y + 1;
    ImagePixelStreamImpl* ys[kernelHeight * kernelWidth * nInChannelTiles];
    for (int kh = 0; kh < kernelHeight; ++kh) { // Instantiates tiles within the same accumulation
        for (int kw = 0; kw < kernelWidth; ++kw) { // Instantiates tiles within the same accumulation
            for (int w = 0; w < nInChannelTiles; ++w) { // Instantiates tiles within the same accumulation
                int accumIdx = (kh * kernelWidth + kw) * nInChannelTiles + w;
                ys[accumIdx] = new ImagePixelStreamImpl(model, outImageWidth, outImageHeight, M->getNOutChannels());
                for(int h = 0; h < nOutChannelTiles; ++h) { // Instantiates independent tiles
                    ConstantMatrixTile* mat = M->getTile(kh, kw, h, w);
                    ImagePixelStreamTile* imageStream = xs->getTile(w);
                    ImagePixelStreamTile* accumStreamIn = (accumIdx == 0)?NULL:ys[accumIdx - 1]->getTile(h); // Partial sum feeding in from previous tile in the same accumulation
                    ImagePixelStreamTile* accumStreamOut = ys[accumIdx]->getTile(h); // Partial sum feeding out to the next tile in the same accumulation
                    // TODO: Convert the following into a single operation with codegened loops
                    for (int ho = 0; ho < outImageHeight; ++ho) {
                        for (int wo = 0; wo < outImageWidth; ++wo) {
                            int hi = ho * stride_y - padding_y + kh;
                            int wi = wo * stride_x - padding_x + kw;
                            bool inputInBounds = hi >= 0
                                                && hi < imageHeight
                                                && wi >= 0
                                                && wi < imageWidth;
                            ProducerOperation* producer;
                            if (inputInBounds) {
                                producer = new MVMOperation(model, mat, imageStream->get(hi, wi));
                                if (accumIdx == 0) {
                                    accumStreamOut->add(ho, wo, producer);
                                } else {
                                    accumStreamOut->add(ho, wo, new ALUVectorOperation(model, ALUVectorOperation::IADD, producer, accumStreamIn->get(ho, wo)));
                                }
                            } else {
                                // Use 0 for input padding
                                if (accumIdx == 0) {
                                    accumStreamOut->add(ho, wo, new SetImmediateOperation(model, 0, mat->height()));
                                } else {
                                    accumStreamOut->add(ho, wo, accumStreamIn->get(ho, wo));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return dequant(ImagePixelStream(ys[kernelHeight * kernelWidth * nInChannelTiles - 1]), M->getOutputScale(), M->getOutputZeroPoint());
}

ImagePixelStream batchnorm(ImagePixelStream xsparam, BatchNormParam bnparam) {
    ImagePixelStreamImpl* xs = xsparam.unwrap();
    BatchNormParamImpl* bn = bnparam.unwrap();
    bn->checkCompatibility(xs);
    ModelImpl* model = xs->getModel();
    ImagePixelStreamImpl* ys = new ImagePixelStreamImpl(model, xs->imageWidth(), xs->imageHeight(), xs->nChannels());
    ConstantVectorImpl* scale = bn->getScale();
    ConstantVectorImpl* shift = bn->getShift();
    for(unsigned int t = 0; t < xs->nTiles(); ++t) {
        ImagePixelStreamTile* xsTile = xs->getTile(t);
        ImagePixelStreamTile* ysTile = ys->getTile(t);
        ConstantVectorOperation *scale_tile = scale->getTile(t)->getOp();
        ConstantVectorOperation *shift_tile = shift->getTile(t)->getOp();
        for (unsigned int h = 0; h < xs->imageHeight(); ++h) {
            for (unsigned int w = 0; w < xs->imageWidth(); ++w) {
                ProducerOperation* x = xsTile->get(h, w);
                ProducerOperation* out_1 = new ALUVectorOperation(model, ALUVectorOperation::FMUL, x, scale_tile);
                ProducerOperation* out_2 = new ALUVectorOperation(model, ALUVectorOperation::FADD, out_1, shift_tile);
                ysTile->add(h, w, out_2);
            }
        }
    }
    return ImagePixelStream(ys);
}

Vector flatten(ImagePixelStream x) {
    ImagePixelStreamImpl* xs = x.unwrap();
    ModelImpl* model = xs->getModel();
    unsigned int flattened_size = xs->imageWidth() * xs->imageHeight() * xs->nChannels();
    unsigned int single_channel_size = xs->imageWidth() * xs->imageHeight();
    VectorImpl* flattened = new VectorImpl(model, flattened_size);

    std::vector<ProducerOperation*> producers;
    std::vector<unsigned int> indices;
    unsigned int index = 0;
    for (unsigned int c = 0; c < xs->nChannels(); ++c) {
        for (unsigned int h = 0; h < xs->imageHeight(); ++h) {
            for (unsigned int w = 0; w < xs->imageWidth(); ++w) {
                ProducerOperation* pixel = xs->getTile(c / MVMU_DIM)->get(h, w);
                producers.push_back(pixel);
                indices.push_back(index / single_channel_size);
                if (producers.size() == MVMU_DIM || index == flattened_size - 1) {
                    ProducerOperation* producer = new VectorRebuildOperation(model, producers, indices);
                    producers.clear();
                    indices.clear();
                    flattened->setTile(index / MVMU_DIM, producer);
                }
                index++;
            }
        }
    }
    return Vector(flattened);
}

ImagePixelStream merge(std::vector<ImagePixelStream>& list) {
    unsigned int len = list.size();
    assert(len > 0);
    for (unsigned int i = 1; i < len; ++i) {
        list[i].unwrap()->checkCompatibility(list[0].unwrap());
    }
    ModelImpl* model = list[0].unwrap()->getModel();
    unsigned int imageWidth = list[0].unwrap()->imageWidth();
    unsigned int imageHeight = list[0].unwrap()->imageHeight();
    unsigned int nChannels_per_image = list[0].unwrap()->nChannels();
    unsigned int nChannels_merged = nChannels_per_image * len;
    unsigned int nTiles_per_image = list[0].unwrap()->nTiles();
    unsigned int nTiles_merged = (nChannels_merged - 1) / MVMU_DIM + 1;
    ImagePixelStreamImpl* ys = new ImagePixelStreamImpl(model, imageWidth, imageHeight, nChannels_merged);
    for (unsigned int h = 0; h < imageHeight; ++h) {
        for (unsigned int w = 0; w < imageWidth; ++w) {
            for (unsigned int t = 0; t < nTiles_merged; ++t) {
                unsigned int length_of_tile;
                if (t == nTiles_merged - 1) {
                    length_of_tile = nChannels_merged % MVMU_DIM;
                } else {
                    length_of_tile = MVMU_DIM;
                }
                std::vector<ProducerOperation*> producers(length_of_tile);
                std::vector<unsigned int> indices(length_of_tile);
                for (unsigned int i = 0; i < length_of_tile; ++i) {
                    unsigned int index = t * MVMU_DIM + i;
                    unsigned int image_index = index / nChannels_per_image;
                    unsigned int tile_index = (index % nChannels_per_image) / MVMU_DIM;
                    unsigned int channel_index = (index % nChannels_per_image) % MVMU_DIM;
                    ImagePixelStreamImpl* xs = list[image_index].unwrap();
                    producers[i] = xs->getTile(tile_index)->get(h, w);
                    indices[i] = channel_index;
                }
                ProducerOperation* producer = new VectorRebuildOperation(model, producers, indices);
                ys->getTile(t)->add(h, w, producer);
            }
        }
    }
    return ImagePixelStream(ys);
}

Vector operator*(TrainingMatrix Mparam, Vector xparam) {
    TrainingMatrixImpl* M = Mparam.unwrap();
    ModelImpl* model = M->getModel();
    VectorImpl* x = xparam.unwrap();
    VectorImpl* y = new VectorImpl(model, M->height());
    M->checkCompatibilityForMVM(x);
    // TODO: Track coalesceable operations
    for(unsigned int h = 0; h < y->nTiles(); ++h) {
        ProducerOperation* accum[x->nTiles()]; // TODO: The following implements a sequential reduction; a tree reduction would be more efficient
        for(unsigned int w = 0; w < x->nTiles(); ++w) {
            TrainingMatrixOperation* trainingOp = new TrainingMatrixOperation(model, M->getTile(h, w), TrainingMatrixOperation::MVM, x->getTile(w));
            if(w == 0) {
                accum[w] = trainingOp;
            } else {
                accum[w] = new ALUVectorOperation(model, ALUVectorOperation::FADD, trainingOp, accum[w - 1]);
            }
        }
        y->setTile(h, accum[x->nTiles() - 1]);
    }
    return Vector(y);
}

Vector operator*(Transpose Mparam, Vector xparam) {
    TrainingMatrixImpl* M = Mparam.unwrap();
    ModelImpl* model = M->getModel();
    VectorImpl* x = xparam.unwrap();
    VectorImpl* y = new VectorImpl(model, M->width());
    M->checkCompatibilityForMVMTranspose(x);
    // TODO: Track coalesceable operations
    for(unsigned int h = 0; h < y->nTiles(); ++h) {
        ProducerOperation* accum[x->nTiles()]; // TODO: The following implements a sequential reduction; a tree reduction would be more efficient
        for(unsigned int w = 0; w < x->nTiles(); ++w) {
            TrainingMatrixOperation* trainingOp = new TrainingMatrixOperation(model, M->getTile(w, h), TrainingMatrixOperation::MVM_TRANSPOSE, x->getTile(w));
            if(w == 0) {
                accum[w] = trainingOp;
            } else {
                accum[w] = new ALUVectorOperation(model, ALUVectorOperation::FADD, trainingOp, accum[w - 1]);
            }
        }
        y->setTile(h, accum[x->nTiles() - 1]);
    }
    return Vector(y);
}

void operator-=(TrainingMatrix Mparam, OuterProduct op) {
    TrainingMatrixImpl* M = Mparam.unwrap();
    ModelImpl* model = M->getModel();
    VectorImpl* x1 = op.unwrap1();
    VectorImpl* x2 = op.unwrap2();
    M->checkCompatibilityForOuterProductAccumulate(x1, x2);
    // TODO: Track coalesceable operations
    for(unsigned int h = 0; h < M->nHeightTiles(); ++h) {
        for(unsigned int w = 0; w < M->nWidthTiles(); ++w) {
            TrainingMatrixOperation* trainingOp = new TrainingMatrixOperation(model, M->getTile(h, w), TrainingMatrixOperation::OUTER_PRODUCT, x1->getTile(h), x2->getTile(w));
        }
    }
}

Operation::Operation(ModelImpl* model, unsigned int length) : model_(model), length_(length) {
    assert(model != NULL);
    model->addOperation(this);
}

ConsumerOperation::ConsumerOperation(ProducerOperation* op1, ProducerOperation* op2) {
    if(op1 != NULL) addOperand(op1);
    if(op2 != NULL) addOperand(op2);
}

DataMovementOperation::DataMovementOperation(bool partial, unsigned int start, unsigned int dataLength)
    : partial_(partial), start_(start), dataLength_(dataLength) {
    assert(!partial || dataLength > 0);
    if (!partial) {
        dataLength_ = length_;
    }
}

TileMemoryReadOperation::TileMemoryReadOperation(TileMemoryWriteOperation* src1, TileMemoryWriteOperation* src2) {
    if (src1 != NULL) addSrc(src1);
    if (src2 != NULL) addSrc(src2);
}

InputOperation::InputOperation(InputVectorTile* src) : src_(src) {
    assert(src != NULL);
}

OutputOperation::OutputOperation(OutputVectorTile* dst) : dst_(dst) {
    assert(dst != NULL);
}

MVMOperation::MVMOperation(ModelImpl* model, ConstantMatrixTile* mat, ProducerOperation* op) : Operation(model, mat->height()), ConsumerOperation(op), mat_(mat), coalescedSet_(NULL) {
    assert(mat != NULL && op != NULL && mat->width() == op->length());
    assert(mat->width() <= MVMU_DIM && mat->height() <= MVMU_DIM && "MVM operations larger than one MVMU are not supported");
    mat->addUser(this);
}

TrainingMatrixOperation::TrainingMatrixOperation(ModelImpl* model, TrainingMatrixTile* mat, OpType opType, ProducerOperation* src1, ProducerOperation* src2) : Operation(model, (opType != MVM_TRANSPOSE)?(mat->height()):(mat->width())), ConsumerOperation(src1, src2), mat_(mat), opType_(opType), coalescedSet_(NULL) {
    assert(mat != NULL && src1 != NULL);
    assert(mat->width() <= MVMU_DIM && mat->height() <= MVMU_DIM && "MVM operations larger than one MVMU are not supported");
    if(opType == MVM) {
        assert(mat->width() == src1->length());
        assert(src2 == NULL);
    } else if(opType == MVM_TRANSPOSE) {
        assert(mat->height() == src1->length());
        assert(src2 == NULL);
    } else if(opType == OUTER_PRODUCT) {
        assert(mat->height() == src1->length());
        assert(src2 != NULL && mat->width() == src2->length());
    } else {
        assert(0 && "Invalid operation type!");
    }
    mat->addUser(this);
}

ALUVectorOperation::ALUVectorOperation(ModelImpl* model, OpCode opCode, ProducerOperation* src1, ProducerOperation* src2) : Operation(model, src1->length()), ConsumerOperation(src1, src2), opCode_(opCode), imm_(0.0f) {
    if (isImmediateOp()) {
        assert(src2 != NULL);
        assert(src2->length() == 1);
    }
    assert(src1 != NULL);
    switch(opCode_) {
        case FADD:
        case FSUB:
        case FMUL:
        case FDIV:
        case FMIN:
        case FMAX:
        case IADD:
        case ISUB:
        case IMUL:
        case IDIV:
        case IMIN:
        case IMAX:
        case AND:
        case OR:
        case XOR:
        case EQ:
        case NEQ:
        case LT:
        case LEQ:
        case GT:
        case GEQ:
            assert(src2 != NULL && src1->length() == src2->length());
    }
}

ALUVectorOperation::ALUVectorOperation(ModelImpl* model, OpCode opCode, ProducerOperation* src1, float imm) : Operation(model, src1->length()), ConsumerOperation(src1), opCode_(opCode), imm_(imm) {
    assert(isImmediateOp());
    assert(src1 != NULL);
}

SetImmediateOperation::SetImmediateOperation(ModelImpl* model, unsigned int imm, unsigned int length, bool address) : Operation(model, length), imm_(imm), isAddress_(address) {
}

CopyOperation::CopyOperation(ModelImpl* model, ProducerOperation* src, bool partial, unsigned int start, unsigned int dataLength)
    : Operation(model, src->length()), ConsumerOperation(src), DataMovementOperation(partial, start, dataLength) {
    assert(src != NULL);
}

LoadOperation::LoadOperation(ModelImpl* model, TileMemoryWriteOperation* src, bool partial, unsigned int start, unsigned int dataLength)
    : Operation(model, src->length()), TileMemoryReadOperation(src), DataMovementOperation(partial, start, dataLength) {
}

StoreOperation::StoreOperation(ModelImpl* model, ProducerOperation* src, bool partial, unsigned int start, unsigned int dataLength)
    : Operation(model, src->length()), ConsumerOperation(src), DataMovementOperation(partial, start, dataLength) {
    assert(src != NULL);
}

SendOperation::SendOperation(ModelImpl* model, TileMemoryWriteOperation* src, bool partial, unsigned int start, unsigned int dataLength)
    : Operation(model, src->length()), TileMemoryReadOperation(src), dst_(NULL), DataMovementOperation(partial, start, dataLength) {
}

ReceiveOperation::ReceiveOperation(ModelImpl* model, SendOperation* src, bool partial, unsigned int start, unsigned int dataLength)
    : Operation(model, src->length()), src_(src), DataMovementOperation(partial, start, dataLength) {
    src->setDst(this);
}

WriteInputOperation::WriteInputOperation(ModelImpl* model, InputVectorTile* src)
    : Operation(model, src->length()), InputOperation(src) {
}

ReadOutputOperation::ReadOutputOperation(ModelImpl* model, TileMemoryWriteOperation* src, OutputVectorTile* dst)
    : Operation(model, src->length()), TileMemoryReadOperation(src), OutputOperation(dst) {
    assert(src->length() == dst->length());
}

VectorRebuildOperation::VectorRebuildOperation(ModelImpl* model, std::vector<ProducerOperation*>& srcs, std::vector<unsigned int>& indices)
    : Operation(model, srcs.size()), ConsumerOperation() {
    assert(srcs.size() == indices.size());
    assert(srcs.size() <= MVMU_DIM && "Rebuild operations larger than one MVMU are not supported");
    for(unsigned int i = 0; i < srcs.size(); ++i) {
        assert(srcs[i] != NULL);
        if (!operandSet_.count(srcs[i])) {
            addOperand(srcs[i]);
            operandSet_.insert(srcs[i]);
            indices_[srcs[i]] = std::vector<unsigned int>();
            places_[srcs[i]] = std::vector<unsigned int>();
        }
        indices_[srcs[i]].push_back(indices[i]);
        places_[srcs[i]].push_back(i);
    }
}

void VectorRebuildOperation::updatePlaceAndIndex(ProducerOperation* producer, LoadOperation* load) {
    assert(operandSet_.count(producer));
    places_[load] = std::vector<unsigned int>(places_[producer].size());
    indices_[load] = std::vector<unsigned int>(indices_[producer].size());
    for (int i = 0; i < places_[producer].size(); ++i) {
        places_[load][i] = places_[producer][i];
        indices_[load][i] = indices_[producer][i] - load->getStart();
    }
    places_.erase(producer);
    indices_.erase(producer);
}

PseudoInputOperation::PseudoInputOperation(ModelImpl* model, InputVectorTile* src) : Operation(model, src->length()), InputOperation(src) {
}

PseudoOutputOperation::PseudoOutputOperation(ModelImpl* model, ProducerOperation* op, OutputVectorTile* dst) : Operation(model, op->length()), ConsumerOperation(op), OutputOperation(dst) {
    assert(op != NULL && op->length() == dst->length());
}

ConstantVectorOperation::ConstantVectorOperation(ModelImpl* model, ConstantVectorTile* src) : Operation(model, src->length()), vec_(src) {
    assert(src != NULL);
}

void LoadOperation::addTileMemoryAddressOperand(ProducerOperation* address) {
    assert(operands_.size() == 0 && "Cannot set tile memory address operand!");
    assert(address->length() == 1 && "Address must be of length 1!");
    operands_.push_back(address);
    address->addUser(this);
}

void StoreOperation::addTileMemoryAddressOperand(ProducerOperation* address) {
    assert(operands_.size() == 1 && "Cannot set tile memory address operand!");
    assert(address->length() == 1 && "Address must be of length 1!");
    operands_.push_back(address);
    address->addUser(this);
}

void SendOperation::setDst(ReceiveOperation* dst) {
    assert(dst_ == NULL && "Cannot reset destination of send operation");
    dst_ = dst;
}

void ConsumerOperation::addOperand(ProducerOperation* op) {
    assert(op != NULL);
    operands_.push_back(op);
    op->addUser(this);
}

void ConsumerOperation::removeOperand(ProducerOperation* op) {
    for(unsigned int i = 0; i < operands_.size(); ++i) {
        if(operands_[i] == op) {
            operands_.erase(operands_.begin() + i);
            op->removeUser(this);
            return;
        }
    }
    assert(0 && "Operand to be removed not found");
}

bool ConsumerOperation::uses(ProducerOperation* op) {
    for(unsigned int i = 0; i < operands_.size(); ++i) {
        if(operands_[i] == op) {
            return true;
        }
    }
    return false;
}

void ConsumerOperation::replaceOperand(ProducerOperation* op, ProducerOperation* replacement) {
    for(unsigned int i = 0; i < operands_.size(); ++i) {
        if(operands_[i] == op) {
            operands_[i] = replacement;
            op->removeUser(this);
            replacement->addUser(this);
        }
    }
}

void TileMemoryReadOperation::addSrc(TileMemoryWriteOperation* src) {
    assert(src != NULL);
    srcs_.push_back(src);
    src->addUser(this);
}

void TileMemoryReadOperation::removeSrc(TileMemoryWriteOperation* src) {
    for(unsigned int i = 0; i < srcs_.size(); ++i) {
        if(srcs_[i] == src) {
            srcs_.erase(srcs_.begin() + i);
            src->removeUser(this);
            return;
        }
    }
    assert(0 && "Source to be removed not found");
}

void TileMemoryReadOperation::replaceSrc(TileMemoryWriteOperation* old, TileMemoryWriteOperation* replacement) {
    for(unsigned int i = 0; i < srcs_.size(); ++i) {
        if(srcs_[i] == old) {
            srcs_[i] = replacement;
            old->removeUser(this);
            replacement->addUser(this);
            return;
        }
    }
    assert(0 && "Source to be replaced not found");
}

void MVMOperation::setCoalescedSet(CoalescedMVMSet* coalescedSet) {
    assert(coalescedSet_ == NULL && "Cannot reassign coalesced set");
    coalescedSet_ = coalescedSet;
}

void MVMOperation::resetCoalescedSet() {
    coalescedSet_ = NULL;
}

void CoalescedMVMSet::add(MVMOperation* mvm, unsigned int pMVMU) {
    assert(mvms_[pMVMU] == NULL);
    mvms_[pMVMU] = mvm;
    mvm->setCoalescedSet(this);
}

void CoalescedMVMSet::removeAll() {
    for(unsigned int i = 0; i < mvms_.size(); ++i) {
        MVMOperation* mvm = mvms_[i];
        if(mvm != NULL) {
            mvms_[i] = NULL;
            mvm->resetCoalescedSet();
        }
    }
}

bool CoalescedMVMSet::isSetLeader(MVMOperation* mvm) {
    for(unsigned int i = 0; i < mvms_.size(); ++i) {
        MVMOperation* m = mvms_[i];
        if(m != NULL) {
            return (m == mvm); // Leader is first non-NULL MVM in the set
        }
    }
    assert(0 && "Unreachable: cannot have caolesced set with all NULL mvms!");
}

bool CoalescedMVMSet::isComplete() {
    for(auto mvm : mvms_) {
        if(mvm == NULL) {
            return false;
        }
    }
    return true;
}

void TrainingMatrixOperation::setCoalescedSet(CoalescedTrainingOperationSet* coalescedSet) {
    assert(coalescedSet_ == NULL && "Cannot reassign coalesced set");
    coalescedSet_ = coalescedSet;
}

void TrainingMatrixOperation::resetCoalescedSet() {
    coalescedSet_ = NULL;
}

void CoalescedTrainingOperationSet::add(TrainingMatrixOperation* trainOp, unsigned int pMVMU) {
    unsigned int index = pMVMU*N_TRAINING_OPERATIONS + trainOp->getOpType();
    assert(trainOps_[index] == NULL);
    trainOps_[index] = trainOp;
    trainOp->setCoalescedSet(this);
}

void CoalescedTrainingOperationSet::removeAll() {
    for(unsigned int i = 0; i < trainOps_.size(); ++i) {
        TrainingMatrixOperation* trainOp = trainOps_[i];
        if(trainOp != NULL) {
            trainOps_[i] = NULL;
            trainOp->resetCoalescedSet();
        }
    }
}

bool CoalescedTrainingOperationSet::isSetLeader(TrainingMatrixOperation* trainOp) {
    for(unsigned int i = 0; i < trainOps_.size(); ++i) {
        TrainingMatrixOperation* t = trainOps_[i];
        if(t != NULL) {
            return (t == trainOp); // Leader is first non-NULL MVM in the set
        }
    }
    assert(0 && "Unreachable: cannot have caolesced set with all NULL mvms!");
}

bool CoalescedTrainingOperationSet::isComplete() {
    for(auto trainOp : trainOps_) {
        if(trainOp == NULL) {
            return false;
        }
    }
    return true;
}

std::string Operation::printNodeName() {
    std::stringstream ss;
    ss << '"' << printOperationType() << "\n" << this << model_->printAssignment(this) << '"';
    return ss.str();
}

std::string Operation::printNodeStyle() {
    return "";
}

std::string MVMOperation::printNodeStyle() {
    return "[style=filled,fillcolor=\"#009933\"]";
}

std::string TrainingMatrixOperation::printNodeStyle() {
    return "[style=filled,fillcolor=\"#009933\"]";
}

std::string ALUVectorOperation::printNodeStyle() {
    return "[style=filled,fillcolor=\"#66FF66\"]";
}

std::string LoadOperation::printNodeStyle() {
    return "[style=filled,fillcolor=\"#FFB366\"]";
}

std::string StoreOperation::printNodeStyle() {
    return "[style=filled,fillcolor=\"#FFB366\"]";
}

std::string SendOperation::printNodeStyle() {
    return "[style=filled,fillcolor=\"#FFFF66\"]";
}

std::string ReceiveOperation::printNodeStyle() {
    return "[style=filled,fillcolor=\"#FFFF66\"]";
}

std::string MVMOperation::printOperationType() {
    std::stringstream ss;
    ss << "MVM: " << mat_->name();
    return ss.str();
}

std::string TrainingMatrixOperation::printOperationType() {
    std::stringstream ss;
    switch(opType_) {
        case MVM:           ss << "MVM";            break;
        case MVM_TRANSPOSE: ss << "MVM_TRANSPOSE";  break;
        case OUTER_PRODUCT: ss << "OUTER_PRODUCT";  break;
    }
    ss << ": " << mat_->name();
    return ss.str();
}

std::string ALUVectorOperation::printOperationType() {
    switch(opCode_) {
        case FADD: return "FADD";
        case FSUB: return "FSUB";
        case FMUL: return "FMUL";
        case FDIV: return "FDIV";
        case IADD: return "IADD";
        case ISUB: return "ISUB";
        case IMUL: return "IMUL";
        case IDIV: return "IDIV";
        case ADDI: return "ADDI";
        case SUBI: return "SUBI";
        case MULI: return "MULI";
        case DIVI: return "DIVI";
        case AND: return "AND";
        case OR: return "OR";
        case XOR: return "XOR";
        case NOT: return "NOT";
        case EQ: return "EQ";
        case NEQ: return "NEQ";
        case LT: return "LT";
        case LEQ: return "LEQ";
        case GT: return "GT";
        case GEQ: return "GEQ";
        case FMIN: return "MIN";
        case FMAX: return "MAX";
        case IMIN: return "IMIN";
        case IMAX: return "IMAX";
        case SIG: return "SIG";
        case TANH: return "TANH";
        case EXP: return "EXP";
        case LOG: return "LOG";
        case RELU: return "RELU";
        case RELUD: return "RELUD";
        case LOG_SOFTMAX: return "LOG_SOFTMAX";
        case LOG_SOFTMAXD: return "LOG_SOFTMAXD";
        case RNDCMP: return "RNDCMP";
        case INTtoFP: return "INTtoFP";
        case FPtoINT: return "FPtoINT";
    }
}

std::string SetImmediateOperation::printOperationType() {
    std::stringstream ss;
    ss << "Set " << imm_;
    return ss.str();
}

std::string CopyOperation::printOperationType() {
    return "Copy";
}

std::string StoreOperation::printOperationType() {
    return "Store";
}

std::string LoadOperation::printOperationType() {
    return "Load";
}

std::string SendOperation::printOperationType() {
    return "Send";
}

std::string ReceiveOperation::printOperationType() {
    return "Receive";
}

std::string WriteInputOperation::printOperationType() {
    return "WriteInput";
}

std::string ReadOutputOperation::printOperationType() {
    return "ReadOutput";
}

std::string VectorRebuildOperation::printOperationType() {
    return "VectorRebuild";
}

std::string PseudoInputOperation::printOperationType() {
    return "PseudoInput";
}

std::string PseudoOutputOperation::printOperationType() {
    return "PseudoOutput";
}

std::string ConstantVectorOperation::printOperationType() {
    return "ConstantVector";
}

void Operation::printNodeAndEdges(std::ostream& fout) {
    fout << printNodeName() << " " << printNodeStyle() << ";" << std::endl;
}

void ProducerOperation::printNodeAndEdges(std::ostream& fout) {
    Operation::printNodeAndEdges(fout);
    for(ConsumerOperation* user : users_) {
        fout << printNodeName() << " -> " << user->printNodeName() << ";" << std::endl;
    }
}

void TileMemoryWriteOperation::printNodeAndEdges(std::ostream& fout) {
    Operation::printNodeAndEdges(fout);
    for(TileMemoryReadOperation* user : users_) {
        fout << printNodeName() << " -> " << user->printNodeName() << ";" << std::endl;
    }
}

void SendOperation::printNodeAndEdges(std::ostream& fout) {
    Operation::printNodeAndEdges(fout);
    fout << printNodeName() << " -> " << dst_->printNodeName() << ";" << std::endl;
}

void InputOperation::printNodeAndEdges(std::ostream& fout) {
    fout << src_->printNodeName() << " -> " << printNodeName() << ";" << std::endl;
}

void OutputOperation::printNodeAndEdges(std::ostream& fout) {
    Operation::printNodeAndEdges(fout);
    fout << printNodeName() << " -> " << dst_->printNodeName() << ";" << std::endl;
}

void WriteInputOperation::printNodeAndEdges(std::ostream& fout) {
    TileMemoryWriteOperation::printNodeAndEdges(fout);
    InputOperation::printNodeAndEdges(fout);
}

void ReadOutputOperation::printNodeAndEdges(std::ostream& fout) {
    OutputOperation::printNodeAndEdges(fout);
}

void VectorRebuildOperation::printNodeAndEdges(std::ostream& fout) {
    Operation::printNodeAndEdges(fout);
    for(ProducerOperation* op : operands_) {
        fout << op->printNodeName() << " -> " << printNodeName() << ";" << std::endl;
    }
}


void PseudoInputOperation::printNodeAndEdges(std::ostream& fout) {
    ProducerOperation::printNodeAndEdges(fout);
    InputOperation::printNodeAndEdges(fout);
}

void PseudoOutputOperation::printNodeAndEdges(std::ostream& fout) {
    OutputOperation::printNodeAndEdges(fout);
}

void ConstantVectorOperation::printNodeAndEdges(std::ostream& fout) {
    Operation::printNodeAndEdges(fout);
    fout << printNodeName() << " -> " << vec_->printNodeName() << ";" << std::endl;
}
