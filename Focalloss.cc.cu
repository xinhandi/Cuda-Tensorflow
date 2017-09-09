// // FocalLoss.cc
// Based on Officinal Tensorflow[github] Template : [https://github.com/tensorflow/tensorflow/tree/0c06d8d9aa88b0596b734c8feb6435d2cf359b7e/tensorflow/core]
// Based on FAIR Research Scientist He.Kaiming Group Paper [Focal Loss for Dense Object Detection] [https://arxiv.org/abs/1708.02002]

/* Copyright 2017
/* Author Deepearthgo
/* CALL TF-BUILD-In-[CUDA-API]
==============================================================================*/

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/Focalloss.h"

#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;
template struct functor::Focalloss<GPUDevice, float>;
template struct functor::Focalloss<GPUDevice, Eigen::half>;
template struct functor::FocallossGrad<GPUDevice, float>;
template struct functor::FocallossGrad<GPUDevice, Eigen::half>;

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
