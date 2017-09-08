// // FocalLoss.cc
// Based on Officinal Tensorflow[github] Template : [https://github.com/tensorflow/tensorflow/tree/0c06d8d9aa88b0596b734c8feb6435d2cf359b7e/tensorflow/core]
// Based on FAIR Research Scientist He.Kaiming Group Paper [Focal Loss for Dense Object Detection] [https://arxiv.org/abs/1708.02002]

/* Copyright 2017
/* Author Deepearthgo
==============================================================================*/

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/Focalloss.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class FocallossOp : public OpKernel {
 public:
  explicit FocallossOp(OpKernelConstruction* context) : OpKernel(context) {
     OP_REQUIRES_OK(context,
                   context->GetAttr("gamma", &gamma));
     OP_REQUIRES_OK(context,
                   context->GetAttr("alpha", &alpha));
  }

  void Compute(OpKernelContext* context) override {
    const Matrix& input = context->input(0);
    const Vec& labels = context->input(1);
   
    OP_REQUIRES(context, input.dims() == 2,
                errors::InvalidArgument("input must be 2-dimensional",
                                        input.shape().DebugString()));
    OP_REQUIRES(context, labels.dims() == 1,
                errors::InvalidArgument("labels must be 1-dimensional",
                                        labels.shape().DebugString()));
    
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input.shape(), &dx));
    
    //pro_
    Tensor scratch1;
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DataTypeToEnum<T>::value,
                                MatrixShape({input.dim_size(2)}), &scratch1));

    //pro_sum
    Tensor scratch2;
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DataTypeToEnum<T>::value,
                                MatrixrShape({input.dim_size(2)}), &scratch2));
    
    functor::BatchNorm<Device, T>()(
        context->eigen_device<Device>(), input.matrix<T>(), labels.vec<T>(),
        gamma, alpha, output->matrix<T>(), scratch1.matrix<T>(), scratch2.matrix<T>());
  }

 private:
  float gamma;
  float alpha;
};

template <typename Device, typename T>
class FocallossGradOp : public OpKernel {
 public:
  explicit BatchNormGradOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("variance_epsilon", &variance_epsilon_));
    OP_REQUIRES_OK(context,
                   context->GetAttr("gamma", &gamma));
    OP_REQUIRES_OK(context,
                   context->GetAttr("alpha", &alpha));
  }

  void Compute(OpKernelContext* context) override {
    const Matrix& input = context->input(0);
    const Vec& labels = context->input(1);
    const Matrix& out_backprop = context->input(2);

    OP_REQUIRES(context, input.dims() == 2,
                errors::InvalidArgument("input must be 2-dimensional",
                                        input.shape().DebugString()));
    OP_REQUIRES(context, labels.dims() == 1,
                errors::InvalidArgument("mean must be 1-dimensional",
                                        labels.shape().DebugString()));
    
    Tensor* dx = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input.shape(), &dx));

    //pro_1
    Matrix scratch3;
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DataTypeToEnum<T>::value,
                                MatrixShape({input.dim_size(2)}), &scratch3));

    //pro_sum
    Matrix scratch4;
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DataTypeToEnum<T>::value,
                                MatrixShape({input.dim_size(2)}), &scratch4));
    
    //pro_
    Matrix scratch5;
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DataTypeToEnum<T>::value,
                                MatrixShape({input.dim_size(2)}), &scratch5));

    //_pt
    Vec scratch6;
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DataTypeToEnum<T>::value,
                                VecShape({input.dim_size(1)}), &scratch6));
    
    //_pt+spsilon
    Tensor scratch7;
    OP_REQUIRES_OK(context, context->allocate_temp(
                                DataTypeToEnum<T>::value,
                                VecShape({input.dim_size(1)}), &scratch7));

    functor::BatchNormGrad<Device, T>()(
        context->eigen_device<Device>(), input.matrix<T>(), labels.vec<T>(),
        gamma, alpha, variance_epsilon_, dx->matrix<T>(), 
        scratch3.matrix<T>(), scratch4.matrix<T>(),scratch5.matrix<T>(), scratch6.vec<T>(),scratch7.vec<T>());
  }

 private:
  float variance_epsilon_;
  float alpha;
  float gamma;
};

#define REGISTER_KERNEL(T)                                         \
  REGISTER_KERNEL_BUILDER(Name("Focalloss") \
                              .Device(DEVICE_CPU)                  \
                              .TypeConstraint<T>("T"),             \
                          FocallossOp<CPUDevice, T>);

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);
#undef REGISTER_KERNEL

#if GOOGLE_CUDA
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                                  \                                                  
  template <>                                                                \                                                                
  void Focalloss<GPUDevice, T>::operator()(                                  \                                
      const GPUDevice& d, typename TTypes<T>::ConstMatrix input,             \
      typename TTypes<T>::ConstVec labels,                                   \
      float gamma, float alpha,                                              \
      typename TTypes<T>::Matrix output, typename TTypes<T>::Matrix scratch1,\
      typename TTypes<T>::Matrix scratch2);                                  \                              
  extern template struct Focalloss<GPUDevice, T>;

#define DECLARE_GPU_SPECS(T) DECLARE_GPU_SPEC(T);

DECLARE_GPU_SPECS(float);
#undef DECLARE_GPU_SPEC
}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_GPU_KERNEL(T)                                     \
  REGISTER_KERNEL_BUILDER(Name("Focalloss") \
                              .Device(DEVICE_GPU)                  \
                              .TypeConstraint<T>("T"),             \
                          FocallossOp<GPUDevice, T>);

REGISTER_GPU_KERNEL(float);
#undef REGISTER_GPU_KERNEL

#endif  // GOOGLE_CUDA

#define REGISTER_KERNEL(T)                                             \
  REGISTER_KERNEL_BUILDER(Name("Focalloss") \
                              .Device(DEVICE_CPU)                      \
                              .TypeConstraint<T>("T"),                 \
                          FocallossOp<CPUDevice, T>);

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);
#undef REGISTER_KERNEL

#if GOOGLE_CUDA
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T)                                                    \
  template <>                                                                  \
  void FocallossGrad<GPUDevice, T>::operator()(                                \
      const GPUDevice& d, typename TTypes<T>::ConstMatrix input,            \
      typename TTypes<T>::ConstVec labels, typename TTypes<T, 2>::ConstMatrix out_backprop,     \
      float gamma, float alpha, float variance_epsilon,                       \
      typename TTypes<T>::Matrix dx, typename TTypes<T>::Matrix scratch3, \
      typename TTypes<T>::Matrix scratch4, typename TTypes<T>::Matrix scratch5,        \
      typename TTypes<T>::Vec scratch6, typename TTypes<T>::Vec scratch7);     \
  extern template struct FocallossGrad<GPUDevice, T>;

#define DECLARE_GPU_SPECS(T) DECLARE_GPU_SPEC(T);

DECLARE_GPU_SPECS(float);
#undef DECLARE_GPU_SPEC
}  // namespace functor

// Registration of the GPU implementations.
#define REGISTER_GPU_KERNEL(T)                                         \
  REGISTER_KERNEL_BUILDER(Name("Focalloss") \
                              .Device(DEVICE_GPU)                      \
                              .TypeConstraint<T>("T"),                 \
                          FocallossOp<GPUDevice, T>);

REGISTER_GPU_KERNEL(float);
#undef REGISTER_GPU_KERNEL

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
