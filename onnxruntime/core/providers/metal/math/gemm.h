// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/providers/cpu/math/gemm.h"
#include "core/providers/metal/metal_execution_provider.h"

namespace onnxruntime {
namespace Metal_ep {

//typedef std::map<OpKernel*, armnn::NetworkId>::iterator GEMMLayersIterator;

template <typename T>
class Gemm : public onnxruntime::OpKernel {
 public:
  Gemm(const OpKernelInfo& info) : onnxruntime::OpKernel(info) {
          int64_t temp;
          ORT_ENFORCE(info.GetAttr<int64_t>("transA", &temp).IsOK());

    ORT_ENFORCE(info.GetAttr<int64_t>("transA", &temp).IsOK());
    trans_A_ = temp == 0 ? CblasNoTrans : CblasTrans;
    ORT_ENFORCE(info.GetAttr<int64_t>("transB", &temp).IsOK());
    trans_B_ = temp == 0 ? CblasNoTrans : CblasTrans;

    ORT_ENFORCE(info.GetAttr<float>("alpha", &alpha_).IsOK());
    ORT_ENFORCE(info.GetAttr<float>("beta", &beta_).IsOK());
  }

  Status Compute(OpKernelContext* context) const  {
    const auto X = context->Input<Tensor>(0);
    const auto W = context->Input<Tensor>(1);
    const auto B = context->Input<Tensor>(2);

    bool useBias = B != nullptr && beta_ != 0;
    bool FC = alpha_ == 1 && (beta_ == 1 || beta_ == 0);

    if (!FC) {
            printf ("MEtal: notf FC\n");
      // return onnxruntime::Gemm<T>::Compute(context);
      return Status::OK();
    }

    GemmHelper helper(X->Shape(), trans_A_ != CblasNoTrans, W->Shape(), trans_B_ != CblasNoTrans, useBias ? B->Shape() : TensorShape({}));

    if (!helper.State().IsOK())
      return helper.State();

    int64_t M = helper.M();
    int64_t N = helper.N();
    int64_t K = helper.K();
    auto Y = context->Output(0, TensorShape({M, N}));

    // if input is empty tensor, return as nothing need to be calculated and we've set the shape for the output
    if (M == 0 || N == 0)
      return Status::OK();

    const T* b_data = B != nullptr ? B->Data<T>() : nullptr;
    const TensorShape* b_shape = B != nullptr ? &B->Shape() : nullptr;

    T* y_data = Y->MutableData<T>();

    auto xShape = X->Shape();
    id<MTLBuffer> _Nullable xBuffer = [metalDevice newBufferWithBytes: X->Data<T>() length: xShape.Size()*sizeof(float) options: MTLResourceCPUCacheModeWriteCombined];
    MPSMatrixDescriptor *descX = [MPSMatrixDescriptor 
        matrixDescriptorWithRows: M
        columns: K
        rowBytes: K*sizeof(float)
        dataType: MPSDataTypeFloat32];
    MPSMatrix *matrixX = [[MPSMatrix alloc] initWithBuffer: xBuffer descriptor: descX];

    auto wShape = W->Shape();
    id<MTLBuffer> _Nullable wBuffer = [metalDevice newBufferWithBytes: W->Data<T>() length: wShape.Size()*sizeof(float) options: MTLResourceCPUCacheModeWriteCombined];
    MPSMatrixDescriptor *descW = [MPSMatrixDescriptor 
        matrixDescriptorWithRows: K
        columns: N
        rowBytes: N*sizeof(float)
        dataType: MPSDataTypeFloat32];
    MPSMatrix *matrixW = [[MPSMatrix alloc] initWithBuffer: wBuffer descriptor: descW];

    id<MTLBuffer> _Nullable resultBuffer = [metalDevice newBufferWithLength: K * M * sizeof (float) options: 0];
    MPSMatrixDescriptor *descResult = [MPSMatrixDescriptor 
        matrixDescriptorWithRows: M
        columns: N
        rowBytes: N*sizeof(float)
        dataType: MPSDataTypeFloat32];
    MPSMatrix *matrixResult = [[MPSMatrix alloc] initWithBuffer: resultBuffer descriptor: descResult];

    MPSMatrixMultiplication *mul = [[MPSMatrixMultiplication alloc]
        initWithDevice: metalDevice
        transposeLeft: trans_A_ == CblasTrans
        transposeRight: trans_B_ == CblasTrans
        resultRows: M
        resultColumns: N
        interiorColumns: K
        alpha: alpha_
        beta: beta_
    ];

    id<MTLCommandBuffer> cmdBuffer = [metalCommandQueue commandBuffer];
    [mul 
        encodeToCommandBuffer:cmdBuffer
        leftMatrix: matrixX
        rightMatrix: matrixW
        resultMatrix: matrixResult
    ];
    [cmdBuffer commit];
    [cmdBuffer waitUntilCompleted];
    auto data = [matrixResult data];
    //memcpy (y_data, [data contents], [data length]);
    return Status::OK();
  }

  ~Gemm() {
    //gemmLayers.erase(this);
  }

//   static Metal::IRuntimePtr initRuntime(){
//     if (Gemm::run)
//       return std::move(Gemm::run);
//     //Metal::IRuntime::CreationOptions options;
//     printf ("Should init Metal")
//   }

 private:
  //static thread_local std::map<OpKernel*, armnn::NetworkId> gemmLayers;
  MetalExecutionProvider* provider_;
  //static Metal::IRuntimePtr run;

  CBLAS_TRANSPOSE trans_A_;
  CBLAS_TRANSPOSE trans_B_;
  float alpha_;
  float beta_;
};

}  // namespace Metal_ep
}  // namespace onnxruntime

