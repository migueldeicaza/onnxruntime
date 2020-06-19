// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/util/math.h"
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

//     ORT_ENFORCE(info.GetAttr<int64_t>("transA", &temp).IsOK());
//     trans_A_ = temp == 0 ? CblasNoTrans : CblasTrans;
//     ORT_ENFORCE(info.GetAttr<int64_t>("transB", &temp).IsOK());
//     trans_B_ = temp == 0 ? CblasNoTrans : CblasTrans;

//     ORT_ENFORCE(info.GetAttr<float>("alpha", &alpha_).IsOK());
//     ORT_ENFORCE(info.GetAttr<float>("beta", &beta_).IsOK());
  }

  Status Compute(OpKernelContext* context) const  {
    //const auto X = context->Input<Tensor>(0);
    //const auto W = context->Input<Tensor>(1);
    const auto B = context->Input<Tensor>(2);

    bool useBias = B != nullptr && beta_ != 0;
    bool FC = alpha_ == 1 && (beta_ == 1 || beta_ == 0);
    if (FC) {
            return Status::OK();
    }
    if (useBias) {
            return Status::OK();
    }
    return Status::OK();
//     if (!FC) {
//       return onnxruntime::Gemm<T>::Compute(context);
//     }

//     GemmHelper helper(X->Shape(), trans_A_ != CblasNoTrans, W->Shape(), trans_B_ != CblasNoTrans, useBias ? B->Shape() : TensorShape({}));

//     if (!helper.State().IsOK())
//       return helper.State();

//     int64_t M = helper.M();
//     int64_t N = helper.N();
//     auto Y = context->Output(0, TensorShape({M, N}));

//     if (trans_A_ == CblasTrans) { // transpose input
//       return onnxruntime::Gemm<T>::Compute(context);
//     }

//     int64_t K = helper.K();
//     LOGS_DEFAULT(VERBOSE) << "Gemm Metal:" << std::endl;
//     if (X) LOGS_DEFAULT(VERBOSE) << "X " << X->Shape().ToString().c_str() << std::endl;
//     if (W) LOGS_DEFAULT(VERBOSE) << "W " << W->Shape().ToString().c_str() << std::endl;
//     if (B) LOGS_DEFAULT(VERBOSE) << "B " << B->Shape().ToString().c_str() << std::endl;
//     LOGS_DEFAULT(VERBOSE) << "Y " << Y->Shape().ToString().c_str() << std::endl;
//     LOGS_DEFAULT(VERBOSE) << "M " << (int)M << ", N " << (int)N << ", K " << (int)K << std::endl;
//     LOGS_DEFAULT(VERBOSE) << "Alfa " << alpha_ << ", Beta " << beta_ << std::endl;
//     LOGS_DEFAULT(VERBOSE) << "trans_A_ " << (trans_A_ == CblasTrans) << std::endl;
//     LOGS_DEFAULT(VERBOSE) << "trans_B_ " << (trans_B_ == CblasTrans) << std::endl;
//     LOGS_DEFAULT(VERBOSE) << std::endl;

//     const T* x_data = X->template Data<T>();
//     const T* w_data = W->template Data<T>();
//     const T* b_data;
//     if (useBias)
//       b_data = B->template Data<T>();
//     T* y_data = Y->template MutableData<T>();

//     Metal::NetworkId* pNetworkId;
//     GEMMLayersIterator it = Gemm::gemmLayers.find((OpKernel*)this);
//     if (it == Gemm::gemmLayers.end()) {
      
//       Metal::NetworkId networkId;

//       Metal::INetworkPtr myNetwork = Metal::INetwork::Create();

//       Metal::TensorShape inputShape = MetalTensorShape(X->Shape());
//       Metal::TensorShape weightShape = MetalTensorShape(W->Shape());
//       Metal::TensorShape outputShape = MetalTensorShape(Y->Shape());

//       Metal::FullyConnectedDescriptor fcDescriptor;
//       fcDescriptor.m_BiasEnabled = useBias;
//       fcDescriptor.m_TransposeWeightMatrix = trans_B_ == CblasTrans;

//       Metal::IConnectableLayer* fc_Metal;

//       Metal::TensorInfo weightsInfo(weightShape, Metal::DataType::Float32);
//       Metal::ConstTensor weights(weightsInfo, w_data);

//       if (fcDescriptor.m_BiasEnabled) {
//         Metal::TensorShape biasShape = MetalTensorShape(B->Shape());
//         if(B->Shape().NumDimensions() == 2){
//           if(B->Shape().GetDims()[0] == 1 && B->Shape().GetDims()[1] > 1)
//             biasShape = {B->Shape().GetDims()[1]};
//         }
//         Metal::TensorInfo biasDesc(biasShape, Metal::DataType::Float32);
//         Metal::ConstTensor bias(biasDesc, b_data);
//         fc_Metal = myNetwork->AddFullyConnectedLayer(fcDescriptor,
//                                                      weights,
//                                                      Metal::Optional<Metal::ConstTensor>(bias),
//                                                      "fc_Metal");
//       } else {
//         fc_Metal = myNetwork->AddFullyConnectedLayer(fcDescriptor,
//                                                      weights,
//                                                      Metal::EmptyOptional(),
//                                                      "fc_Metal");
//       }

//       Metal::IConnectableLayer *InputLayer  = myNetwork->AddInputLayer(0);
//       armnn::IConnectableLayer *OutputLayer = myNetwork->AddOutputLayer(0);

//       InputLayer->GetOutputSlot(0).Connect(fc_armnn->GetInputSlot(0));
//       fc_armnn->GetOutputSlot(0).Connect(OutputLayer->GetInputSlot(0));

//       //Set the tensors in the network.
//       armnn::TensorInfo inputTensorInfo(inputShape, armnn::DataType::Float32);
//       InputLayer->GetOutputSlot(0).SetTensorInfo(inputTensorInfo);

//       armnn::TensorInfo outputTensorInfo(outputShape, armnn::DataType::Float32);
//       fc_armnn->GetOutputSlot(0).SetTensorInfo(outputTensorInfo);

//       // Optimise ArmNN network
//       armnn::IOptimizedNetworkPtr optNet = armnn::Optimize(*myNetwork, {armnn::Compute::CpuAcc}, Gemm::run->GetDeviceSpec());

//       if (optNet == nullptr) {
//         return onnxruntime::Gemm<T>::Compute(context);
//       }

//       // Load graph into runtime
//       Gemm::run->LoadNetwork(networkId, std::move(optNet));

//       std::pair<GEMMLayersIterator, bool> ret;
//       ret = Gemm::gemmLayers.insert(std::pair<OpKernel*, armnn::NetworkId>((OpKernel*)this, networkId));
//       pNetworkId = &ret.first->second;

//     } else {
//       pNetworkId = &it->second;
//     }

//     armnn::InputTensors inputTensors{{0, armnn::ConstTensor(Gemm::run->GetInputTensorInfo(*pNetworkId, 0),
//                                                           x_data)}};
//     armnn::OutputTensors outputTensors{{0, armnn::Tensor(Gemm::run->GetOutputTensorInfo(*pNetworkId, 0),
//                                                          y_data)}};

//     Gemm::run->EnqueueWorkload(*pNetworkId, inputTensors, outputTensors);

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

// template <typename T>
// thread_local std::map<OpKernel*, armnn::NetworkId> onnxruntime::armnn_ep::Gemm<T>::gemmLayers;

// template <typename T>
// armnn::IRuntimePtr Gemm<T>::run = Gemm<T>::initRuntime();

}  // namespace Metal_ep
}  // namespace onnxruntime

