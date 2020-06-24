// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2020, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#include "metal_execution_provider.h"
#include "core/framework/allocator.h"
#include "core/framework/op_kernel.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/compute_capability.h"
#include "contrib_ops/cpu/cpu_contrib_kernels.h"
#include "metal_fwd.h"

namespace onnxruntime {

constexpr const char* Metal = "Metal";
constexpr const char* Metal_CPU = "MetalCpu";

namespace Metal_ep {

// Forward declarations of op kernels
// class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kMetalExecutionProvider, kOnnxDomain, 1, 10, Conv);
// class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMetalExecutionProvider, kOnnxDomain, 11, Conv);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kMetalExecutionProvider, kOnnxDomain, 7, 8, Gemm);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kMetalExecutionProvider, kOnnxDomain, 9, 10, Gemm);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kMetalExecutionProvider, kOnnxDomain, 11, Gemm);

// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMetalExecutionProvider, kOnnxDomain, 7, 9, float, AveragePool);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMetalExecutionProvider, kOnnxDomain, 10, 10, float, AveragePool);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMetalExecutionProvider, kOnnxDomain, 11, float, AveragePool);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMetalExecutionProvider, kOnnxDomain, 1, 7, float, MaxPool);
// class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMetalExecutionProvider, kOnnxDomain, 8, 11, float, MaxPool);

// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMetalExecutionProvider, kOnnxDomain, 1, float, GlobalAveragePool);
// class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMetalExecutionProvider, kOnnxDomain, 1, float, GlobalMaxPool);

static void RegisterMetalKernels(KernelRegistry& kernel_registry) {

  // kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kMetalExecutionProvider, kOnnxDomain, 1, 10, Conv)>());
  // kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kMetalExecutionProvider, kOnnxDomain, 11, Conv)>());

  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kMetalExecutionProvider, kOnnxDomain, 7, 8, Gemm)>());
  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kMetalExecutionProvider, kOnnxDomain, 9, 10, Gemm)>());
  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kMetalExecutionProvider, kOnnxDomain, 11, Gemm)>());

  // kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMetalExecutionProvider, kOnnxDomain, 7, 9, float, AveragePool)>());
  // kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMetalExecutionProvider, kOnnxDomain, 10, 10, float, AveragePool)>());
  // kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMetalExecutionProvider, kOnnxDomain, 11, float, AveragePool)>());
  // kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMetalExecutionProvider, kOnnxDomain, 1, 7, float, MaxPool)>());
  // kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kMetalExecutionProvider, kOnnxDomain, 8, 11, float, MaxPool)>());

  // kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMetalExecutionProvider, kOnnxDomain, 1, float, GlobalAveragePool)>());
  // kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kMetalExecutionProvider, kOnnxDomain, 1, float, GlobalMaxPool)>());

}

id<MTLDevice> _Nonnull metalDevice;
id<MTLCommandQueue> _Nonnull metalCommandQueue;

void metal_init()
{
  // Get the metal device and commandQueue to be used.
  metalDevice = MTLCreateSystemDefaultDevice();
  metalCommandQueue = [metalDevice newCommandQueue];
}

std::shared_ptr<KernelRegistry> GetMetalKernelRegistry() {
  std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();
  RegisterMetalKernels(*kernel_registry);

  return kernel_registry;
}

}  // namespace Metal_ep

MetalExecutionProvider::MetalExecutionProvider(const MetalExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kMetalExecutionProvider} {
  ORT_UNUSED_PARAMETER(info);

  DeviceAllocatorRegistrationInfo device_info(
      {OrtMemTypeDefault,
       [](int) {
         return onnxruntime::make_unique<CPUAllocator>(OrtMemoryInfo(Metal_CPU, OrtAllocatorType::OrtDeviceAllocator));
       },
       std::numeric_limits<size_t>::max()});

  InsertAllocator(CreateAllocator(device_info));

  DeviceAllocatorRegistrationInfo cpu_memory_info(
      {OrtMemTypeCPUOutput,
       [](int) {
         return onnxruntime::make_unique<CPUAllocator>(
             OrtMemoryInfo(Metal_CPU, OrtAllocatorType::OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeCPUOutput));
       },
       std::numeric_limits<size_t>::max()});
 
  Metal_ep::metal_init();
  InsertAllocator(CreateAllocator(cpu_memory_info));
}

MetalExecutionProvider::~MetalExecutionProvider() {
}

std::shared_ptr<KernelRegistry> MetalExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry = onnxruntime::Metal_ep::GetMetalKernelRegistry();
  return kernel_registry;
}

std::vector<std::unique_ptr<ComputeCapability>>
MetalExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph,
                                    const std::vector<const KernelRegistry*>& kernel_registries) const {
  std::vector<std::unique_ptr<ComputeCapability>>
      result = IExecutionProvider::GetCapability(graph, kernel_registries);

  return result;
}

}  // namespace onnxruntime
