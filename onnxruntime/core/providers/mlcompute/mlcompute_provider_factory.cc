// Copyright Microsoft Corp 2021

#include "core/providers/mlcompute/mlcompute_execution_provider.h"
#include "core/session/abi_session_options_impl.h"

using namespace onnxruntime;

namespace onnxruntime {
struct MlComputeProviderFactory : IExecutionProviderFactory {
  MlComputeProviderFactory() {}
  ~MlComputeProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;
};

std::unique_ptr<IExecutionProvider> MlComputeProviderFactory::CreateProvider() {
  return onnxruntime::make_unique<MlComputeExecutionProvider>();
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_MlCompute() {
  return std::make_shared<onnxruntime::MlComputeProviderFactory>();
}
}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_MlCompute, _In_ OrtSessionOptions* options) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_MlCompute());
  return nullptr;
}
