// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2020, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/metal/metal_provider_factory.h"
#include "metal_execution_provider.h"
#include "core/session/abi_session_options_impl.h"

namespace onnxruntime {

struct MetalProviderFactory : IExecutionProviderFactory {
  MetalProviderFactory() {}
  ~MetalProviderFactory() override {}
  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  
};

std::unique_ptr<IExecutionProvider> MetalProviderFactory::CreateProvider() {
  MetalExecutionProviderInfo info;
  return std::make_unique<MetalExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Metal() {
  return std::make_shared<onnxruntime::MetalProviderFactory>();
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_Metal, _In_ OrtSessionOptions* options) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_Metal());
  return nullptr;
}