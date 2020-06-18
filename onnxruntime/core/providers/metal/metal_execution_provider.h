// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2020, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"
#include "core/graph/constants.h"

namespace onnxruntime {

// Information needed to construct Metal execution providers.
struct MetalExecutionProviderInfo {
  bool create_arena{true};

  explicit MetalExecutionProviderInfo(bool use_arena)
      : create_arena(use_arena) {}

  MetalExecutionProviderInfo() = default;
};

// Logical device representation.
class MetalExecutionProvider : public IExecutionProvider {
 public:
  explicit MetalExecutionProvider(const MetalExecutionProviderInfo& info);
  virtual ~MetalExecutionProvider();

  std::vector<std::unique_ptr<ComputeCapability>> GetCapability(
      const onnxruntime::GraphViewer& graph,
      const std::vector<const KernelRegistry*>& kernel_registries) const override;

  const void* GetExecutionHandle() const noexcept override {
    // The Metal interface does not return anything interesting.
    return nullptr;
  }

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;
};

}  // namespace onnxruntime
