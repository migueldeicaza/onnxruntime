// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/execution_provider.h"
#include "core/providers/mlcompute/mlcompute_provider_factory.h"

namespace onnxruntime {
namespace mlcompute {
class Model;
}

class MlComputeExecutionProvider : public IExecutionProvider {
 public:
  MlComputeExecutionProvider();
  virtual ~MlComputeExecutionProvider();

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph_view,
                const std::vector<const KernelRegistry*>& /*kernel_registries*/) const override;

  // we implement the Compile that takes FusedNodeAndGraph instances
  FusionStyle GetFusionStyle() const override { return FusionStyle::FilteredGraphViewer; }

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  common::Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;
#endif

 private:
  // unique counter to name each fused kernel across the entire model
  mutable int metadef_id_{0};

};
}  // namespace onnxruntime
