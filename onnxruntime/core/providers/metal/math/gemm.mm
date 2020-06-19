
#include "core/providers/metal/metal_common.h"
#include "core/providers/metal/math/gemm.h"
#include "core/providers/metal/metal_fwd.h"

namespace onnxruntime {
namespace Metal_ep {

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Gemm,
    kOnnxDomain,
    7,
    8,
    kMetalExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Gemm<float>);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Gemm,
    kOnnxDomain,
    9,
    10,
    kMetalExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Gemm<float>);

ONNX_OPERATOR_KERNEL_EX(
    Gemm,
    kOnnxDomain,
    11,
    kMetalExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Gemm<float>);

}  // namespace Metal_ep
}  // namespace onnxruntime