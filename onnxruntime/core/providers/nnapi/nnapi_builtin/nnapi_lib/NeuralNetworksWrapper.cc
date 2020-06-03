/*
 * Copyright (C) 2017 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "NeuralNetworksWrapper.h"

namespace android {
namespace nn {
namespace wrapper {

OperandType::OperandType(Type type, const std::vector<uint32_t>& d, float scale, int32_t zeroPoint)
    : type(type), dimensions(std::move(d)) {
  operandType = {
      .type = static_cast<int32_t>(type),
      .dimensionCount = static_cast<uint32_t>(dimensions.size()),
      .dimensions = dimensions.size() > 0 ? dimensions.data() : nullptr,
      .scale = scale,
      .zeroPoint = zeroPoint,
  };
}

OperandType::OperandType(const OperandType& other) {
  type = other.type;
  dimensions = other.dimensions;
  operandType = {
      .type = static_cast<int32_t>(type),
      .dimensionCount = static_cast<uint32_t>(dimensions.size()),
      .dimensions = dimensions.size() > 0 ? dimensions.data() : nullptr,
      .scale = other.operandType.scale,
      .zeroPoint = other.operandType.zeroPoint,
  };
}  // namespace wrapper

size_t OperandType::GetElementByteSize() const {
  size_t element_size;
  switch (type) {
    case Type::TENSOR_BOOL8:
      element_size = 1;
      break;
    case Type::TENSOR_FLOAT16:
      element_size = 2;
      break;
    case Type::TENSOR_FLOAT32:
    case Type::FLOAT32:
      element_size = 4;
      break;
    case Type::TENSOR_INT32:
      element_size = 4;
      break;
    case Type::TENSOR_QUANT8_SYMM_PER_CHANNEL:
      element_size = 1;
      break;
    case Type::TENSOR_QUANT8_ASYMM:
      element_size = 1;
      break;
    case Type::TENSOR_QUANT16_SYMM:
      element_size = 2;
      break;
    case Type::TENSOR_QUANT16_ASYMM:
      element_size = 2;
      break;
    default:
      throw std::invalid_argument("Wrong type: " +
                                  typeToStr(type));
  }

  return element_size;
}

size_t OperandType::GetOperandByteSize() const {
  return Product(dimensions) * GetElementByteSize();
}

}  // namespace wrapper
}  // namespace nn
}  // namespace android