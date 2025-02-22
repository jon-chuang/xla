/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_EXPERIMENTAL_CONV_EMITTER_CONV_EMITTER_H_
#define XLA_EXPERIMENTAL_CONV_EMITTER_CONV_EMITTER_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {
namespace experimental {

// Builds MLIR using custom_call that represents a foward convolution.
//
// The generated function has the following signature:
// func @<function_name>(%output: memref<physical_layout...>,
//                       %input: memref<physical_layout...>,
//                       %filter: memref<physical_layout...>) { ... }
//
// Note that the custom_call is XLA/GPU-specific, as it calls into cuDNN's
// forward convolution. However, here we are building a MLIR custom emitter, and
// we are not calling into cuDNN. We just want to borrow the HLO representation
// that already exists in XLA/GPU backend.
//
// `input`, `filter`, `output` are convolution inputs.
StatusOr<mlir::func::FuncOp> EmitConvolutionForwardAsMlir(
    HloInstruction* conv, absl::string_view function_name,
    mlir::MLIRContext* context);

// Returns OkStatus() if convolution can be implemented by this emitter.
Status ConvIsImplemented(const HloInstruction* conv);

}  // namespace experimental
}  // namespace xla

#endif  // XLA_EXPERIMENTAL_CONV_EMITTER_CONV_EMITTER_H_
