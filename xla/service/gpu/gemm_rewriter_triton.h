/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef XLA_SERVICE_GPU_GEMM_REWRITER_TRITON_H_
#define XLA_SERVICE_GPU_GEMM_REWRITER_TRITON_H_

#include <array>
#include <vector>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/stream_executor/device_description.h"
#include "tsl/protobuf/autotuning.pb.h"

namespace xla {
namespace gpu {

// Apply split K configuration from the tiling to the fused dot() computation:
// bitcast the operands, change the output shape and the dot dimensions.
Status MakeDotComputationSplitKBatch(
    HloComputation* computation,
    const tensorflow::AutotuneResult::TritonGemmKey& tiling);

// Apply split K configuration from the tiling to the fusion instruction:
// in addition to MakeDotComputationSplitKBatch on its computation add the
// necessary reduction after it.
Status MakeDotSplitKBatch(
    HloInstruction* dot_fusion,
    const tensorflow::AutotuneResult::TritonGemmKey& tiling);

// Filters GEMMs which are better to handle using Triton.
bool IsTritonHandledGEMM(const HloInstruction&,
                         se::CudaComputeCapability cuda_compute_capability);

// Analysis of iteration of HLO shapes within a fusion around dot().
class DotFusionAnalysis {
 public:
  // Description of basic iteration: `count` elements separated by `stride`.
  struct IterationSpecFragment {
    int64_t stride;
    int64_t count;
  };

  // Description of complex iteration over a sequence of several strides.
  // Describes a logically contiguous dimension of a tensor physically
  // separated into multiple fragments by other dimensions.
  using IterationSpec = std::vector<IterationSpecFragment>;

  // Execute analysis of fusion rooted with the instruction.
  // split_k indicates whether this operation was converted to the split-K
  // form and tells the analysis how to interpret the batch dimensions.
  explicit DotFusionAnalysis(const HloInstruction* root, int64_t split_k = 1);

  // Description of iteration of given dimension of given operand of `root`.
  const IterationSpec& IterSpec(const int operand_number,
                                const int dimension) const {
    return iter_specs_.at(operand_number).at(dimension);
  }
  // Parameter HLO instruction corresponding to Nth operand of `root`.
  const HloInstruction* OperandToParameter(const int operand_number) const {
    return operand_to_parameter_.at(operand_number);
  }

 private:
  // Dimension number -> iteration spec for both dot operands.
  std::array<absl::flat_hash_map<int, IterationSpec>, 2> iter_specs_;
  // Computation parameters corresponding to both dot operands.
  std::array<const HloInstruction*, 2> operand_to_parameter_;
};

// Rewrite compatible dot() calls into custom calls with fused computations
// that target Triton-based matmul emitter.
class GemmRewriterTriton : public HloModulePass {
 public:
  explicit GemmRewriterTriton(se::CudaComputeCapability cc)
      : cuda_compute_capability_(cc) {}
  absl::string_view name() const override { return "triton-gemm-rewriter"; }

  using HloPassInterface::Run;
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  se::CudaComputeCapability cuda_compute_capability_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_GEMM_REWRITER_TRITON_H_
