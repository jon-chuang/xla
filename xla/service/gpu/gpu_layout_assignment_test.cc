/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/gpu_layout_assignment.h"

#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_matchers.h"
#include "xla/layout_util.h"
#include "xla/service/computation_layout.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/gemm_rewriter.h"
#include "xla/service/hlo_parser.h"
#include "xla/shape_layout.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/status_matchers.h"

namespace xla {
namespace gpu {
namespace {

namespace op = xla::testing::opcode_matchers;
using ::testing::AllOf;
using ::tsl::testing::IsOkAndHolds;

class LayoutAssignmentTest : public HloTestBase {
 public:
  se::CudaComputeCapability GetCudaComputeCapability() {
    return backend()
        .default_stream_executor()
        ->GetDeviceDescription()
        .cuda_compute_capability();
  }
};

TEST_F(LayoutAssignmentTest, Elementwise) {
  Shape ashape = ShapeUtil::MakeShape(F32, {42, 12});
  Shape ashape_in_row_major(ashape);
  Shape ashape_in_col_major(ashape);
  *ashape_in_row_major.mutable_layout() = LayoutUtil::MakeLayout({1, 0});
  *ashape_in_col_major.mutable_layout() = LayoutUtil::MakeLayout({0, 1});

  // Enumerate all possible combinations of layouts.
  for (const Shape& lhs_shape_with_layout :
       {ashape_in_row_major, ashape_in_col_major}) {
    for (const Shape& rhs_shape_with_layout :
         {ashape_in_row_major, ashape_in_col_major}) {
      for (const Shape& result_shape_with_layout :
           {ashape_in_row_major, ashape_in_col_major}) {
        // GpuLayoutAssignment should assign the same layout to "add" and its
        // two operands.
        auto builder = HloComputation::Builder(TestName());
        auto x = builder.AddInstruction(
            HloInstruction::CreateParameter(0, ashape, "x"));
        auto y = builder.AddInstruction(
            HloInstruction::CreateParameter(1, ashape, "y"));
        auto add = builder.AddInstruction(
            HloInstruction::CreateBinary(ashape, HloOpcode::kAdd, x, y));
        auto module = CreateNewVerifiedModule();
        HloComputation* computation =
            module->AddEntryComputation(builder.Build(add));

        ComputationLayout computation_layout(
            computation->ComputeProgramShape());
        *computation_layout.mutable_parameter_layout(0) =
            ShapeLayout(lhs_shape_with_layout);
        *computation_layout.mutable_parameter_layout(1) =
            ShapeLayout(rhs_shape_with_layout);
        *computation_layout.mutable_result_layout() =
            ShapeLayout(result_shape_with_layout);

        GpuLayoutAssignment layout_assignment(
            &computation_layout, backend().default_stream_executor());
        EXPECT_THAT(layout_assignment.Run(module.get()), IsOkAndHolds(true));

        for (const HloInstruction* operand : add->operands()) {
          EXPECT_TRUE(LayoutUtil::Equal(add->shape().layout(),
                                        operand->shape().layout()));
        }
      }
    }
  }
}

TEST_F(LayoutAssignmentTest, DotLayoutUnchangedIfValid) {
  const char* hlo_text = R"(
  HloModule DotLayout
  ENTRY dot {
    p0 = f32[5,2,3]{1,2,0} parameter(0)
    p1 = f32[5,3,4]{1,2,0} parameter(1)
    ROOT dot.1330.10585 = f32[5,2,4]{2,1,0} dot(p0, p1),
      lhs_batch_dims={0}, lhs_contracting_dims={2},
      rhs_batch_dims={0}, rhs_contracting_dims={1}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(&computation_layout,
                                        backend().default_stream_executor());
  EXPECT_THAT(layout_assignment.Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              AllOf(op::Dot(op::ShapeWithLayout("f32[5,2,3]{1,2,0}"),
                            op::ShapeWithLayout("f32[5,3,4]{1,2,0}")),
                    op::ShapeWithLayout("f32[5,2,4]{2,1,0}")));
}

TEST_F(LayoutAssignmentTest, DotLayoutSetToDefaultIfDefaultValid) {
  const char* hlo_text = R"(
  HloModule DotLayout
  ENTRY dot {
    p0 = f32[5,3,2] parameter(0)
    p1 = f32[5,4,3]{0,1,2} parameter(1)
    ROOT dot.1330.10585 = f32[5,2,4] dot(p0, p1),
      lhs_batch_dims={0}, lhs_contracting_dims={1},
      rhs_batch_dims={0}, rhs_contracting_dims={2}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(&computation_layout,
                                        backend().default_stream_executor());

  EXPECT_THAT(layout_assignment.Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              AllOf(op::Dot(op::ShapeWithLayout("f32[5,3,2]{2,1,0}"),
                            op::ShapeWithLayout("f32[5,4,3]{2,1,0}")),
                    op::ShapeWithLayout("f32[5,2,4]{2,1,0}")));
}

TEST_F(LayoutAssignmentTest, DotOperandLayoutSetToBatchRowsColsOtherwise) {
  const char* hlo_text = R"(
  HloModule DotLayout
  ENTRY dot {
    p0 = f32[2,3,5]{2,1,0} parameter(0)
    p1 = f32[3,4,5] parameter(1)
    ROOT dot.1330.10585 = f32[5,2,4] dot(p0, p1),
      lhs_batch_dims={2}, lhs_contracting_dims={1},
      rhs_batch_dims={2}, rhs_contracting_dims={0}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(&computation_layout,
                                        backend().default_stream_executor());

  EXPECT_THAT(layout_assignment.Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Dot(op::ShapeWithLayout("f32[2,3,5]{0,1,2}"),
                      op::ShapeWithLayout("f32[3,4,5]{1,0,2}")));
}

TEST_F(LayoutAssignmentTest, DotOperandInconsistentDimLayouts) {
  const char* hlo_text = R"(
  HloModule DotLayout
  ENTRY dot {
    p0 = f32[5,6,2,3] parameter(0)
    p1 = f32[6,5,3,4] parameter(1)
    ROOT dot.1330.10585 = f32[5,6,2,4] dot(p0, p1),
      lhs_batch_dims={0,1}, lhs_contracting_dims={3},
      rhs_batch_dims={1,0}, rhs_contracting_dims={2}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(&computation_layout,
                                        backend().default_stream_executor());

  EXPECT_THAT(layout_assignment.Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Dot(op::ShapeWithLayout("f32[5,6,2,3]{3,2,1,0}"),
                      op::ShapeWithLayout("f32[6,5,3,4]{3,2,0,1}")));
}

TEST_F(LayoutAssignmentTest, TransposedDotLayout) {
  const char* hlo_text = R"(
  HloModule DotLayout
  ENTRY dot {
    p0 = f32[5,2,3] parameter(0)
    p1 = f32[5,3,4] parameter(1)
    dot = f32[5,2,4] dot(p0, p1),
      lhs_batch_dims={0}, lhs_contracting_dims={2},
      rhs_batch_dims={0}, rhs_contracting_dims={1}
    ROOT out = f32[2,5,4] transpose(dot), dimensions={1,0,2}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(&computation_layout,
                                        backend().default_stream_executor());

  EXPECT_THAT(layout_assignment.Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              AllOf(op::Transpose(
                        AllOf(op::Dot(op::ShapeWithLayout("f32[5,2,3]{2,1,0}"),
                                      op::ShapeWithLayout("f32[5,3,4]{2,1,0}")),
                              op::ShapeWithLayout("f32[5,2,4]{2,0,1}"))),
                    op::ShapeWithLayout("f32[2,5,4]{2,1,0}")));
}

TEST_F(LayoutAssignmentTest, DotLayoutS8) {
  const char* hlo_text = R"(
  HloModule DotLayout
  ENTRY int8_t {
    p0 = s8[32,64] parameter(0)
    p1 = s8[64,96] parameter(1)
    ROOT out = s32[32,96] dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(&computation_layout,
                                        backend().default_stream_executor());

  EXPECT_THAT(layout_assignment.Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Dot(op::ShapeWithLayout("s8[32,64]{1,0}"),
                      op::ShapeWithLayout("s8[64,96]{0,1}")));
}

TEST_F(LayoutAssignmentTest, SortLayout) {
  const char* hlo_text = R"(
  HloModule SortLayout

  compare {
    p.0.lhs = f32[] parameter(0)
    p.0.rhs = f32[] parameter(1)
    p.1.lhs = f32[] parameter(2)
    p.1.rhs = f32[] parameter(3)
    ROOT lt = pred[] compare(p.0.lhs, p.0.rhs), direction=LT
  }

  ENTRY sort {
    keys = f32[3,2]{0,1} constant({{0,1},{0,1},{0,1}})
    values = f32[2,3]{1,0} parameter(0)
    transpose = f32[3,2]{1,0} transpose(values), dimensions={1,0}
    ROOT sort = (f32[3,2]{1,0}, f32[3,2]{1,0}) sort(keys, transpose),
      dimensions={1}, to_apply=compare
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(&computation_layout,
                                        backend().default_stream_executor());

  EXPECT_THAT(layout_assignment.Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Sort(op::ShapeWithLayout("f32[3,2]{1,0}"),
                       op::ShapeWithLayout("f32[3,2]{1,0}")));
}

TEST_F(LayoutAssignmentTest, FftLayout) {
  const char* hlo_text = R"(
  HloModule Fft_module

  ENTRY Fft {
    input = c64[8,32]{0,1} parameter(0)
    fft = c64[8,32] fft(input), fft_type=FFT, fft_length={32}
    ROOT transpose = c64[32,8] transpose(fft), dimensions={1,0}
  })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));

  ComputationLayout computation_layout(
      module->entry_computation()->ComputeProgramShape(),
      /*ignore_layouts=*/false);
  GpuLayoutAssignment layout_assignment(&computation_layout,
                                        backend().default_stream_executor());

  EXPECT_THAT(layout_assignment.Run(module.get()), IsOkAndHolds(true));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Copy(op::Transpose(
                  AllOf(op::Fft(op::ShapeWithLayout("c64[8,32]{1,0}")),
                        op::ShapeWithLayout("c64[8,32]{1,0}")))));
}

TEST_F(LayoutAssignmentTest, CustomCallConstrainedAlias) {
  const char* module_str = R"(
HloModule TestModule

ENTRY entry {
  Arg_0 = f32[2,5,5]{2,1,0} parameter(0)
  Arg_1 = f32[2,5,5]{2,1,0} parameter(1)
  Arg_2 = f32[2,5,5]{2,1,0} parameter(2)
  dot.0 = f32[2,5,5]{2,1,0} dot(Arg_1, Arg_2), lhs_batch_dims={0}, lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={2}, operand_precision={highest,highest}
  custom-call.0 = (f32[2,5,5]{1,2,0}, s8[16]{0}, s8[16]{0}) custom-call(Arg_0, dot.0), custom_call_target="dummy_call", operand_layout_constraints={f32[2,5,5]{1,2,0}, f32[2,5,5]{1,2,0}}, output_to_operand_aliasing={{0}: (1, {})}
  ROOT get-tuple-element.0 = f32[2,5,5]{1,2,0} get-tuple-element(custom-call.0), index=0
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(module_str));
  ComputationLayout computation_layout(
      m->entry_computation()->ComputeProgramShape());

  GpuLayoutAssignment layout_assignment(&computation_layout,
                                        backend().default_stream_executor());

  EXPECT_THAT(layout_assignment.Run(m.get()), IsOkAndHolds(true));

  const HloInstruction* call_0 = FindInstruction(m.get(), "custom-call.0");
  auto expect_layout = [](const Shape& shape,
                          absl::Span<const int64_t> minor_to_major) {
    const Layout expected = LayoutUtil::MakeLayout(minor_to_major);
    EXPECT_TRUE(LayoutUtil::Equal(shape.layout(), expected))
        << "Expected layout " << expected << ", actual " << shape.layout();
  };
  expect_layout(ShapeUtil::GetSubshape(call_0->shape(), {0}), {1, 2, 0});
  expect_layout(call_0->operand(0)->shape(), {1, 2, 0});
  expect_layout(call_0->operand(1)->shape(), {1, 2, 0});
}

TEST_F(LayoutAssignmentTest, ConvCuDNNBF16) {
  if (!GetCudaComputeCapability().IsAtLeast(
          se::CudaComputeCapability::AMPERE)) {
    GTEST_SKIP() << "Conv with Bfloat16 uses NHWC layout for "
                    "architectures with Tensor Cores.";
  }

  const char* hlo = R"(

  HloModule jit_conv_general_dilated

  ENTRY main.4 {
    Arg_0.1 = bf16[1,64,64,16]{3,2,1,0} parameter(0), sharding={replicated}
    Arg_1.2 = bf16[3,3,16,32]{3,2,1,0} parameter(1), sharding={replicated}
    ROOT convolution.3 = bf16[1,64,64,32]{3,2,1,0} convolution(Arg_0.1, Arg_1.2), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f, metadata={op_name="jit(conv_general_dilated)/jit(main)/conv_general_dilated[window_strides=(1, 1) padding=((1, 1), (1, 1)) lhs_dilation=(1, 1) rhs_dilation=(1, 1) dimension_numbers=ConvDimensionNumbers(lhs_spec=(0, 3, 1, 2), rhs_spec=(3, 2, 0, 1), out_spec=(0, 3, 1, 2)) feature_group_count=1 batch_group_count=1 lhs_shape=(1, 64, 64, 16) rhs_shape=(3, 3, 16, 32) precision=None preferred_element_type=None]" source_file="/usr/local/lib/python3.8/dist-packages/flax/linen/linear.py" source_line=438}
  }
)";

  MatchOptimizedHlo(hlo, R"(
  // CHECK: [[P0:%[^ ]+]] = bf16[1,64,64,16]{3,2,1,0} parameter(0), sharding={replicated}
  // CHECK: [[P1:%[^ ]+]] = bf16[3,3,16,32]{3,2,1,0} parameter(1), sharding={replicated}
  // CHECK-NEXT: [[P2:%[^ ]+]] = bf16[32,3,3,16]{3,2,1,0} transpose([[P1]]), dimensions={3,0,1,2}
  // CHECK-NEXT: %cudnn-conv.1 = (bf16[1,64,64,32]{3,2,1,0}, u8[0]{0}) custom-call([[P0]], [[P2]]), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_o01i->b01f, custom_call_target="__cudnn$convForward"
  )");
}

TEST_F(LayoutAssignmentTest, ConvCuDNNFP16) {
  if (!GetCudaComputeCapability().IsAtLeast(se::CudaComputeCapability::VOLTA)) {
    GTEST_SKIP() << "Conv with FP16 uses NHWC layout for "
                    "architectures with Tensor Cores.";
  }

  const char* hlo = R"(

  HloModule jit_conv_general_dilated

  ENTRY main.4 {
    Arg_0.1 = f16[1,64,64,16]{3,2,1,0} parameter(0), sharding={replicated}
    Arg_1.2 = f16[3,3,16,32]{3,2,1,0} parameter(1), sharding={replicated}
    ROOT convolution.3 = f16[1,64,64,32]{3,2,1,0} convolution(Arg_0.1, Arg_1.2), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_01io->b01f
  }
)";

  MatchOptimizedHlo(hlo, R"(
  // CHECK: [[P0:%[^ ]+]] = f16[1,64,64,16]{3,2,1,0} parameter(0), sharding={replicated}
  // CHECK: [[P1:%[^ ]+]] = f16[3,3,16,32]{3,2,1,0} parameter(1), sharding={replicated}
  // CHECK-NEXT: [[P2:%[^ ]+]] = f16[32,3,3,16]{3,2,1,0} transpose([[P1]]), dimensions={3,0,1,2}
  // CHECK-NEXT: %cudnn-conv.1 = (f16[1,64,64,32]{3,2,1,0}, u8[0]{0}) custom-call([[P0]], [[P2]]), window={size=3x3 pad=1_1x1_1}, dim_labels=b01f_o01i->b01f, custom_call_target="__cudnn$convForward"
  )");
}

}  // namespace
}  // namespace gpu
}  // namespace xla
