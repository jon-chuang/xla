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

#include <memory>
#include <utility>
#include <vector>

#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/test_macros.h"

namespace xla {
namespace {

using ::testing::NotNull;

// Makes a DeviceAssignment device#i to replica_id #i.
DeviceAssignment MakeDeviceAssn(int64_t num_replicas) {
  DeviceAssignment assn(/*replica_count=*/num_replicas,
                        /*computation_count=*/1);
  for (int64_t i = 0; i < num_replicas; ++i) {
    assn(i, 0) = i;
  }
  return assn;
}

// E2E tests for collective ops. These will generally verify some HLO transform
// for collectives (for example, sync -> async conversion) and correct
// execution of the transformed HLO.

// E2E test for async collectives. Tested with both async collective enabled
// and disabled. Verify that async collective is generated when enabled
// in the end-to-end compilation for GPU's and that the execution produces
// correct result.
class AsyncCollectiveOps : public HloTestBase,
                           public ::testing::WithParamInterface<bool> {
 public:
  AsyncCollectiveOps() : num_devices_(backend().device_count()) {
    VLOG(1) << "Running with " << num_devices_ << " devices";
  }

 protected:
  StatusOr<std::vector<Literal>> ExecuteReplicated(Executable* executable,
                                                   int64_t num_replicas) {
    DeviceAssignment device_assignment = MakeDeviceAssn(num_replicas);
    return HloTestBase::ExecuteReplicated(
        /*executable_provider*/ [&](int64_t) { return executable; },
        /*argument_count_provider*/ [](int64_t) { return 0; },
        /*argument_provider*/ [](int64_t, int64_t) { return nullptr; },
        num_replicas, /*run_hlo_passes=*/false, &device_assignment);
  }

  const int64_t num_devices_;
};

XLA_TEST_P(AsyncCollectiveOps, AsyncAllReduce) {
  const absl::string_view kModuleStr = R"(
      HloModule test

      apply_op {
        x = u32[] parameter(0)
        y = u32[] parameter(1)
        ROOT apply_op = u32[] add(x, y)
      }

      ENTRY test_computation {
        id = u32[] replica-id()
        ROOT all-reduce = u32[] all-reduce(id), to_apply=apply_op
      }
    )";

  const int64_t kNumReplicas = 2;
  const bool enable_async_all_reduce = GetParam();
  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);
  config.mutable_debug_options()->set_xla_gpu_enable_async_all_reduce(
      enable_async_all_reduce);
  config.mutable_debug_options()->set_xla_gpu_enable_async_collectives_to_sync(
      false);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr, config));

  TF_ASSERT_OK_AND_ASSIGN(
      auto executable,
      CreateExecutable(std::move(module), /*run_hlo_passes=*/true));
  EXPECT_TRUE(executable->has_module());

  if (enable_async_all_reduce) {
    HloInstruction* all_reduce_start =
        FindInstruction(&executable->module(), HloOpcode::kAllReduceStart);
    HloInstruction* all_reduce_done =
        FindInstruction(&executable->module(), HloOpcode::kAllReduceDone);
    EXPECT_THAT(all_reduce_start, NotNull());
    EXPECT_THAT(all_reduce_done, NotNull());
  } else {
    HloInstruction* all_reduce =
        FindInstruction(&executable->module(), HloOpcode::kAllReduce);
    EXPECT_THAT(all_reduce, NotNull());
  }

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                          ExecuteReplicated(executable.get(), kNumReplicas));
  ASSERT_EQ(results.size(), kNumReplicas);
  // sum [0, num_devices)
  const uint32_t expected = kNumReplicas * (kNumReplicas - 1) / 2;
  for (int i = 0; i < kNumReplicas; ++i) {
    LiteralTestUtil::ExpectR0Equal<uint32_t>(expected, results[i]);
  }
}

XLA_TEST_P(AsyncCollectiveOps, AsyncAllGather) {
  const absl::string_view kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    id = u32[] replica-id()
    id2 = u32[1, 2] broadcast(id), dimensions={}
    a0 = u32[1, 2] constant({{10, 15}})
    a1 = u32[1, 2] add(id2, a0)
    allgather = u32[2, 2] all-gather(a1), dimensions={0}
    ROOT out = u32[4] reshape(allgather)
  }
  )";
  const int64_t kNumReplicas = 2;
  const bool enable_async_all_gather = GetParam();

  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);
  config.mutable_debug_options()->set_xla_gpu_enable_async_all_gather(
      enable_async_all_gather);
  config.mutable_debug_options()->set_xla_gpu_enable_async_collectives_to_sync(
      false);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr, config));

  TF_ASSERT_OK_AND_ASSIGN(
      auto executable,
      CreateExecutable(std::move(module), /*run_hlo_passes=*/true));
  EXPECT_TRUE(executable->has_module());
  if (enable_async_all_gather) {
    HloInstruction* all_gather_start =
        FindInstruction(&executable->module(), HloOpcode::kAllGatherStart);
    HloInstruction* all_gather_done =
        FindInstruction(&executable->module(), HloOpcode::kAllGatherDone);
    EXPECT_THAT(all_gather_start, NotNull());
    EXPECT_THAT(all_gather_done, NotNull());
  } else {
    HloInstruction* all_gather =
        FindInstruction(&executable->module(), HloOpcode::kAllGather);
    EXPECT_THAT(all_gather, NotNull());
  }

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                          ExecuteReplicated(executable.get(), kNumReplicas));

  ASSERT_EQ(results.size(), kNumReplicas);
  for (const Literal& result : results) {
    LiteralTestUtil::ExpectR1Equal<uint32_t>({10, 15, 11, 16}, result);
  }
}

XLA_TEST_P(AsyncCollectiveOps, AsyncCollectivePermute) {
  const absl::string_view kModuleStr = R"(
  HloModule test
  ENTRY test_computation {
    replica = u32[] replica-id()
    ten = u32[] constant(10)
    sum = u32[] add(replica, ten)
    p = u32[2] broadcast(sum), dimensions={}
    permute = u32[2] collective-permute(p), source_target_pairs={{1,0}, {0,1}}
    ROOT copy = u32[2] copy(permute)
  }
  )";
  const int64_t kNumReplicas = 2;
  const bool enable_async_collective_permute = GetParam();
  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);
  config.mutable_debug_options()->set_xla_gpu_enable_async_collective_permute(
      enable_async_collective_permute);
  config.mutable_debug_options()->set_xla_gpu_enable_async_collectives_to_sync(
      false);

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr, config));
  TF_ASSERT_OK_AND_ASSIGN(
      auto executable,
      CreateExecutable(std::move(module), /*run_hlo_passes=*/true));
  EXPECT_TRUE(executable->has_module());
  if (enable_async_collective_permute) {
    HloInstruction* cp_start = FindInstruction(
        &executable->module(), HloOpcode::kCollectivePermuteStart);
    HloInstruction* cp_done = FindInstruction(
        &executable->module(), HloOpcode::kCollectivePermuteDone);
    EXPECT_THAT(cp_start, NotNull());
    EXPECT_THAT(cp_done, NotNull());
  } else {
    HloInstruction* cp =
        FindInstruction(&executable->module(), HloOpcode::kCollectivePermute);
    EXPECT_THAT(cp, NotNull());
  }

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                          ExecuteReplicated(executable.get(), kNumReplicas));
  ASSERT_EQ(results.size(), kNumReplicas);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({11, 11}, results[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({10, 10}, results[1]);
}

XLA_TEST_P(AsyncCollectiveOps, AsyncReduceScatter) {
  const absl::string_view kModuleStr = R"(
  HloModule test
  add {
    lhs = u32[] parameter(0)
    rhs = u32[] parameter(1)
    ROOT add = u32[] add(lhs, rhs)
  }

  ENTRY main {
    c0 = u32[8] constant({1, 2, 3, 4, 5, 6, 7, 8})
    c1 = u32[8] constant({10, 11, 12, 13, 14, 15, 16, 17})
    zero = u32[] constant(0)
    id = u32[] replica-id()
    p = pred[] compare(id, zero), direction=EQ
    pb = pred[8] broadcast(p), dimensions={}
    // data = c0 for replica 0 and c1 for replica 1
    data = u32[8] select(pb, c0, c1)
    ROOT ars = u32[4] reduce-scatter(data), replica_groups={},
                      dimensions={0}, to_apply=add
  }
  )";

  const int64_t kNumReplicas = 2;
  const bool enable_async_reduce_scatter = GetParam();
  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);
  config.mutable_debug_options()->set_xla_gpu_enable_async_reduce_scatter(
      enable_async_reduce_scatter);
  config.mutable_debug_options()->set_xla_gpu_enable_async_collectives_to_sync(
      false);

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr, config));
  TF_ASSERT_OK_AND_ASSIGN(
      auto executable,
      CreateExecutable(std::move(module), /*run_hlo_passes=*/true));
  EXPECT_TRUE(executable->has_module());
  if (enable_async_reduce_scatter) {
    HloInstruction* rs_start =
        FindInstruction(&executable->module(), HloOpcode::kAsyncStart);
    HloInstruction* rs_done =
        FindInstruction(&executable->module(), HloOpcode::kAsyncDone);
    ASSERT_THAT(rs_start, NotNull());
    ASSERT_THAT(rs_done, NotNull());
    HloAsyncInstruction* rs_start_async = Cast<HloAsyncInstruction>(rs_start);
    EXPECT_EQ(rs_start_async->async_wrapped_opcode(),
              HloOpcode::kReduceScatter);
  } else {
    HloInstruction* rs =
        FindInstruction(&executable->module(), HloOpcode::kReduceScatter);
    EXPECT_THAT(rs, NotNull());
  }

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                          ExecuteReplicated(executable.get(), kNumReplicas));
  LiteralTestUtil::ExpectR1Equal<uint32_t>({11, 13, 15, 17}, results[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({19, 21, 23, 25}, results[1]);
}

XLA_TEST_P(AsyncCollectiveOps, AsyncAllToAllWithSplitDim) {
  const absl::string_view kModuleStr = R"(
  HloModule test

  ENTRY test_computation {
    id = u32[] replica-id()
    id2 = u32[2] broadcast(id), dimensions={}
    a0 = u32[2] constant({10, 15})
    a1 = u32[2] add(id2, a0)
    ROOT a2a = u32[2] all-to-all(u32[2] a1), dimensions={0}
  }
  )";
  const int64_t kNumReplicas = 2;
  const bool enable_async_all_to_all = GetParam();
  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);
  config.mutable_debug_options()->set_xla_gpu_enable_async_all_to_all(
      enable_async_all_to_all);
  config.mutable_debug_options()->set_xla_gpu_enable_async_collectives_to_sync(
      false);

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr, config));
  TF_ASSERT_OK_AND_ASSIGN(
      auto executable,
      CreateExecutable(std::move(module), /*run_hlo_passes=*/true));
  EXPECT_TRUE(executable->has_module());

  if (enable_async_all_to_all) {
    HloInstruction* a2a_start =
        FindInstruction(&executable->module(), HloOpcode::kAsyncStart);
    HloInstruction* a2a_done =
        FindInstruction(&executable->module(), HloOpcode::kAsyncDone);
    ASSERT_THAT(a2a_start, NotNull());
    ASSERT_THAT(a2a_done, NotNull());
    HloAsyncInstruction* rs_start_async = Cast<HloAsyncInstruction>(a2a_start);
    EXPECT_EQ(rs_start_async->async_wrapped_opcode(), HloOpcode::kAllToAll);
  } else {
    HloInstruction* a2a =
        FindInstruction(&executable->module(), HloOpcode::kAllToAll);
    EXPECT_THAT(a2a, NotNull());
  }

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                          ExecuteReplicated(executable.get(), kNumReplicas));
  ASSERT_EQ(results.size(), kNumReplicas);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({10, 11}, results[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({15, 16}, results[1]);
}

XLA_TEST_P(AsyncCollectiveOps, AsyncAllToAllWithoutSplitDim) {
  const absl::string_view kModuleStr = R"(
  HloModule test

  ENTRY test_computation {
    id = u32[] replica-id()
    id2 = u32[2] broadcast(id), dimensions={}
    a0 = u32[2] constant({10, 15})
    a1 = u32[2] add(id2, a0)
    a2 = u32[2] constant({4, 4})
    a3 = u32[2] multiply(a1, a2)
    // r0 : a1 = {10, 15}, a2 = {40, 60)
    // r1 : a1 = {11, 16}, a1 = {44, 64}
    // r0: a2a element 0 = {10, 15}, a2a element 1 = {11, 16}
    // r0: a2a element 0 = {40, 60}, a2a element 1 = {44, 64}
    a2a = (u32[2], u32[2]) all-to-all(u32[2] a1, u32[2] a3), replica_groups={{0,1}}
    gte0 = get-tuple-element(a2a), index=0
    gte1 = get-tuple-element(a2a), index=1
    ROOT x = u32[4] concatenate(gte0, gte1), dimensions={0}
  }
  )";
  const int64_t kNumReplicas = 2;
  const bool enable_async_all_to_all = GetParam();
  HloModuleConfig config =
      GetModuleConfigForTest(/*replica_count=*/kNumReplicas);
  config.mutable_debug_options()->set_xla_gpu_enable_async_all_to_all(
      enable_async_all_to_all);
  config.mutable_debug_options()->set_xla_gpu_enable_async_collectives_to_sync(
      false);

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr, config));
  TF_ASSERT_OK_AND_ASSIGN(
      auto executable,
      CreateExecutable(std::move(module), /*run_hlo_passes=*/true));
  EXPECT_TRUE(executable->has_module());
  if (enable_async_all_to_all) {
    HloInstruction* a2a_start =
        FindInstruction(&executable->module(), HloOpcode::kAsyncStart);
    HloInstruction* a2a_done =
        FindInstruction(&executable->module(), HloOpcode::kAsyncDone);
    ASSERT_THAT(a2a_start, NotNull());
    ASSERT_THAT(a2a_done, NotNull());
    HloAsyncInstruction* rs_start_async = Cast<HloAsyncInstruction>(a2a_start);
    EXPECT_EQ(rs_start_async->async_wrapped_opcode(), HloOpcode::kAllToAll);
  } else {
    HloInstruction* a2a =
        FindInstruction(&executable->module(), HloOpcode::kAllToAll);
    EXPECT_THAT(a2a, NotNull());
  }

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Literal> results,
                          ExecuteReplicated(executable.get(), kNumReplicas));
  ASSERT_EQ(results.size(), kNumReplicas);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({10, 15, 11, 16}, results[0]);
  LiteralTestUtil::ExpectR1Equal<uint32_t>({40, 60, 44, 64}, results[1]);
}

INSTANTIATE_TEST_SUITE_P(AsyncCollectiveOps, AsyncCollectiveOps,
                         ::testing::Bool());

}  // namespace
}  // namespace xla
