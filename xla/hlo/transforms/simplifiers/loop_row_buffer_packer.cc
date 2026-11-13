/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/hlo/transforms/simplifiers/loop_row_buffer_packer.h"

#include <cstdint>
#include <iterator>
#include <map>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/while_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/side_effect_util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

namespace m = ::xla::match;

// Wider rows keep their own update: packing copies every row once more.
constexpr int64_t kMaxRowElements = 128;
// Smaller groups keep their updates: the concatenate would not pay for itself.
constexpr int64_t kMinBuffersPerGroup = 8;

// Whether `instruction` writes one whole row into a static [N] or [N, m]
// buffer, at a scalar row index. The row lands at column 0 whatever the column
// index: dynamic-update-slice clamps the start so that the update fits.
bool IsRowWrite(const HloInstruction* instruction) {
  // A packed write would not carry the annotations or the ordering of the
  // writes it replaces.
  if (instruction->opcode() != HloOpcode::kDynamicUpdateSlice ||
      instruction->has_sharding() ||
      !instruction->frontend_attributes().map().empty() ||
      instruction->HasControlDependencies()) {
    return false;
  }
  const Shape& update = instruction->operand(1)->shape();
  Shape row = instruction->shape();
  if (row.is_dynamic() || update.is_dynamic() || row.dimensions().empty() ||
      row.dimensions().size() > 2) {
    return false;
  }
  row.set_dimensions(0, 1);
  return ShapeUtil::Compatible(update, row) &&
         ShapeUtil::IsScalar(instruction->operand(2)->shape());
}

// How often the body and the condition of `while_instr` read each element of
// the loop state, or nullopt if they use the state other than by element.
std::optional<std::vector<int64_t>> CountReads(
    const HloInstruction* while_instr) {
  std::vector<int64_t> reads(while_instr->shape().tuple_shapes().size(), 0);
  for (const HloComputation* computation :
       {while_instr->while_body(), while_instr->while_condition()}) {
    for (const HloInstruction* user :
         computation->parameter_instruction(0)->users()) {
      if (user->opcode() != HloOpcode::kGetTupleElement) {
        return std::nullopt;
      }
      ++reads[user->tuple_index()];
    }
  }
  return reads;
}

// Whether element `index` of the loop state is a buffer that the body only
// writes, one whole row of at most kMaxRowElements per iteration.
bool IsRowBuffer(const HloInstruction* while_instr, int64_t index,
                 absl::Span<const int64_t> reads) {
  const HloComputation* body = while_instr->while_body();
  const HloInstruction* root = body->root_instruction();
  const HloInstruction* write = root->operand(index);
  // The write is the only reader of the buffer, and the next iteration the
  // only reader of the write, so no other value depends on the buffer.
  return IsRowWrite(write) &&
         ShapeUtil::ElementsIn(write->operand(1)->shape()) <= kMaxRowElements &&
         Match(write->operand(0),
               m::GetTupleElement(m::Op().Is(body->parameter_instruction(0)),
                                  index)) &&
         reads[index] == 1 && write->operand(0)->user_count() == 1 &&
         write->user_count() == 1 &&
         absl::c_count(root->operands(), write) == 1;
}

// `value` with tuple element selections of tuple instructions resolved.
HloInstruction* LookThroughTuples(HloInstruction* value) {
  while (value->opcode() == HloOpcode::kGetTupleElement &&
         value->operand(0)->opcode() == HloOpcode::kTuple) {
    value = value->mutable_operand(0)->mutable_operand(value->tuple_index());
  }
  return value;
}

// `values`, arrays of `rows` rows each, side by side as the columns of one
// array.
HloInstruction* Concatenate(HloComputation* computation,
                            absl::Span<HloInstruction* const> values,
                            int64_t rows) {
  const PrimitiveType type = values.front()->shape().element_type();
  std::vector<HloInstruction*> pieces;
  pieces.reserve(values.size());
  int64_t columns = 0;
  for (HloInstruction* value : values) {
    const int64_t width = ShapeUtil::ElementsIn(value->shape()) / rows;
    pieces.push_back(
        value->shape().dimensions().size() == 2
            ? value
            : computation->AddInstruction(HloInstruction::CreateReshape(
                  ShapeUtil::MakeShape(type, {rows, width}), value)));
    columns += width;
  }
  return computation->AddInstruction(HloInstruction::CreateConcatenate(
      ShapeUtil::MakeShape(type, {rows, columns}), pieces, 1));
}

// `buffer` with `row` written at row `index`.
HloInstruction* WriteRow(HloInstruction* buffer, HloInstruction* row,
                         HloInstruction* index) {
  HloComputation* computation = buffer->parent();
  HloInstruction* zero =
      computation->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::Zero(index->shape().element_type())));
  return computation->AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
      buffer->shape(), buffer, row, {index, zero}));
}

// The buffer of shape `packed_shape` that holds the buffers `values` as its
// columns.
HloInstruction* PackValues(HloComputation* computation,
                           std::vector<HloInstruction*> values,
                           const Shape& packed_shape) {
  for (HloInstruction*& value : values) {
    value = LookThroughTuples(value);
  }
  HloInstruction* first = values.front();
  // Buffers filled with one scalar pack into a buffer filled with it.
  if (absl::c_all_of(values, [&](const HloInstruction* value) {
        return value->opcode() == HloOpcode::kBroadcast &&
               ShapeUtil::IsScalar(value->operand(0)->shape()) &&
               value->operand(0) == first->operand(0);
      })) {
    return computation->AddInstruction(HloInstruction::CreateBroadcast(
        packed_shape, first->mutable_operand(0), {}));
  }
  // Buffers with one row written at the same index, such as the first
  // iteration of a peeled loop, pack into the packed bases with the packed row
  // written.
  if (absl::c_all_of(values, [&](const HloInstruction* value) {
        return IsRowWrite(value) && value->operand(2) == first->operand(2);
      })) {
    std::vector<HloInstruction*> bases, rows;
    for (HloInstruction* value : values) {
      bases.push_back(value->mutable_operand(0));
      rows.push_back(value->mutable_operand(1));
    }
    return WriteRow(PackValues(computation, std::move(bases), packed_shape),
                    Concatenate(computation, rows, /*rows=*/1),
                    first->mutable_operand(2));
  }
  return Concatenate(computation, values, packed_shape.dimensions(0));
}

absl::StatusOr<bool> PackLoop(HloInstruction* while_instr) {
  HloComputation* body = while_instr->while_body();
  // The next state must be a tuple instruction. WhileUtil builds the packed
  // loop on the main thread, without the sharding and the compute type of the
  // loop.
  if (body->root_instruction()->opcode() != HloOpcode::kTuple ||
      !while_instr->parent()->IsMainThread() || while_instr->has_sharding() ||
      while_instr->frontend_attributes().map().contains(kXlaComputeTypeAttr)) {
    return false;
  }
  std::optional<std::vector<int64_t>> reads = CountReads(while_instr);
  if (!reads.has_value()) {
    return false;
  }

  // The buffers of a group share the element type, the row count and the row
  // index instruction, so they take the same row in every iteration.
  const int64_t num_elements = while_instr->shape().tuple_shapes().size();
  std::map<std::tuple<PrimitiveType, int64_t, int64_t>, std::vector<int64_t>>
      candidates;
  for (int64_t index = 0; index < num_elements; ++index) {
    if (IsRowBuffer(while_instr, index, *reads)) {
      const HloInstruction* write = body->root_instruction()->operand(index);
      candidates[{write->shape().element_type(), write->shape().dimensions(0),
                  write->operand(2)->unique_id()}]
          .push_back(index);
    }
  }
  std::vector<std::vector<int64_t>> groups;
  for (auto& [key, group] : candidates) {
    if (group.size() >= kMinBuffersPerGroup) {
      groups.push_back(std::move(group));
    }
  }
  if (groups.empty()) {
    return false;
  }

  HloComputation* parent = while_instr->parent();
  HloInstruction* init = while_instr->mutable_operand(0);
  std::vector<HloInstruction*> packed_inits;
  for (const std::vector<int64_t>& group : groups) {
    const Shape& first = while_instr->shape().tuple_shapes(group.front());
    int64_t columns = 0;
    std::vector<HloInstruction*> inits;
    inits.reserve(group.size());
    for (int64_t index : group) {
      const Shape& shape = while_instr->shape().tuple_shapes(index);
      columns += ShapeUtil::ElementsIn(shape) / shape.dimensions(0);
      inits.push_back(
          init->opcode() == HloOpcode::kTuple
              ? init->mutable_operand(index)
              : parent->AddInstruction(
                    HloInstruction::CreateGetTupleElement(init, index)));
    }
    packed_inits.push_back(
        PackValues(parent, std::move(inits),
                   ShapeUtil::MakeShape(first.element_type(),
                                        {first.dimensions(0), columns})));
  }

  // Each packed buffer becomes a new element of the loop state.
  ABSL_ASSIGN_OR_RETURN(
      WhileUtil::MakeInstructionsLiveInResult result,
      WhileUtil::MakeInstructionsLiveIn(while_instr, packed_inits));
  HloInstruction* new_while = result.new_while_instr;
  HloComputation* new_body = new_while->while_body();
  const auto& body_map = result.while_body_instruction_map;

  for (int64_t g = 0; g < groups.size(); ++g) {
    // The body writes one packed row per iteration.
    std::vector<HloInstruction*> writes, rows;
    for (int64_t index : groups[g]) {
      writes.push_back(
          body_map.at(body->root_instruction()->mutable_operand(index)));
      rows.push_back(writes.back()->mutable_operand(1));
    }
    ABSL_RETURN_IF_ERROR(new_body->root_instruction()->ReplaceOperandWith(
        num_elements + g, WriteRow(result.while_body_live_in_values[g],
                                   Concatenate(new_body, rows, /*rows=*/1),
                                   writes.front()->mutable_operand(2))));

    // Each old buffer passes through the loop unwritten, for
    // WhileLoopSimplifier to remove, and each loop output becomes its columns
    // of the packed buffer.
    HloInstruction* packed = parent->AddInstruction(
        HloInstruction::CreateGetTupleElement(new_while, num_elements + g));
    const int64_t num_rows = packed->shape().dimensions(0);
    int64_t column = 0;
    for (int64_t k = 0; k < groups[g].size(); ++k) {
      ABSL_RETURN_IF_ERROR(
          writes[k]->ReplaceAllUsesWith(writes[k]->mutable_operand(0)));
      ABSL_RETURN_IF_ERROR(new_body->RemoveInstruction(writes[k]));
      HloInstruction* output =
          result.replacement_instr->mutable_operand(groups[g][k]);
      const int64_t width = ShapeUtil::ElementsIn(output->shape()) / num_rows;
      HloInstruction* columns =
          parent->AddInstruction(HloInstruction::CreateSlice(
              ShapeUtil::MakeShape(packed->shape().element_type(),
                                   {num_rows, width}),
              packed, {0, column}, {num_rows, column + width}, {1, 1}));
      if (output->shape().dimensions().size() == 1) {
        columns = parent->AddInstruction(
            HloInstruction::CreateReshape(output->shape(), columns));
      }
      ABSL_RETURN_IF_ERROR(parent->ReplaceInstruction(output, columns));
      column += width;
    }
    VLOG(1) << "Packed " << groups[g].size() << " row buffers of "
            << new_while->name() << " into "
            << ShapeUtil::HumanString(packed->shape());
  }
  return true;
}

}  // namespace

absl::StatusOr<bool> LoopRowBufferPacker::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // Loops of callees come first: packing a loop clones its body, and with it
  // the loops inside.
  std::vector<HloInstruction*> loops;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    absl::c_copy_if(computation->instructions(), std::back_inserter(loops),
                    HloPredicateIsOp<HloOpcode::kWhile>);
  }
  bool changed = false;
  for (HloInstruction* loop : loops) {
    ABSL_ASSIGN_OR_RETURN(bool packed, PackLoop(loop));
    changed |= packed;
  }
  return changed;
}

}  // namespace xla
