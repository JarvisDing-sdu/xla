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

#ifndef XLA_PJRT_DYNAMIC_PAD_TO_STATIC_H_
#define XLA_PJRT_DYNAMIC_PAD_TO_STATIC_H_

#include <cstddef>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "xla/shape.h"

namespace xla {

class HloModule;

// Clears the metadata prefix of `parameter_shapes` read by `PadRealToStatic`
// (including across offloaded calls); fails if the parameter has other uses.
absl::Status ClearPadRealToStaticParameterMetadataPrefixes(
    const HloModule& hlo_module, absl::Span<Shape> parameter_shapes);

// Returns true if dynamic `main` argument `arg_index` feeds `PadRealToStatic`
// and should have its metadata prefix cleared; fails on shared or aliased use.
absl::StatusOr<bool> ShouldClearPadRealToStaticMetadataPrefix(
    mlir::func::FuncOp main, size_t arg_index);

}  // namespace xla

#endif  // XLA_PJRT_DYNAMIC_PAD_TO_STATIC_H_
