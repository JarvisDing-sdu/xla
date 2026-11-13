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

#include "xla/packing.h"

#include <cstddef>
#include <cstdint>

#include "absl/types/span.h"
#include "xla/tsl/lib/math/math_util.h"
#include "xla/tsl/platform/logging.h"

// First undef to prevent error when re-included.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "xla/packing.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace xla {
namespace HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

void Pack4Impl(const uint8_t* HWY_RESTRICT in, size_t count,
               uint8_t* HWY_RESTRICT out) {
  const hn::ScalableTag<uint8_t> d8;
  const size_t lanes = hn::Lanes(d8);
  const auto mask_4 = hn::Set(d8, 0x0F);
  const size_t vector_stride = 2 * lanes;

#if HWY_TARGET <= HWY_AVX3
  // Multiply each pair of adjacent masked bytes by (1, 16) via vpmaddubsw so
  // each 16-bit lane holds low + 16 * high in [0, 255] without saturation,
  // then pack the low bytes and finish any <128-element tail with AVX-512BW
  // masked loads/stores.
  const hn::Rebind<int8_t, decltype(d8)> di8;
  const hn::Repartition<int16_t, decltype(d8)> di16;
  const hn::Half<decltype(d8)> d8_half;
  const auto mul_1_16 = hn::Dup128VecFromValues(di8, 1, 16, 1, 16, 1, 16, 1, 16,
                                                1, 16, 1, 16, 1, 16, 1, 16);

  // Up to lanes inputs: one masked block. The optimizer sees that the products
  // fit in a byte, so DemoteTo becomes a masked vpmovwb store.
  if (count <= lanes) {
    if (count > 0) {
      const auto v = hn::And(hn::LoadN(d8, in, count), mask_4);
      const auto p = hn::SatWidenMulPairwiseAdd(di16, v, mul_1_16);
      hn::StoreN(hn::DemoteTo(d8_half, p), d8_half, out, (count + 1) / 2);
    }
    return;
  }

  auto mul = mul_1_16;
#if HWY_COMPILER_GCC && HWY_TARGET <= HWY_AVX3_DL && HWY_TARGET != HWY_AVX3_ZEN4
  // Hide mul from the optimizer. Otherwise LLVM's known bits for vpmaddubsw
  // prove that vpackuswb cannot saturate, and with AVX512VBMI it fuses
  // vpackuswb + vpermq into one vpermt2b, which is 3 uops (2 on port 5) on
  // Intel instead of 2. On AMD Zen4 the fused form is faster.
  __asm__("" : "+v"(mul.raw));
#endif
  // Packs vector_stride inputs at src into lanes bytes at dst.
  const auto pack_block = [&](const uint8_t* src, uint8_t* dst) HWY_ATTR {
    const auto v0 = hn::And(hn::LoadU(d8, src), mask_4);
    const auto v1 = hn::And(hn::LoadU(d8, src + lanes), mask_4);
    const auto p0 = hn::SatWidenMulPairwiseAdd(di16, v0, mul);
    const auto p1 = hn::SatWidenMulPairwiseAdd(di16, v1, mul);
    hn::StoreU(hn::OrderedDemote2To(d8, p0, p1), d8, dst);
  };
  size_t remaining = count;
  // Inputs shorter than one block skip both loops with a single compare.
  if (remaining >= vector_stride) {
    // Unrolled 4x by hand, then up to 3 leftover blocks one at a time.
    // HWY_UNROLL(1) stops LLVM from adding its own runtime unrolling, whose
    // trip-count setup and remainder loop slow down small inputs.
    HWY_UNROLL(1)
    for (; remaining >= 4 * vector_stride; in += 4 * vector_stride,
                                           out += 4 * lanes,
                                           remaining -= 4 * vector_stride) {
      pack_block(in, out);
      pack_block(in + vector_stride, out + lanes);
      pack_block(in + 2 * vector_stride, out + 2 * lanes);
      pack_block(in + 3 * vector_stride, out + 3 * lanes);
    }
    HWY_UNROLL(1)
    for (; remaining >= vector_stride;
         in += vector_stride, out += lanes, remaining -= vector_stride) {
      pack_block(in, out);
    }
  }
  if (remaining > lanes) {
    const auto v0 = hn::And(hn::LoadU(d8, in), mask_4);
    const auto v1 =
        hn::And(hn::LoadN(d8, in + lanes, remaining - lanes), mask_4);
    const auto p0 = hn::SatWidenMulPairwiseAdd(di16, v0, mul);
    const auto p1 = hn::SatWidenMulPairwiseAdd(di16, v1, mul);
    hn::StoreN(hn::OrderedDemote2To(d8, p0, p1), d8, out, (remaining + 1) / 2);
  } else if (remaining > 0) {
    const auto v = hn::And(hn::LoadN(d8, in, remaining), mask_4);
    const auto p = hn::SatWidenMulPairwiseAdd(di16, v, mul);
    hn::StoreN(hn::DemoteTo(d8_half, p), d8_half, out, (remaining + 1) / 2);
  }
#else
  const size_t vector_inputs = (count / vector_stride) * vector_stride;

  for (size_t i = 0; i < vector_inputs; i += vector_stride) {
    hn::Vec<decltype(d8)> v0, v1;
    hn::LoadInterleaved2(d8, in + i, v0, v1);
    auto packed =
        hn::Or(hn::And(v0, mask_4), hn::ShiftLeft<4>(hn::And(v1, mask_4)));
    hn::StoreU(packed, d8, out + i / 2);
  }

  constexpr uint8_t mask = 0x0F;
  const size_t aligned_inputs = count / 2;
  for (size_t i = vector_inputs / 2; i < aligned_inputs; ++i) {
    out[i] = (in[2 * i] & mask) | ((in[2 * i + 1] & mask) << 4);
  }
  if (count % 2 != 0) {
    out[aligned_inputs] = in[2 * aligned_inputs] & mask;
  }
#endif
}

void Pack2Impl(const uint8_t* HWY_RESTRICT in, size_t count,
               uint8_t* HWY_RESTRICT out) {
  const hn::ScalableTag<uint8_t> d8;
  const size_t lanes = hn::Lanes(d8);
  const auto mask_2 = hn::Set(d8, 0x03);
  const size_t vector_stride = 4 * lanes;
  const size_t vector_inputs = (count / vector_stride) * vector_stride;

  for (size_t i = 0; i < vector_inputs; i += vector_stride) {
    hn::Vec<decltype(d8)> v0, v1, v2, v3;
    hn::LoadInterleaved4(d8, in + i, v0, v1, v2, v3);
    auto p01 =
        hn::Or(hn::And(v0, mask_2), hn::ShiftLeft<2>(hn::And(v1, mask_2)));
    auto p23 = hn::Or(hn::ShiftLeft<4>(hn::And(v2, mask_2)),
                      hn::ShiftLeft<6>(hn::And(v3, mask_2)));
    auto packed = hn::Or(p01, p23);
    hn::StoreU(packed, d8, out + i / 4);
  }

  constexpr uint8_t mask = 0x03;
  const size_t aligned_inputs = count / 4;
  for (size_t i = vector_inputs / 4; i < aligned_inputs; ++i) {
    out[i] = (in[4 * i] & mask) | ((in[4 * i + 1] & mask) << 2) |
             ((in[4 * i + 2] & mask) << 4) | ((in[4 * i + 3] & mask) << 6);
  }
  if (const size_t remainder = count % 4; remainder != 0) {
    uint8_t byte = 0;
    for (size_t j = 0; j < remainder; ++j) {
      byte |= (in[4 * aligned_inputs + j] & mask) << (2 * j);
    }
    out[aligned_inputs] = byte;
  }
}

void Pack1Impl(const uint8_t* HWY_RESTRICT in, size_t count,
               uint8_t* HWY_RESTRICT out) {
  const hn::ScalableTag<uint8_t> d8;
  const size_t lanes = hn::Lanes(d8);
  const auto mask_1 = hn::Set(d8, 0x01);
  const size_t vector_inputs = lanes >= 8 ? (count / lanes) * lanes : 0;

  for (size_t i = 0; i < vector_inputs; i += lanes) {
    auto v = hn::LoadU(d8, in + i);
    auto m = hn::TestBit(v, mask_1);
    hn::StoreMaskBits(d8, m, out + i / 8);
  }

  const size_t aligned_inputs = count / 8;
  for (size_t i = vector_inputs / 8; i < aligned_inputs; ++i) {
    uint8_t byte = 0;
    for (size_t j = 0; j < 8; ++j) {
      byte |= (in[8 * i + j] & 1) << j;
    }
    out[i] = byte;
  }
  if (const size_t remainder = count % 8; remainder != 0) {
    uint8_t byte = 0;
    for (size_t j = 0; j < remainder; ++j) {
      byte |= (in[8 * aligned_inputs + j] & 1) << j;
    }
    out[aligned_inputs] = byte;
  }
}

void Unpack4Impl(const uint8_t* HWY_RESTRICT in, size_t count,
                 uint8_t* HWY_RESTRICT out) {
  const hn::ScalableTag<uint8_t> d8;
  const size_t lanes = hn::Lanes(d8);

#if HWY_TARGET <= HWY_AVX3
  // Zero-extend each packed byte into a 16-bit lane (0x00HL), then expand
  // (wide | (wide << 4)) & 0x0F0F -> 0x0H0L ([0x0L, 0x0H] in little-endian).
  // Finish any <64-element tail with a single AVX-512BW masked load/store.
  const hn::Half<decltype(d8)> d8_half;
  const hn::Repartition<uint16_t, decltype(d8)> du16;
  const auto mask_0f0f = hn::Set(du16, uint16_t{0x0F0F});
  // Unpacks lanes / 2 bytes at src into lanes outputs at dst.
  const auto unpack_vector = [&](const uint8_t* src, uint8_t* dst) HWY_ATTR {
    const auto wide = hn::PromoteTo(du16, hn::LoadU(d8_half, src));
    const auto unpacked =
        hn::And(hn::Or(wide, hn::ShiftLeft<4>(wide)), mask_0f0f);
    hn::StoreU(hn::BitCast(d8, unpacked), d8, dst);
  };
  size_t remaining = count;
  if constexpr (HWY_TARGET <= HWY_AVX3_DL) {
    // Same loop structure as in Pack4Impl, for the same reasons.
    if (remaining >= lanes) {
      HWY_UNROLL(1)
      for (; remaining >= 4 * lanes;
           in += 2 * lanes, out += 4 * lanes, remaining -= 4 * lanes) {
        unpack_vector(in, out);
        unpack_vector(in + lanes / 2, out + lanes);
        unpack_vector(in + lanes, out + 2 * lanes);
        unpack_vector(in + 3 * lanes / 2, out + 3 * lanes);
      }
      HWY_UNROLL(1)
      for (; remaining >= lanes;
           in += lanes / 2, out += lanes, remaining -= lanes) {
        unpack_vector(in, out);
      }
    }
  } else {
    // AVX3 (Skylake-SP, Cascade Lake) keeps the plain loop, which LLVM
    // unrolls itself; the loops above measured 4.7% slower there at 10000
    // outputs.
    for (; remaining >= lanes;
         in += lanes / 2, out += lanes, remaining -= lanes) {
      unpack_vector(in, out);
    }
  }
  if (remaining > 0) {
    const auto wide =
        hn::PromoteTo(du16, hn::LoadN(d8_half, in, (remaining + 1) / 2));
    const auto unpacked =
        hn::And(hn::Or(wide, hn::ShiftLeft<4>(wide)), mask_0f0f);
    hn::StoreN(hn::BitCast(d8, unpacked), d8, out, remaining);
  }
#else
  const auto mask_4 = hn::Set(d8, 0x0F);
  const size_t vector_stride = 2 * lanes;
  const size_t vector_outputs = (count / vector_stride) * vector_stride;

  for (size_t i = 0; i < vector_outputs; i += vector_stride) {
    auto packed = hn::LoadU(d8, in + i / 2);
    auto low = hn::And(packed, mask_4);
    auto high = hn::And(hn::ShiftRight<4>(packed), mask_4);
    hn::StoreInterleaved2(low, high, d8, out + i);
  }

  constexpr uint8_t mask = 0x0F;
  const size_t aligned_outputs = count / 2;
  for (size_t i = vector_outputs / 2; i < aligned_outputs; ++i) {
    const uint8_t byte = in[i];
    out[2 * i] = byte & mask;
    out[2 * i + 1] = (byte >> 4) & mask;
  }
  if (count % 2 != 0) {
    out[2 * aligned_outputs] = in[aligned_outputs] & mask;
  }
#endif
}

void Unpack2Impl(const uint8_t* HWY_RESTRICT in, size_t count,
                 uint8_t* HWY_RESTRICT out) {
  const hn::ScalableTag<uint8_t> d8;
  const size_t lanes = hn::Lanes(d8);
  const auto mask_2 = hn::Set(d8, 0x03);
  const size_t vector_stride = 4 * lanes;
  const size_t vector_outputs = (count / vector_stride) * vector_stride;

  for (size_t i = 0; i < vector_outputs; i += vector_stride) {
    auto packed = hn::LoadU(d8, in + i / 4);
    auto b0 = hn::And(packed, mask_2);
    auto b1 = hn::And(hn::ShiftRight<2>(packed), mask_2);
    auto b2 = hn::And(hn::ShiftRight<4>(packed), mask_2);
    auto b3 = hn::And(hn::ShiftRight<6>(packed), mask_2);
    hn::StoreInterleaved4(b0, b1, b2, b3, d8, out + i);
  }

  constexpr uint8_t mask = 0x03;
  const size_t aligned_outputs = count / 4;
  for (size_t i = vector_outputs / 4; i < aligned_outputs; ++i) {
    const uint8_t byte = in[i];
    out[4 * i] = byte & mask;
    out[4 * i + 1] = (byte >> 2) & mask;
    out[4 * i + 2] = (byte >> 4) & mask;
    out[4 * i + 3] = (byte >> 6) & mask;
  }
  if (const size_t remainder = count % 4; remainder != 0) {
    const uint8_t byte = in[aligned_outputs];
    for (size_t j = 0; j < remainder; ++j) {
      out[4 * aligned_outputs + j] = (byte >> (2 * j)) & mask;
    }
  }
}

void Unpack1Impl(const uint8_t* HWY_RESTRICT in, size_t count,
                 uint8_t* HWY_RESTRICT out) {
  const hn::ScalableTag<uint8_t> d8;
  const size_t lanes = hn::Lanes(d8);
  const auto one = hn::Set(d8, 0x01);
  const size_t vector_outputs = lanes >= 8 ? (count / lanes) * lanes : 0;

  for (size_t i = 0; i < vector_outputs; i += lanes) {
    auto m = hn::LoadMaskBits(d8, in + i / 8);
    auto v = hn::IfThenElseZero(m, one);
    hn::StoreU(v, d8, out + i);
  }

  const size_t aligned_outputs = count / 8;
  for (size_t i = vector_outputs / 8; i < aligned_outputs; ++i) {
    const uint8_t byte = in[i];
    for (int j = 0; j < 8; ++j) {
      out[8 * i + j] = (byte >> j) & 1;
    }
  }
  if (const size_t remainder = count % 8; remainder != 0) {
    const uint8_t byte = in[aligned_outputs];
    for (size_t j = 0; j < remainder; ++j) {
      out[8 * aligned_outputs + j] = (byte >> j) & 1;
    }
  }
}

}  // namespace HWY_NAMESPACE
}  // namespace xla
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace xla {

HWY_EXPORT(Pack4Impl);
HWY_EXPORT(Pack2Impl);
HWY_EXPORT(Pack1Impl);
HWY_EXPORT(Unpack4Impl);
HWY_EXPORT(Unpack2Impl);
HWY_EXPORT(Unpack1Impl);

template <>
void PackIntNHwy<4>(absl::Span<const char> input, absl::Span<char> output) {
  constexpr size_t kElementsPerByte = 2;
  const size_t required_output_size =
      tsl::MathUtil::CeilOfRatio(input.size(), kElementsPerByte);
  CHECK_GE(output.size(), required_output_size)
      << "Output span too small for packed elements: " << output.size() << " < "
      << required_output_size;
  HWY_DYNAMIC_DISPATCH(Pack4Impl)(
      reinterpret_cast<const uint8_t*>(input.data()), input.size(),
      reinterpret_cast<uint8_t*>(output.data()));
}

template <>
void PackIntNHwy<2>(absl::Span<const char> input, absl::Span<char> output) {
  constexpr size_t kElementsPerByte = 4;
  const size_t required_output_size =
      tsl::MathUtil::CeilOfRatio(input.size(), kElementsPerByte);
  CHECK_GE(output.size(), required_output_size)
      << "Output span too small for packed elements: " << output.size() << " < "
      << required_output_size;
  HWY_DYNAMIC_DISPATCH(Pack2Impl)(
      reinterpret_cast<const uint8_t*>(input.data()), input.size(),
      reinterpret_cast<uint8_t*>(output.data()));
}

template <>
void PackIntNHwy<1>(absl::Span<const char> input, absl::Span<char> output) {
  constexpr size_t kElementsPerByte = 8;
  const size_t required_output_size =
      tsl::MathUtil::CeilOfRatio(input.size(), kElementsPerByte);
  CHECK_GE(output.size(), required_output_size)
      << "Output span too small for packed elements: " << output.size() << " < "
      << required_output_size;
  HWY_DYNAMIC_DISPATCH(Pack1Impl)(
      reinterpret_cast<const uint8_t*>(input.data()), input.size(),
      reinterpret_cast<uint8_t*>(output.data()));
}

void PackIntNHwy(int bits_per_element, absl::Span<const char> input,
                 absl::Span<char> output) {
  if (bits_per_element == 1) {
    PackIntNHwy<1>(input, output);
  } else if (bits_per_element == 2) {
    PackIntNHwy<2>(input, output);
  } else if (bits_per_element == 4) {
    PackIntNHwy<4>(input, output);
  } else {
    LOG(FATAL) << "Invalid bits_per_element: " << bits_per_element;
  }
}

template <>
void UnpackIntNHwy<4>(absl::Span<const char> input, absl::Span<char> output) {
  constexpr size_t kElementsPerByte = 2;
  const size_t required_input_size =
      tsl::MathUtil::CeilOfRatio(output.size(), kElementsPerByte);
  CHECK_GE(input.size(), required_input_size)
      << "Input span too small for unpacked elements: " << input.size() << " < "
      << required_input_size;
  HWY_DYNAMIC_DISPATCH(Unpack4Impl)(
      reinterpret_cast<const uint8_t*>(input.data()), output.size(),
      reinterpret_cast<uint8_t*>(output.data()));
}

template <>
void UnpackIntNHwy<2>(absl::Span<const char> input, absl::Span<char> output) {
  constexpr size_t kElementsPerByte = 4;
  const size_t required_input_size =
      tsl::MathUtil::CeilOfRatio(output.size(), kElementsPerByte);
  CHECK_GE(input.size(), required_input_size)
      << "Input span too small for unpacked elements: " << input.size() << " < "
      << required_input_size;
  HWY_DYNAMIC_DISPATCH(Unpack2Impl)(
      reinterpret_cast<const uint8_t*>(input.data()), output.size(),
      reinterpret_cast<uint8_t*>(output.data()));
}

template <>
void UnpackIntNHwy<1>(absl::Span<const char> input, absl::Span<char> output) {
  constexpr size_t kElementsPerByte = 8;
  const size_t required_input_size =
      tsl::MathUtil::CeilOfRatio(output.size(), kElementsPerByte);
  CHECK_GE(input.size(), required_input_size)
      << "Input span too small for unpacked elements: " << input.size() << " < "
      << required_input_size;
  HWY_DYNAMIC_DISPATCH(Unpack1Impl)(
      reinterpret_cast<const uint8_t*>(input.data()), output.size(),
      reinterpret_cast<uint8_t*>(output.data()));
}

void UnpackIntNHwy(int bits_per_element, absl::Span<const char> input,
                   absl::Span<char> output) {
  if (bits_per_element == 1) {
    UnpackIntNHwy<1>(input, output);
  } else if (bits_per_element == 2) {
    UnpackIntNHwy<2>(input, output);
  } else if (bits_per_element == 4) {
    UnpackIntNHwy<4>(input, output);
  } else {
    LOG(FATAL) << "Invalid bits_per_element: " << bits_per_element;
  }
}

}  // namespace xla
#endif  // HWY_ONCE
