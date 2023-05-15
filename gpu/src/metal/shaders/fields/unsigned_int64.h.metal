#ifndef unsigned_int64_h
#define unsigned_int64_h

#include <metal_stdlib>

template <const uint64_t NUM_LIMBS>
struct UnsignedInteger64 {
    metal::array<uint64_t, NUM_LIMBS> limbs;

    constexpr UnsignedInteger64() = default;
};

#endif /* unsigned_int64_h */
