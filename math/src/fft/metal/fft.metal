#include <metal_stdlib>

// adapted from ministark
uint64_t mul(const uint64_t lhs, const uint64_t rhs)
{
    // u128 x = u128(lhs) * u128(rhs);
    uint64_t xl = lhs * rhs; // x.low;
    uint64_t xh = metal::mulhi(lhs, rhs);
    uint64_t tmp = xl << 32;
    uint64_t a_overflow = xl > (0xFFFFFFFFFFFFFFFF - tmp);
    uint64_t a = xl + tmp;
    uint64_t b = a - (a >> 32) - a_overflow;
    uint r_underflow = xh < b;
    uint64_t r = xh - b;
    uint adj = -r_underflow;
    return r - adj;
}

uint64_t field_mul(uint64_t a, uint64_t b, uint64_t mod) {
    return mul(a, b) % mod; 
}

uint64_t field_pow(uint64_t base, uint exp, uint64_t mod) {
    uint64_t result = 1;

    while (exp > 0) {
        if (exp & 1) {
            result = field_mul(result, base, mod);
        }
        base = field_mul(base, base, mod);
        exp = exp >> 1;
    }

    return result;
}

kernel void gen_twiddles(
    constant uint64_t* mod [[ buffer(0) ]],
    constant uint64_t* omega [[ buffer(1) ]],
    device uint64_t* result [[ buffer(2) ]],
    uint index [[ thread_position_in_grid ]]
)
{
    result[index] = field_pow(*omega, index, *mod);
}
