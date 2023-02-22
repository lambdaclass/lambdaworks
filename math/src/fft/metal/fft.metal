uint64_t field_mul(uint64_t a, uint64_t b, uint64_t mod) {
  return a * b % mod; 
}

uint64_t field_pow(uint64_t base, uint exp, uint64_t mod) {
    uint64_t result = 1;

    while (exp > 0) {
        if ((exp & 1) == 1) {
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
    result[index] = field_pow(omega[0], index, mod[0]);
}

// TODO: everything else needed.
