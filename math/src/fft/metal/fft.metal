constant uint64_t omega [[ function_constant(0) ]];
constant uint64_t mod [[ function_constant(1) ]];

uint64_t field_mul(uint64_t a, uint64_t b) {
  return a * b % mod; 
}

uint64_t field_pow(uint64_t base, uint exp) {
    uint64_t result = 1;

    while (exp > 0) {
        if ((exp & 1) == 1) {
            result = field_mul(result, base);
        }
        base = field_mul(base, base);
        exp = exp >> 1;
    }

    return result;
}

kernel void gen_twiddles(
    device uint64_t* result,
    uint index [[ thread_position_in_grid ]]
)
{
    result[index] = field_pow(omega, index);
}

// TODO: everything else needed.
