# External Comparison Benchmarks

Criterion benchmarks comparing Lambdaworks against [Arkworks](https://github.com/arkworks-rs) and [Plonky3](https://github.com/Plonky3/Plonky3).

## Running

```bash
# All benchmarks
cargo bench --bench criterion_external_comparisons

# Filter by name (regex match on benchmark group/id)
cargo bench --bench criterion_external_comparisons -- "Goldilocks"
cargo bench --bench criterion_external_comparisons -- "BN254 G1"
cargo bench --bench criterion_external_comparisons -- "CFFT"

# Only a specific category
cargo bench --bench criterion_external_comparisons -- "FFT"
cargo bench --bench criterion_external_comparisons -- "Pairing"
cargo bench --bench criterion_external_comparisons -- "MSM"

# List all benchmarks without running them
cargo bench --bench criterion_external_comparisons -- --list
```


## What's compared

### Fields (Lambdaworks vs Plonky3)

| Benchmark | LW field | P3 field | Operations |
|-----------|----------|----------|------------|
| Goldilocks | `Goldilocks64Field` | `p3_goldilocks::Goldilocks` | mul, add, sub, square, inv, pow |
| BabyBear | `Babybear31PrimeField` | `p3_baby_bear::BabyBear` | mul, add, sub, square, inv, pow |
| Mersenne31 | `Mersenne31Field` | `p3_mersenne_31::Mersenne31` | mul, add, sub, square, inv, pow |
| BabyBear Fp4 | `Degree4BabyBearExtensionField` | `BinomialExtensionField<BabyBear, 4>` | mul, add, sub, square, inv |

### Fields (Lambdaworks vs Arkworks)

| Benchmark | Fields | Operations |
|-----------|--------|------------|
| BN254 | Fr (scalar), Fq (base) | mul, add, sub, square, inv, sqrt, pow |
| BLS12-381 | Fr (scalar), Fq (base) | mul, add, sub, square, inv, sqrt, pow |
| Extension fields | Fp2, Fp6, Fp12 (both curves) | mul, add, sub, square, inv |
| Batch inversion | Fr, Fq (both curves) | batch inverse |

### FFT (Lambdaworks vs Plonky3)

| Benchmark | Field | Operations | Sizes |
|-----------|-------|------------|-------|
| Goldilocks FFT | Goldilocks | fft, ifft | 2^12 - 2^18 |
| BabyBear FFT | BabyBear | fft, ifft | 2^12 - 2^18 |
| Mersenne31 CFFT | Mersenne31 | circle fft, circle ifft | 2^12 - 2^18 |
| Coset FFT (LDE) | Goldilocks, BabyBear | coset fft with blowup 2,4,8 | 2^12 - 2^18 |
| Batch FFT | Goldilocks, BabyBear | batch fft (4,8,16,32 polys) | 2^12 - 2^16 |
| IFFT | Goldilocks, BabyBear | ifft, batch ifft | 2^12 - 2^18 |

### Curves (Lambdaworks vs Arkworks)

| Benchmark | Curve | Operations | Sizes |
|-----------|-------|------------|-------|
| G1 ops | BN254, BLS12-381 | add, double, scalar_mul | 100 - 1000 |
| G2 ops | BN254, BLS12-381 | add, double, scalar_mul | 100 - 1000 |
| MSM | BN254, BLS12-381 | multi-scalar multiplication | 2^8 - 2^14 |
| Pairing | BN254, BLS12-381 | pairing (1,2,4,8), miller loop, final exp | - |
| Serialization | BN254, BLS12-381 (G1+G2) | compress, decompress | 100 - 10000 |
| Subgroup check | BN254, BLS12-381 (G1+G2) | subgroup membership | 100 - 10000 |

## Structure

```
external_comparisons/
├── fields/                          # Field arithmetic comparisons
│   ├── goldilocks_comparison.rs     # LW vs P3
│   ├── baby_bear_comparison.rs      # LW vs P3
│   ├── mersenne31_comparison.rs     # LW vs P3
│   ├── babybear_fp4_comparison.rs   # LW vs P3 (extension field)
│   ├── bn254_field_comparison.rs    # LW vs Arkworks (Fr + Fq)
│   ├── bls12_381_field_comparison.rs# LW vs Arkworks (Fr + Fq)
│   ├── extension_field_comparison.rs# LW vs Arkworks (Fp2, Fp6, Fp12)
│   └── batch_inversion_comparison.rs# LW vs Arkworks
├── fft/                             # FFT comparisons
│   ├── goldilocks_fft_comparison.rs # LW vs P3
│   ├── baby_bear_fft_comparison.rs  # LW vs P3
│   ├── mersenne31_cfft_comparison.rs# LW vs P3 (Circle FFT)
│   ├── coset_fft_comparison.rs      # LW vs P3 (LDE)
│   ├── batch_fft_comparison.rs      # LW vs P3
│   └── ifft_comparison.rs           # LW vs P3
└── curves/                          # Elliptic curve comparisons
    ├── bn254_curve_comparison.rs    # LW vs Arkworks (G1)
    ├── bls12_381_curve_comparison.rs# LW vs Arkworks (G1)
    ├── bn254_g2_comparison.rs       # LW vs Arkworks (G2)
    ├── bls12_381_g2_comparison.rs   # LW vs Arkworks (G2)
    ├── bn254_msm_comparison.rs      # LW vs Arkworks
    ├── bls12_381_msm_comparison.rs  # LW vs Arkworks
    ├── bn254_pairing_comparison.rs  # LW vs Arkworks
    ├── bls12_381_pairing_comparison.rs # LW vs Arkworks
    ├── point_serialization_comparison.rs # LW vs Arkworks
    └── subgroup_check_comparison.rs # LW vs Arkworks
```

## Notes

- All benchmarks use seed `0xBEEF` for reproducibility.
- Criterion sample size is 10 (configurable in `criterion_external_comparisons.rs`).
- Plonky3 v0.4.2 (git main branch, pinned in Cargo.lock).
- Arkworks crates are version 0.5.
