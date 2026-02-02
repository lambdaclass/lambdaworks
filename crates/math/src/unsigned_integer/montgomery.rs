use super::element::UnsignedInteger;

pub struct MontgomeryAlgorithms;

// ARM64 assembly implementations for Montgomery multiplication
#[cfg(all(target_arch = "aarch64", feature = "asm"))]
mod aarch64_asm {
    use core::arch::asm;

    /// ARM64 inline assembly for 4-limb addition: r = a + b
    /// Returns (result, overflow)
    /// Uses ADDS/ADCS chain for efficient carry propagation
    /// Note: lambdaworks uses big-endian limb order (limbs[0] = MSB)
    #[inline(always)]
    pub fn add_4_limbs_asm(a: &[u64; 4], b: &[u64; 4]) -> ([u64; 4], bool) {
        let mut r = [0u64; 4];
        let overflow: u64;

        unsafe {
            asm!(
                // Load a[3], a[2], a[1], a[0] (LSB to MSB order for addition)
                // Add with carry chain starting from LSB (index 3)
                "adds {r3}, {a3}, {b3}",     // r[3] = a[3] + b[3], set carry
                "adcs {r2}, {a2}, {b2}",     // r[2] = a[2] + b[2] + carry
                "adcs {r1}, {a1}, {b1}",     // r[1] = a[1] + b[1] + carry
                "adcs {r0}, {a0}, {b0}",     // r[0] = a[0] + b[0] + carry
                "adc {ovf}, xzr, xzr",       // capture final overflow

                a0 = in(reg) a[0],
                a1 = in(reg) a[1],
                a2 = in(reg) a[2],
                a3 = in(reg) a[3],
                b0 = in(reg) b[0],
                b1 = in(reg) b[1],
                b2 = in(reg) b[2],
                b3 = in(reg) b[3],
                r0 = out(reg) r[0],
                r1 = out(reg) r[1],
                r2 = out(reg) r[2],
                r3 = out(reg) r[3],
                ovf = out(reg) overflow,
                options(pure, nomem, nostack),
            );
        }

        (r, overflow != 0)
    }

    /// ARM64 inline assembly for 4-limb subtraction: r = a - b
    /// Returns (result, borrow)
    /// Uses SUBS/SBCS chain for efficient borrow propagation
    #[inline(always)]
    pub fn sub_4_limbs_asm(a: &[u64; 4], b: &[u64; 4]) -> ([u64; 4], bool) {
        let mut r = [0u64; 4];
        let no_borrow: u64;

        unsafe {
            asm!(
                // Subtract with borrow chain starting from LSB (index 3)
                "subs {r3}, {a3}, {b3}",     // r[3] = a[3] - b[3], set borrow
                "sbcs {r2}, {a2}, {b2}",     // r[2] = a[2] - b[2] - borrow
                "sbcs {r1}, {a1}, {b1}",     // r[1] = a[1] - b[1] - borrow
                "sbcs {r0}, {a0}, {b0}",     // r[0] = a[0] - b[0] - borrow
                "cset {nb}, cs",             // nb = 1 if no borrow (carry set), 0 if borrow

                a0 = in(reg) a[0],
                a1 = in(reg) a[1],
                a2 = in(reg) a[2],
                a3 = in(reg) a[3],
                b0 = in(reg) b[0],
                b1 = in(reg) b[1],
                b2 = in(reg) b[2],
                b3 = in(reg) b[3],
                r0 = out(reg) r[0],
                r1 = out(reg) r[1],
                r2 = out(reg) r[2],
                r3 = out(reg) r[3],
                nb = out(reg) no_borrow,
                options(pure, nomem, nostack),
            );
        }

        (r, no_borrow == 0) // borrow occurred if no_borrow is 0
    }

    /// ARM64 inline assembly for 6-limb addition: r = a + b
    /// Returns (result, overflow)
    #[inline(always)]
    pub fn add_6_limbs_asm(a: &[u64; 6], b: &[u64; 6]) -> ([u64; 6], bool) {
        let mut r = [0u64; 6];
        let overflow: u64;

        unsafe {
            asm!(
                // Add with carry chain starting from LSB (index 5)
                "adds {r5}, {a5}, {b5}",
                "adcs {r4}, {a4}, {b4}",
                "adcs {r3}, {a3}, {b3}",
                "adcs {r2}, {a2}, {b2}",
                "adcs {r1}, {a1}, {b1}",
                "adcs {r0}, {a0}, {b0}",
                "adc {ovf}, xzr, xzr",

                a0 = in(reg) a[0],
                a1 = in(reg) a[1],
                a2 = in(reg) a[2],
                a3 = in(reg) a[3],
                a4 = in(reg) a[4],
                a5 = in(reg) a[5],
                b0 = in(reg) b[0],
                b1 = in(reg) b[1],
                b2 = in(reg) b[2],
                b3 = in(reg) b[3],
                b4 = in(reg) b[4],
                b5 = in(reg) b[5],
                r0 = out(reg) r[0],
                r1 = out(reg) r[1],
                r2 = out(reg) r[2],
                r3 = out(reg) r[3],
                r4 = out(reg) r[4],
                r5 = out(reg) r[5],
                ovf = out(reg) overflow,
                options(pure, nomem, nostack),
            );
        }

        (r, overflow != 0)
    }

    /// ARM64 inline assembly for 6-limb subtraction: r = a - b
    /// Returns (result, borrow)
    #[inline(always)]
    pub fn sub_6_limbs_asm(a: &[u64; 6], b: &[u64; 6]) -> ([u64; 6], bool) {
        let mut r = [0u64; 6];
        let no_borrow: u64;

        unsafe {
            asm!(
                "subs {r5}, {a5}, {b5}",
                "sbcs {r4}, {a4}, {b4}",
                "sbcs {r3}, {a3}, {b3}",
                "sbcs {r2}, {a2}, {b2}",
                "sbcs {r1}, {a1}, {b1}",
                "sbcs {r0}, {a0}, {b0}",
                "cset {nb}, cs",

                a0 = in(reg) a[0],
                a1 = in(reg) a[1],
                a2 = in(reg) a[2],
                a3 = in(reg) a[3],
                a4 = in(reg) a[4],
                a5 = in(reg) a[5],
                b0 = in(reg) b[0],
                b1 = in(reg) b[1],
                b2 = in(reg) b[2],
                b3 = in(reg) b[3],
                b4 = in(reg) b[4],
                b5 = in(reg) b[5],
                r0 = out(reg) r[0],
                r1 = out(reg) r[1],
                r2 = out(reg) r[2],
                r3 = out(reg) r[3],
                r4 = out(reg) r[4],
                r5 = out(reg) r[5],
                nb = out(reg) no_borrow,
                options(pure, nomem, nostack),
            );
        }

        (r, no_borrow == 0)
    }

    /// ARM64 optimized CIOS Montgomery multiplication for 4 limbs (256-bit)
    /// Uses pure Rust with u128 arithmetic which LLVM optimizes well for ARM64
    /// Note: lambdaworks uses big-endian limb order (limbs[0] = MSB)
    #[inline(always)]
    pub fn cios_4_limbs(a: &[u64; 4], b: &[u64; 4], q: &[u64; 4], mu: u64) -> [u64; 4] {
        // Use the same algorithm as the generic CIOS but specialized for 4 limbs
        // LLVM generates efficient ARM64 code with MUL/UMULH for u128 operations
        const N: usize = 4;
        let mut t = [0u64; N];
        let mut t_extra = [0u64; 2];

        // Iterate from LSB to MSB (i = N-1 down to 0)
        for i in (0..N).rev() {
            // t += a * b[i]
            let mut c: u128 = 0;
            for j in (0..N).rev() {
                let cs = t[j] as u128 + (a[j] as u128) * (b[i] as u128) + c;
                c = cs >> 64;
                t[j] = cs as u64;
            }
            let cs = (t_extra[1] as u128) + c;
            t_extra[0] = (cs >> 64) as u64;
            t_extra[1] = cs as u64;

            // m := t[N-1] * mu mod 2^64
            let m = t[N - 1].wrapping_mul(mu) as u128;

            // t += m * q, then shift right by 64
            let mut c: u128 = (t[N - 1] as u128 + m * (q[N - 1] as u128)) >> 64;
            for j in (0..N - 1).rev() {
                let cs = t[j] as u128 + m * (q[j] as u128) + c;
                c = cs >> 64;
                t[j + 1] = cs as u64;
            }
            let cs = (t_extra[1] as u128) + c;
            c = cs >> 64;
            t[0] = cs as u64;
            t_extra[1] = t_extra[0] + c as u64;
            t_extra[0] = 0;
        }

        // Final reduction
        let overflow = t_extra[1] > 0;
        if overflow || const_ge_4(&t, q) {
            sub_4(&mut t, q);
        }
        t
    }

    /// ARM64 optimized CIOS Montgomery multiplication for 6 limbs (384-bit)
    #[inline(always)]
    pub fn cios_6_limbs(a: &[u64; 6], b: &[u64; 6], q: &[u64; 6], mu: u64) -> [u64; 6] {
        const N: usize = 6;
        let mut t = [0u64; N];
        let mut t_extra = [0u64; 2];

        for i in (0..N).rev() {
            // t += a * b[i]
            let mut c: u128 = 0;
            for j in (0..N).rev() {
                let cs = t[j] as u128 + (a[j] as u128) * (b[i] as u128) + c;
                c = cs >> 64;
                t[j] = cs as u64;
            }
            let cs = (t_extra[1] as u128) + c;
            t_extra[0] = (cs >> 64) as u64;
            t_extra[1] = cs as u64;

            // m := t[N-1] * mu mod 2^64
            let m = t[N - 1].wrapping_mul(mu) as u128;

            // t += m * q, then shift right by 64
            let mut c: u128 = (t[N - 1] as u128 + m * (q[N - 1] as u128)) >> 64;
            for j in (0..N - 1).rev() {
                let cs = t[j] as u128 + m * (q[j] as u128) + c;
                c = cs >> 64;
                t[j + 1] = cs as u64;
            }
            let cs = (t_extra[1] as u128) + c;
            c = cs >> 64;
            t[0] = cs as u64;
            t_extra[1] = t_extra[0] + c as u64;
            t_extra[0] = 0;
        }

        // Final reduction
        let overflow = t_extra[1] > 0;
        if overflow || const_ge_6(&t, q) {
            sub_6(&mut t, q);
        }
        t
    }

    #[inline(always)]
    fn const_ge_4(a: &[u64; 4], b: &[u64; 4]) -> bool {
        for i in 0..4 {
            if a[i] > b[i] {
                return true;
            }
            if a[i] < b[i] {
                return false;
            }
        }
        true // equal
    }

    #[inline(always)]
    fn const_ge_6(a: &[u64; 6], b: &[u64; 6]) -> bool {
        for i in 0..6 {
            if a[i] > b[i] {
                return true;
            }
            if a[i] < b[i] {
                return false;
            }
        }
        true // equal
    }

    #[inline(always)]
    fn sub_4(a: &mut [u64; 4], b: &[u64; 4]) {
        let (result, _) = sub_4_limbs_asm(a, b);
        *a = result;
    }

    #[inline(always)]
    fn sub_6(a: &mut [u64; 6], b: &[u64; 6]) {
        let (result, _) = sub_6_limbs_asm(a, b);
        *a = result;
    }

    /// Modular addition for 4 limbs: r = (a + b) mod q
    /// Uses conditional subtraction for reduction
    #[inline(always)]
    pub fn mod_add_4_limbs(a: &[u64; 4], b: &[u64; 4], q: &[u64; 4]) -> [u64; 4] {
        let (sum, overflow) = add_4_limbs_asm(a, b);
        if overflow || const_ge_4(&sum, q) {
            let (reduced, _) = sub_4_limbs_asm(&sum, q);
            reduced
        } else {
            sum
        }
    }

    /// Modular subtraction for 4 limbs: r = (a - b) mod q
    /// Adds modulus if borrow occurs
    #[inline(always)]
    pub fn mod_sub_4_limbs(a: &[u64; 4], b: &[u64; 4], q: &[u64; 4]) -> [u64; 4] {
        let (diff, borrow) = sub_4_limbs_asm(a, b);
        if borrow {
            let (result, _) = add_4_limbs_asm(&diff, q);
            result
        } else {
            diff
        }
    }

    /// Modular addition for 6 limbs: r = (a + b) mod q
    #[inline(always)]
    pub fn mod_add_6_limbs(a: &[u64; 6], b: &[u64; 6], q: &[u64; 6]) -> [u64; 6] {
        let (sum, overflow) = add_6_limbs_asm(a, b);
        if overflow || const_ge_6(&sum, q) {
            let (reduced, _) = sub_6_limbs_asm(&sum, q);
            reduced
        } else {
            sum
        }
    }

    /// Modular subtraction for 6 limbs: r = (a - b) mod q
    #[inline(always)]
    pub fn mod_sub_6_limbs(a: &[u64; 6], b: &[u64; 6], q: &[u64; 6]) -> [u64; 6] {
        let (diff, borrow) = sub_6_limbs_asm(a, b);
        if borrow {
            let (result, _) = add_6_limbs_asm(&diff, q);
            result
        } else {
            diff
        }
    }

    /// ARM64 optimized CIOS for 4 limbs with spare bit optimization (EdMSM Algorithm 2)
    /// For moduli where the high limb is < 2^63 - 1
    /// Uses pure Rust with u128 (LLVM generates good ARM64 code)
    #[inline(always)]
    pub fn cios_4_limbs_optimized(a: &[u64; 4], b: &[u64; 4], q: &[u64; 4], mu: u64) -> [u64; 4] {
        const N: usize = 4;
        let mut t = [0u64; N];

        for i in (0..N).rev() {
            // t += a * b[i]
            let mut c: u128 = 0;
            for j in (0..N).rev() {
                let cs = t[j] as u128 + (a[j] as u128) * (b[i] as u128) + c;
                c = cs >> 64;
                t[j] = cs as u64;
            }
            let t_extra = c as u64;

            // m := t[N-1] * mu mod 2^64
            let m = t[N - 1].wrapping_mul(mu) as u128;

            // t += m * q, then shift right by 64
            let mut c: u128 = (t[N - 1] as u128 + m * (q[N - 1] as u128)) >> 64;
            for j in (0..N - 1).rev() {
                let cs = t[j] as u128 + m * (q[j] as u128) + c;
                c = cs >> 64;
                t[j + 1] = cs as u64;
            }
            let cs = (t_extra as u128) + c;
            t[0] = cs as u64;
        }

        // Final reduction (no overflow possible with spare bit)
        if const_ge_4(&t, q) {
            sub_4(&mut t, q);
        }
        t
    }

    /// ARM64 inline assembly CIOS for 4 limbs with spare bit optimization
    /// Implements EdMSM Algorithm 2 using true ARM64 inline assembly
    ///
    /// Register allocation strategy:
    /// - t0-t3: Accumulator limbs (t[0] = MSB, t[3] = LSB in lambdaworks convention)
    /// - a0-a3: Operand a limbs (loaded once)
    /// - q0-q3: Modulus q limbs (loaded once)
    /// - bi: Current b[i] limb (changes each iteration)
    /// - mu: Montgomery constant
    /// - m: Reduction factor m = t[3] * mu
    /// - lo, hi: Temporary for MUL/UMULH results
    /// - c: Carry accumulator
    ///
    /// Key optimizations:
    /// - Unrolled inner loops for all 4 iterations
    /// - MUL/UMULH pairs for 64x64->128 multiplication
    /// - ADDS/ADCS chains for efficient carry propagation
    /// - No overflow tracking (spare bit guarantee)
    /// - Constant-time conditional reduction using CSEL
    #[inline(always)]
    pub fn cios_4_limbs_asm_optimized(
        a: &[u64; 4],
        b: &[u64; 4],
        q: &[u64; 4],
        mu: u64,
    ) -> [u64; 4] {
        let mut t = [0u64; 4];

        // Note: lambdaworks uses big-endian limb order (limbs[0] = MSB)
        // We iterate i from 3 down to 0 (LSB to MSB of b)
        // For each iteration:
        //   1. t += a * b[i] with carry propagation
        //   2. m = t[3] * mu (compute reduction factor)
        //   3. t = (t + m * q) >> 64 (reduce and shift)
        //
        // Correct carry pattern for (c_out, t[j]) = t[j] + a[j]*b[i] + c_in:
        //   mul lo, aj, bi       // lo = (a[j] * b[i])[63:0]
        //   umulh hi, aj, bi     // hi = (a[j] * b[i])[127:64]
        //   adds tj, tj, lo      // tj += lo, carry1
        //   adcs hi, hi, xzr     // hi += carry1
        //   adds tj, tj, c       // tj += c_in, carry2
        //   adc c, hi, xzr       // c_out = hi + carry2

        unsafe {
            asm!(
                // ===== ITERATION i=3 (b[3] = LSB) =====
                // t += a * b[3], c starts at 0
                // t[3] = t[3] + a[3] * b[3] + 0
                "mul {lo}, {a3}, {b3}",
                "umulh {c}, {a3}, {b3}",
                "adds {t3}, {t3}, {lo}",
                "adc {c}, {c}, xzr",

                // t[2] = t[2] + a[2] * b[3] + c
                "mul {lo}, {a2}, {b3}",
                "umulh {hi}, {a2}, {b3}",
                "adds {t2}, {t2}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t2}, {t2}, {c}",
                "adc {c}, {hi}, xzr",

                // t[1] = t[1] + a[1] * b[3] + c
                "mul {lo}, {a1}, {b3}",
                "umulh {hi}, {a1}, {b3}",
                "adds {t1}, {t1}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t1}, {t1}, {c}",
                "adc {c}, {hi}, xzr",

                // t[0] = t[0] + a[0] * b[3] + c
                "mul {lo}, {a0}, {b3}",
                "umulh {hi}, {a0}, {b3}",
                "adds {t0}, {t0}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t0}, {t0}, {c}",
                "adc {t_extra}, {hi}, xzr",   // t_extra = final carry

                // m = t[3] * mu
                "mul {m}, {t3}, {mu}",

                // Reduction: t = (t + m * q) >> 64
                // c = (t[3] + m * q[3]) >> 64 (discard low 64 bits)
                "mul {lo}, {m}, {q3}",
                "umulh {c}, {m}, {q3}",
                "adds {lo}, {t3}, {lo}",
                "adc {c}, {c}, xzr",

                // t[3] = t[2] + m * q[2] + c
                "mul {lo}, {m}, {q2}",
                "umulh {hi}, {m}, {q2}",
                "adds {t3}, {t2}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t3}, {t3}, {c}",
                "adc {c}, {hi}, xzr",

                // t[2] = t[1] + m * q[1] + c
                "mul {lo}, {m}, {q1}",
                "umulh {hi}, {m}, {q1}",
                "adds {t2}, {t1}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t2}, {t2}, {c}",
                "adc {c}, {hi}, xzr",

                // t[1] = t[0] + m * q[0] + c
                "mul {lo}, {m}, {q0}",
                "umulh {hi}, {m}, {q0}",
                "adds {t1}, {t0}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t1}, {t1}, {c}",
                "adc {c}, {hi}, xzr",

                // t[0] = t_extra + c
                "add {t0}, {t_extra}, {c}",

                // ===== ITERATION i=2 (b[2]) =====
                "mul {lo}, {a3}, {b2}",
                "umulh {c}, {a3}, {b2}",
                "adds {t3}, {t3}, {lo}",
                "adc {c}, {c}, xzr",

                "mul {lo}, {a2}, {b2}",
                "umulh {hi}, {a2}, {b2}",
                "adds {t2}, {t2}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t2}, {t2}, {c}",
                "adc {c}, {hi}, xzr",

                "mul {lo}, {a1}, {b2}",
                "umulh {hi}, {a1}, {b2}",
                "adds {t1}, {t1}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t1}, {t1}, {c}",
                "adc {c}, {hi}, xzr",

                "mul {lo}, {a0}, {b2}",
                "umulh {hi}, {a0}, {b2}",
                "adds {t0}, {t0}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t0}, {t0}, {c}",
                "adc {t_extra}, {hi}, xzr",

                "mul {m}, {t3}, {mu}",

                "mul {lo}, {m}, {q3}",
                "umulh {c}, {m}, {q3}",
                "adds {lo}, {t3}, {lo}",
                "adc {c}, {c}, xzr",

                "mul {lo}, {m}, {q2}",
                "umulh {hi}, {m}, {q2}",
                "adds {t3}, {t2}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t3}, {t3}, {c}",
                "adc {c}, {hi}, xzr",

                "mul {lo}, {m}, {q1}",
                "umulh {hi}, {m}, {q1}",
                "adds {t2}, {t1}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t2}, {t2}, {c}",
                "adc {c}, {hi}, xzr",

                "mul {lo}, {m}, {q0}",
                "umulh {hi}, {m}, {q0}",
                "adds {t1}, {t0}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t1}, {t1}, {c}",
                "adc {c}, {hi}, xzr",

                "add {t0}, {t_extra}, {c}",

                // ===== ITERATION i=1 (b[1]) =====
                "mul {lo}, {a3}, {b1}",
                "umulh {c}, {a3}, {b1}",
                "adds {t3}, {t3}, {lo}",
                "adc {c}, {c}, xzr",

                "mul {lo}, {a2}, {b1}",
                "umulh {hi}, {a2}, {b1}",
                "adds {t2}, {t2}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t2}, {t2}, {c}",
                "adc {c}, {hi}, xzr",

                "mul {lo}, {a1}, {b1}",
                "umulh {hi}, {a1}, {b1}",
                "adds {t1}, {t1}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t1}, {t1}, {c}",
                "adc {c}, {hi}, xzr",

                "mul {lo}, {a0}, {b1}",
                "umulh {hi}, {a0}, {b1}",
                "adds {t0}, {t0}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t0}, {t0}, {c}",
                "adc {t_extra}, {hi}, xzr",

                "mul {m}, {t3}, {mu}",

                "mul {lo}, {m}, {q3}",
                "umulh {c}, {m}, {q3}",
                "adds {lo}, {t3}, {lo}",
                "adc {c}, {c}, xzr",

                "mul {lo}, {m}, {q2}",
                "umulh {hi}, {m}, {q2}",
                "adds {t3}, {t2}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t3}, {t3}, {c}",
                "adc {c}, {hi}, xzr",

                "mul {lo}, {m}, {q1}",
                "umulh {hi}, {m}, {q1}",
                "adds {t2}, {t1}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t2}, {t2}, {c}",
                "adc {c}, {hi}, xzr",

                "mul {lo}, {m}, {q0}",
                "umulh {hi}, {m}, {q0}",
                "adds {t1}, {t0}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t1}, {t1}, {c}",
                "adc {c}, {hi}, xzr",

                "add {t0}, {t_extra}, {c}",

                // ===== ITERATION i=0 (b[0] = MSB) =====
                "mul {lo}, {a3}, {b0}",
                "umulh {c}, {a3}, {b0}",
                "adds {t3}, {t3}, {lo}",
                "adc {c}, {c}, xzr",

                "mul {lo}, {a2}, {b0}",
                "umulh {hi}, {a2}, {b0}",
                "adds {t2}, {t2}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t2}, {t2}, {c}",
                "adc {c}, {hi}, xzr",

                "mul {lo}, {a1}, {b0}",
                "umulh {hi}, {a1}, {b0}",
                "adds {t1}, {t1}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t1}, {t1}, {c}",
                "adc {c}, {hi}, xzr",

                "mul {lo}, {a0}, {b0}",
                "umulh {hi}, {a0}, {b0}",
                "adds {t0}, {t0}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t0}, {t0}, {c}",
                "adc {t_extra}, {hi}, xzr",

                "mul {m}, {t3}, {mu}",

                "mul {lo}, {m}, {q3}",
                "umulh {c}, {m}, {q3}",
                "adds {lo}, {t3}, {lo}",
                "adc {c}, {c}, xzr",

                "mul {lo}, {m}, {q2}",
                "umulh {hi}, {m}, {q2}",
                "adds {t3}, {t2}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t3}, {t3}, {c}",
                "adc {c}, {hi}, xzr",

                "mul {lo}, {m}, {q1}",
                "umulh {hi}, {m}, {q1}",
                "adds {t2}, {t1}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t2}, {t2}, {c}",
                "adc {c}, {hi}, xzr",

                "mul {lo}, {m}, {q0}",
                "umulh {hi}, {m}, {q0}",
                "adds {t1}, {t0}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t1}, {t1}, {c}",
                "adc {c}, {hi}, xzr",

                "add {t0}, {t_extra}, {c}",

                // ===== Final conditional reduction =====
                // if t >= q, then t = t - q
                // Use constant-time CSEL based on comparison
                "subs {lo}, {t3}, {q3}",     // r3 = t3 - q3
                "sbcs {hi}, {t2}, {q2}",     // r2 = t2 - q2 - borrow
                "sbcs {c}, {t1}, {q1}",      // r1 = t1 - q1 - borrow
                "sbcs {m}, {t0}, {q0}",      // r0 = t0 - q0 - borrow
                // If no borrow (carry set), use subtracted values; else keep original
                "csel {t3}, {lo}, {t3}, cs",
                "csel {t2}, {hi}, {t2}, cs",
                "csel {t1}, {c}, {t1}, cs",
                "csel {t0}, {m}, {t0}, cs",

                // Inputs
                a0 = in(reg) a[0],
                a1 = in(reg) a[1],
                a2 = in(reg) a[2],
                a3 = in(reg) a[3],
                b0 = in(reg) b[0],
                b1 = in(reg) b[1],
                b2 = in(reg) b[2],
                b3 = in(reg) b[3],
                q0 = in(reg) q[0],
                q1 = in(reg) q[1],
                q2 = in(reg) q[2],
                q3 = in(reg) q[3],
                mu = in(reg) mu,

                // Outputs (accumulator)
                t0 = inout(reg) t[0],
                t1 = inout(reg) t[1],
                t2 = inout(reg) t[2],
                t3 = inout(reg) t[3],

                // Temporaries
                lo = out(reg) _,
                hi = out(reg) _,
                c = out(reg) _,
                m = out(reg) _,
                t_extra = out(reg) _,

                options(nostack),
            );
        }

        t
    }

    /// ARM64 optimized CIOS for 6 limbs with spare bit optimization (EdMSM Algorithm 2)
    /// Uses pure Rust with u128 (LLVM generates good ARM64 code)
    #[inline(always)]
    pub fn cios_6_limbs_optimized(a: &[u64; 6], b: &[u64; 6], q: &[u64; 6], mu: u64) -> [u64; 6] {
        const N: usize = 6;
        let mut t = [0u64; N];

        for i in (0..N).rev() {
            // t += a * b[i]
            let mut c: u128 = 0;
            for j in (0..N).rev() {
                let cs = t[j] as u128 + (a[j] as u128) * (b[i] as u128) + c;
                c = cs >> 64;
                t[j] = cs as u64;
            }
            let t_extra = c as u64;

            // m := t[N-1] * mu mod 2^64
            let m = t[N - 1].wrapping_mul(mu) as u128;

            // t += m * q, then shift right by 64
            let mut c: u128 = (t[N - 1] as u128 + m * (q[N - 1] as u128)) >> 64;
            for j in (0..N - 1).rev() {
                let cs = t[j] as u128 + m * (q[j] as u128) + c;
                c = cs >> 64;
                t[j + 1] = cs as u64;
            }
            let cs = (t_extra as u128) + c;
            t[0] = cs as u64;
        }

        // Final reduction (no overflow possible with spare bit)
        if const_ge_6(&t, q) {
            sub_6(&mut t, q);
        }
        t
    }

    /// ARM64 inline assembly CIOS for 6 limbs with spare bit optimization
    /// Implements EdMSM Algorithm 2 using true ARM64 inline assembly for 384-bit fields
    ///
    /// CORRECTED VERSION: Uses proper carry propagation pattern matching 4-limb version.
    /// The key pattern for (c_out, t[j]) = t[j] + a[j]*b[i] + c_in is:
    ///   mul lo, aj, bi       // lo = (a[j] * b[i])[63:0]
    ///   umulh hi, aj, bi     // hi = (a[j] * b[i])[127:64]
    ///   adds tj, tj, lo      // tj += lo, carry1
    ///   adcs hi, hi, xzr     // hi += carry1
    ///   adds tj, tj, c       // tj += c_in, carry2
    ///   adc c, hi, xzr       // c_out = hi + carry2
    #[inline(always)]
    pub fn cios_6_limbs_asm_optimized(
        a: &[u64; 6],
        b: &[u64; 6],
        q: &[u64; 6],
        mu: u64,
    ) -> [u64; 6] {
        let mut t = [0u64; 6];

        unsafe {
            asm!(
                // ===== ITERATION i=5 (b[5] = LSB) =====
                "ldr {bi}, [{b_ptr}, #40]",

                // t[5] += a[5] * b[5] (first multiply, no incoming carry)
                "ldr {ax}, [{a_ptr}, #40]",
                "mul {lo}, {ax}, {bi}",
                "umulh {c}, {ax}, {bi}",
                "adds {t5}, {t5}, {lo}",
                "adc {c}, {c}, xzr",

                // t[4] += a[4] * b[5] + c
                "ldr {ax}, [{a_ptr}, #32]",
                "mul {lo}, {ax}, {bi}",
                "umulh {hi}, {ax}, {bi}",
                "adds {t4}, {t4}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t4}, {t4}, {c}",
                "adc {c}, {hi}, xzr",

                // t[3] += a[3] * b[5] + c
                "ldr {ax}, [{a_ptr}, #24]",
                "mul {lo}, {ax}, {bi}",
                "umulh {hi}, {ax}, {bi}",
                "adds {t3}, {t3}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t3}, {t3}, {c}",
                "adc {c}, {hi}, xzr",

                // t[2] += a[2] * b[5] + c
                "ldr {ax}, [{a_ptr}, #16]",
                "mul {lo}, {ax}, {bi}",
                "umulh {hi}, {ax}, {bi}",
                "adds {t2}, {t2}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t2}, {t2}, {c}",
                "adc {c}, {hi}, xzr",

                // t[1] += a[1] * b[5] + c
                "ldr {ax}, [{a_ptr}, #8]",
                "mul {lo}, {ax}, {bi}",
                "umulh {hi}, {ax}, {bi}",
                "adds {t1}, {t1}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t1}, {t1}, {c}",
                "adc {c}, {hi}, xzr",

                // t[0] += a[0] * b[5] + c, t_extra = final overflow
                "ldr {ax}, [{a_ptr}]",
                "mul {lo}, {ax}, {bi}",
                "umulh {hi}, {ax}, {bi}",
                "adds {t0}, {t0}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t0}, {t0}, {c}",
                "adc {t_extra}, {hi}, xzr",

                // m = t[5] * mu
                "mul {m}, {t5}, {mu}",

                // Reduction: t = (t + m * q) >> 64
                // First: discard (t[5] + m * q[5]) mod 2^64, keep carry
                "ldr {qx}, [{q_ptr}, #40]",
                "mul {lo}, {m}, {qx}",
                "umulh {c}, {m}, {qx}",
                "adds {lo}, {t5}, {lo}",
                "adc {c}, {c}, xzr",

                // t[5] = t[4] + m * q[4] + c
                "ldr {qx}, [{q_ptr}, #32]",
                "mul {lo}, {m}, {qx}",
                "umulh {hi}, {m}, {qx}",
                "adds {t5}, {t4}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t5}, {t5}, {c}",
                "adc {c}, {hi}, xzr",

                // t[4] = t[3] + m * q[3] + c
                "ldr {qx}, [{q_ptr}, #24]",
                "mul {lo}, {m}, {qx}",
                "umulh {hi}, {m}, {qx}",
                "adds {t4}, {t3}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t4}, {t4}, {c}",
                "adc {c}, {hi}, xzr",

                // t[3] = t[2] + m * q[2] + c
                "ldr {qx}, [{q_ptr}, #16]",
                "mul {lo}, {m}, {qx}",
                "umulh {hi}, {m}, {qx}",
                "adds {t3}, {t2}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t3}, {t3}, {c}",
                "adc {c}, {hi}, xzr",

                // t[2] = t[1] + m * q[1] + c
                "ldr {qx}, [{q_ptr}, #8]",
                "mul {lo}, {m}, {qx}",
                "umulh {hi}, {m}, {qx}",
                "adds {t2}, {t1}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t2}, {t2}, {c}",
                "adc {c}, {hi}, xzr",

                // t[1] = t[0] + m * q[0] + c
                "ldr {qx}, [{q_ptr}]",
                "mul {lo}, {m}, {qx}",
                "umulh {hi}, {m}, {qx}",
                "adds {t1}, {t0}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t1}, {t1}, {c}",
                "adc {c}, {hi}, xzr",

                // t[0] = t_extra + c
                "add {t0}, {t_extra}, {c}",

                // ===== ITERATION i=4 =====
                "ldr {bi}, [{b_ptr}, #32]",

                "ldr {ax}, [{a_ptr}, #40]",
                "mul {lo}, {ax}, {bi}",
                "umulh {c}, {ax}, {bi}",
                "adds {t5}, {t5}, {lo}",
                "adc {c}, {c}, xzr",

                "ldr {ax}, [{a_ptr}, #32]",
                "mul {lo}, {ax}, {bi}",
                "umulh {hi}, {ax}, {bi}",
                "adds {t4}, {t4}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t4}, {t4}, {c}",
                "adc {c}, {hi}, xzr",

                "ldr {ax}, [{a_ptr}, #24]",
                "mul {lo}, {ax}, {bi}",
                "umulh {hi}, {ax}, {bi}",
                "adds {t3}, {t3}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t3}, {t3}, {c}",
                "adc {c}, {hi}, xzr",

                "ldr {ax}, [{a_ptr}, #16]",
                "mul {lo}, {ax}, {bi}",
                "umulh {hi}, {ax}, {bi}",
                "adds {t2}, {t2}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t2}, {t2}, {c}",
                "adc {c}, {hi}, xzr",

                "ldr {ax}, [{a_ptr}, #8]",
                "mul {lo}, {ax}, {bi}",
                "umulh {hi}, {ax}, {bi}",
                "adds {t1}, {t1}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t1}, {t1}, {c}",
                "adc {c}, {hi}, xzr",

                "ldr {ax}, [{a_ptr}]",
                "mul {lo}, {ax}, {bi}",
                "umulh {hi}, {ax}, {bi}",
                "adds {t0}, {t0}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t0}, {t0}, {c}",
                "adc {t_extra}, {hi}, xzr",

                "mul {m}, {t5}, {mu}",

                "ldr {qx}, [{q_ptr}, #40]",
                "mul {lo}, {m}, {qx}",
                "umulh {c}, {m}, {qx}",
                "adds {lo}, {t5}, {lo}",
                "adc {c}, {c}, xzr",

                "ldr {qx}, [{q_ptr}, #32]",
                "mul {lo}, {m}, {qx}",
                "umulh {hi}, {m}, {qx}",
                "adds {t5}, {t4}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t5}, {t5}, {c}",
                "adc {c}, {hi}, xzr",

                "ldr {qx}, [{q_ptr}, #24]",
                "mul {lo}, {m}, {qx}",
                "umulh {hi}, {m}, {qx}",
                "adds {t4}, {t3}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t4}, {t4}, {c}",
                "adc {c}, {hi}, xzr",

                "ldr {qx}, [{q_ptr}, #16]",
                "mul {lo}, {m}, {qx}",
                "umulh {hi}, {m}, {qx}",
                "adds {t3}, {t2}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t3}, {t3}, {c}",
                "adc {c}, {hi}, xzr",

                "ldr {qx}, [{q_ptr}, #8]",
                "mul {lo}, {m}, {qx}",
                "umulh {hi}, {m}, {qx}",
                "adds {t2}, {t1}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t2}, {t2}, {c}",
                "adc {c}, {hi}, xzr",

                "ldr {qx}, [{q_ptr}]",
                "mul {lo}, {m}, {qx}",
                "umulh {hi}, {m}, {qx}",
                "adds {t1}, {t0}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t1}, {t1}, {c}",
                "adc {c}, {hi}, xzr",

                "add {t0}, {t_extra}, {c}",

                // ===== ITERATION i=3 =====
                "ldr {bi}, [{b_ptr}, #24]",

                "ldr {ax}, [{a_ptr}, #40]",
                "mul {lo}, {ax}, {bi}",
                "umulh {c}, {ax}, {bi}",
                "adds {t5}, {t5}, {lo}",
                "adc {c}, {c}, xzr",

                "ldr {ax}, [{a_ptr}, #32]",
                "mul {lo}, {ax}, {bi}",
                "umulh {hi}, {ax}, {bi}",
                "adds {t4}, {t4}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t4}, {t4}, {c}",
                "adc {c}, {hi}, xzr",

                "ldr {ax}, [{a_ptr}, #24]",
                "mul {lo}, {ax}, {bi}",
                "umulh {hi}, {ax}, {bi}",
                "adds {t3}, {t3}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t3}, {t3}, {c}",
                "adc {c}, {hi}, xzr",

                "ldr {ax}, [{a_ptr}, #16]",
                "mul {lo}, {ax}, {bi}",
                "umulh {hi}, {ax}, {bi}",
                "adds {t2}, {t2}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t2}, {t2}, {c}",
                "adc {c}, {hi}, xzr",

                "ldr {ax}, [{a_ptr}, #8]",
                "mul {lo}, {ax}, {bi}",
                "umulh {hi}, {ax}, {bi}",
                "adds {t1}, {t1}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t1}, {t1}, {c}",
                "adc {c}, {hi}, xzr",

                "ldr {ax}, [{a_ptr}]",
                "mul {lo}, {ax}, {bi}",
                "umulh {hi}, {ax}, {bi}",
                "adds {t0}, {t0}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t0}, {t0}, {c}",
                "adc {t_extra}, {hi}, xzr",

                "mul {m}, {t5}, {mu}",

                "ldr {qx}, [{q_ptr}, #40]",
                "mul {lo}, {m}, {qx}",
                "umulh {c}, {m}, {qx}",
                "adds {lo}, {t5}, {lo}",
                "adc {c}, {c}, xzr",

                "ldr {qx}, [{q_ptr}, #32]",
                "mul {lo}, {m}, {qx}",
                "umulh {hi}, {m}, {qx}",
                "adds {t5}, {t4}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t5}, {t5}, {c}",
                "adc {c}, {hi}, xzr",

                "ldr {qx}, [{q_ptr}, #24]",
                "mul {lo}, {m}, {qx}",
                "umulh {hi}, {m}, {qx}",
                "adds {t4}, {t3}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t4}, {t4}, {c}",
                "adc {c}, {hi}, xzr",

                "ldr {qx}, [{q_ptr}, #16]",
                "mul {lo}, {m}, {qx}",
                "umulh {hi}, {m}, {qx}",
                "adds {t3}, {t2}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t3}, {t3}, {c}",
                "adc {c}, {hi}, xzr",

                "ldr {qx}, [{q_ptr}, #8]",
                "mul {lo}, {m}, {qx}",
                "umulh {hi}, {m}, {qx}",
                "adds {t2}, {t1}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t2}, {t2}, {c}",
                "adc {c}, {hi}, xzr",

                "ldr {qx}, [{q_ptr}]",
                "mul {lo}, {m}, {qx}",
                "umulh {hi}, {m}, {qx}",
                "adds {t1}, {t0}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t1}, {t1}, {c}",
                "adc {c}, {hi}, xzr",

                "add {t0}, {t_extra}, {c}",

                // ===== ITERATION i=2 =====
                "ldr {bi}, [{b_ptr}, #16]",

                "ldr {ax}, [{a_ptr}, #40]",
                "mul {lo}, {ax}, {bi}",
                "umulh {c}, {ax}, {bi}",
                "adds {t5}, {t5}, {lo}",
                "adc {c}, {c}, xzr",

                "ldr {ax}, [{a_ptr}, #32]",
                "mul {lo}, {ax}, {bi}",
                "umulh {hi}, {ax}, {bi}",
                "adds {t4}, {t4}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t4}, {t4}, {c}",
                "adc {c}, {hi}, xzr",

                "ldr {ax}, [{a_ptr}, #24]",
                "mul {lo}, {ax}, {bi}",
                "umulh {hi}, {ax}, {bi}",
                "adds {t3}, {t3}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t3}, {t3}, {c}",
                "adc {c}, {hi}, xzr",

                "ldr {ax}, [{a_ptr}, #16]",
                "mul {lo}, {ax}, {bi}",
                "umulh {hi}, {ax}, {bi}",
                "adds {t2}, {t2}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t2}, {t2}, {c}",
                "adc {c}, {hi}, xzr",

                "ldr {ax}, [{a_ptr}, #8]",
                "mul {lo}, {ax}, {bi}",
                "umulh {hi}, {ax}, {bi}",
                "adds {t1}, {t1}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t1}, {t1}, {c}",
                "adc {c}, {hi}, xzr",

                "ldr {ax}, [{a_ptr}]",
                "mul {lo}, {ax}, {bi}",
                "umulh {hi}, {ax}, {bi}",
                "adds {t0}, {t0}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t0}, {t0}, {c}",
                "adc {t_extra}, {hi}, xzr",

                "mul {m}, {t5}, {mu}",

                "ldr {qx}, [{q_ptr}, #40]",
                "mul {lo}, {m}, {qx}",
                "umulh {c}, {m}, {qx}",
                "adds {lo}, {t5}, {lo}",
                "adc {c}, {c}, xzr",

                "ldr {qx}, [{q_ptr}, #32]",
                "mul {lo}, {m}, {qx}",
                "umulh {hi}, {m}, {qx}",
                "adds {t5}, {t4}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t5}, {t5}, {c}",
                "adc {c}, {hi}, xzr",

                "ldr {qx}, [{q_ptr}, #24]",
                "mul {lo}, {m}, {qx}",
                "umulh {hi}, {m}, {qx}",
                "adds {t4}, {t3}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t4}, {t4}, {c}",
                "adc {c}, {hi}, xzr",

                "ldr {qx}, [{q_ptr}, #16]",
                "mul {lo}, {m}, {qx}",
                "umulh {hi}, {m}, {qx}",
                "adds {t3}, {t2}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t3}, {t3}, {c}",
                "adc {c}, {hi}, xzr",

                "ldr {qx}, [{q_ptr}, #8]",
                "mul {lo}, {m}, {qx}",
                "umulh {hi}, {m}, {qx}",
                "adds {t2}, {t1}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t2}, {t2}, {c}",
                "adc {c}, {hi}, xzr",

                "ldr {qx}, [{q_ptr}]",
                "mul {lo}, {m}, {qx}",
                "umulh {hi}, {m}, {qx}",
                "adds {t1}, {t0}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t1}, {t1}, {c}",
                "adc {c}, {hi}, xzr",

                "add {t0}, {t_extra}, {c}",

                // ===== ITERATION i=1 =====
                "ldr {bi}, [{b_ptr}, #8]",

                "ldr {ax}, [{a_ptr}, #40]",
                "mul {lo}, {ax}, {bi}",
                "umulh {c}, {ax}, {bi}",
                "adds {t5}, {t5}, {lo}",
                "adc {c}, {c}, xzr",

                "ldr {ax}, [{a_ptr}, #32]",
                "mul {lo}, {ax}, {bi}",
                "umulh {hi}, {ax}, {bi}",
                "adds {t4}, {t4}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t4}, {t4}, {c}",
                "adc {c}, {hi}, xzr",

                "ldr {ax}, [{a_ptr}, #24]",
                "mul {lo}, {ax}, {bi}",
                "umulh {hi}, {ax}, {bi}",
                "adds {t3}, {t3}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t3}, {t3}, {c}",
                "adc {c}, {hi}, xzr",

                "ldr {ax}, [{a_ptr}, #16]",
                "mul {lo}, {ax}, {bi}",
                "umulh {hi}, {ax}, {bi}",
                "adds {t2}, {t2}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t2}, {t2}, {c}",
                "adc {c}, {hi}, xzr",

                "ldr {ax}, [{a_ptr}, #8]",
                "mul {lo}, {ax}, {bi}",
                "umulh {hi}, {ax}, {bi}",
                "adds {t1}, {t1}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t1}, {t1}, {c}",
                "adc {c}, {hi}, xzr",

                "ldr {ax}, [{a_ptr}]",
                "mul {lo}, {ax}, {bi}",
                "umulh {hi}, {ax}, {bi}",
                "adds {t0}, {t0}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t0}, {t0}, {c}",
                "adc {t_extra}, {hi}, xzr",

                "mul {m}, {t5}, {mu}",

                "ldr {qx}, [{q_ptr}, #40]",
                "mul {lo}, {m}, {qx}",
                "umulh {c}, {m}, {qx}",
                "adds {lo}, {t5}, {lo}",
                "adc {c}, {c}, xzr",

                "ldr {qx}, [{q_ptr}, #32]",
                "mul {lo}, {m}, {qx}",
                "umulh {hi}, {m}, {qx}",
                "adds {t5}, {t4}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t5}, {t5}, {c}",
                "adc {c}, {hi}, xzr",

                "ldr {qx}, [{q_ptr}, #24]",
                "mul {lo}, {m}, {qx}",
                "umulh {hi}, {m}, {qx}",
                "adds {t4}, {t3}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t4}, {t4}, {c}",
                "adc {c}, {hi}, xzr",

                "ldr {qx}, [{q_ptr}, #16]",
                "mul {lo}, {m}, {qx}",
                "umulh {hi}, {m}, {qx}",
                "adds {t3}, {t2}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t3}, {t3}, {c}",
                "adc {c}, {hi}, xzr",

                "ldr {qx}, [{q_ptr}, #8]",
                "mul {lo}, {m}, {qx}",
                "umulh {hi}, {m}, {qx}",
                "adds {t2}, {t1}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t2}, {t2}, {c}",
                "adc {c}, {hi}, xzr",

                "ldr {qx}, [{q_ptr}]",
                "mul {lo}, {m}, {qx}",
                "umulh {hi}, {m}, {qx}",
                "adds {t1}, {t0}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t1}, {t1}, {c}",
                "adc {c}, {hi}, xzr",

                "add {t0}, {t_extra}, {c}",

                // ===== ITERATION i=0 (MSB) =====
                "ldr {bi}, [{b_ptr}]",

                "ldr {ax}, [{a_ptr}, #40]",
                "mul {lo}, {ax}, {bi}",
                "umulh {c}, {ax}, {bi}",
                "adds {t5}, {t5}, {lo}",
                "adc {c}, {c}, xzr",

                "ldr {ax}, [{a_ptr}, #32]",
                "mul {lo}, {ax}, {bi}",
                "umulh {hi}, {ax}, {bi}",
                "adds {t4}, {t4}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t4}, {t4}, {c}",
                "adc {c}, {hi}, xzr",

                "ldr {ax}, [{a_ptr}, #24]",
                "mul {lo}, {ax}, {bi}",
                "umulh {hi}, {ax}, {bi}",
                "adds {t3}, {t3}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t3}, {t3}, {c}",
                "adc {c}, {hi}, xzr",

                "ldr {ax}, [{a_ptr}, #16]",
                "mul {lo}, {ax}, {bi}",
                "umulh {hi}, {ax}, {bi}",
                "adds {t2}, {t2}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t2}, {t2}, {c}",
                "adc {c}, {hi}, xzr",

                "ldr {ax}, [{a_ptr}, #8]",
                "mul {lo}, {ax}, {bi}",
                "umulh {hi}, {ax}, {bi}",
                "adds {t1}, {t1}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t1}, {t1}, {c}",
                "adc {c}, {hi}, xzr",

                "ldr {ax}, [{a_ptr}]",
                "mul {lo}, {ax}, {bi}",
                "umulh {hi}, {ax}, {bi}",
                "adds {t0}, {t0}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t0}, {t0}, {c}",
                "adc {t_extra}, {hi}, xzr",

                "mul {m}, {t5}, {mu}",

                "ldr {qx}, [{q_ptr}, #40]",
                "mul {lo}, {m}, {qx}",
                "umulh {c}, {m}, {qx}",
                "adds {lo}, {t5}, {lo}",
                "adc {c}, {c}, xzr",

                "ldr {qx}, [{q_ptr}, #32]",
                "mul {lo}, {m}, {qx}",
                "umulh {hi}, {m}, {qx}",
                "adds {t5}, {t4}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t5}, {t5}, {c}",
                "adc {c}, {hi}, xzr",

                "ldr {qx}, [{q_ptr}, #24]",
                "mul {lo}, {m}, {qx}",
                "umulh {hi}, {m}, {qx}",
                "adds {t4}, {t3}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t4}, {t4}, {c}",
                "adc {c}, {hi}, xzr",

                "ldr {qx}, [{q_ptr}, #16]",
                "mul {lo}, {m}, {qx}",
                "umulh {hi}, {m}, {qx}",
                "adds {t3}, {t2}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t3}, {t3}, {c}",
                "adc {c}, {hi}, xzr",

                "ldr {qx}, [{q_ptr}, #8]",
                "mul {lo}, {m}, {qx}",
                "umulh {hi}, {m}, {qx}",
                "adds {t2}, {t1}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t2}, {t2}, {c}",
                "adc {c}, {hi}, xzr",

                "ldr {qx}, [{q_ptr}]",
                "mul {lo}, {m}, {qx}",
                "umulh {hi}, {m}, {qx}",
                "adds {t1}, {t0}, {lo}",
                "adcs {hi}, {hi}, xzr",
                "adds {t1}, {t1}, {c}",
                "adc {c}, {hi}, xzr",

                "add {t0}, {t_extra}, {c}",

                // ===== Final conditional reduction =====
                // if t >= q, then t = t - q
                "ldr {qx}, [{q_ptr}, #40]",
                "subs {lo}, {t5}, {qx}",
                "ldr {qx}, [{q_ptr}, #32]",
                "sbcs {hi}, {t4}, {qx}",
                "ldr {qx}, [{q_ptr}, #24]",
                "sbcs {c}, {t3}, {qx}",
                "ldr {qx}, [{q_ptr}, #16]",
                "sbcs {m}, {t2}, {qx}",
                "ldr {qx}, [{q_ptr}, #8]",
                "sbcs {ax}, {t1}, {qx}",
                "ldr {qx}, [{q_ptr}]",
                "sbcs {bi}, {t0}, {qx}",

                // If no borrow (carry set), use subtracted values
                "csel {t5}, {lo}, {t5}, cs",
                "csel {t4}, {hi}, {t4}, cs",
                "csel {t3}, {c}, {t3}, cs",
                "csel {t2}, {m}, {t2}, cs",
                "csel {t1}, {ax}, {t1}, cs",
                "csel {t0}, {bi}, {t0}, cs",

                a_ptr = in(reg) a.as_ptr(),
                b_ptr = in(reg) b.as_ptr(),
                q_ptr = in(reg) q.as_ptr(),
                mu = in(reg) mu,

                t0 = inout(reg) t[0],
                t1 = inout(reg) t[1],
                t2 = inout(reg) t[2],
                t3 = inout(reg) t[3],
                t4 = inout(reg) t[4],
                t5 = inout(reg) t[5],

                lo = out(reg) _,
                hi = out(reg) _,
                c = out(reg) _,
                m = out(reg) _,
                t_extra = out(reg) _,
                ax = out(reg) _,
                bi = out(reg) _,
                qx = out(reg) _,

                options(nostack),
            );
        }

        t
    }
}

#[cfg(all(target_arch = "aarch64", feature = "asm"))]
impl MontgomeryAlgorithms {
    /// ARM64 assembly optimized CIOS multiplication
    /// Automatically dispatches to the appropriate implementation based on NUM_LIMBS
    #[inline(always)]
    pub fn cios_asm<const NUM_LIMBS: usize>(
        a: &UnsignedInteger<NUM_LIMBS>,
        b: &UnsignedInteger<NUM_LIMBS>,
        q: &UnsignedInteger<NUM_LIMBS>,
        mu: &u64,
    ) -> UnsignedInteger<NUM_LIMBS> {
        // Dispatch to specialized assembly implementations
        match NUM_LIMBS {
            4 => {
                let a_arr: &[u64; 4] = a.limbs.as_slice().try_into().unwrap();
                let b_arr: &[u64; 4] = b.limbs.as_slice().try_into().unwrap();
                let q_arr: &[u64; 4] = q.limbs.as_slice().try_into().unwrap();
                let result = aarch64_asm::cios_4_limbs(a_arr, b_arr, q_arr, *mu);
                let mut limbs = [0u64; NUM_LIMBS];
                limbs.copy_from_slice(&result);
                UnsignedInteger { limbs }
            }
            6 => {
                let a_arr: &[u64; 6] = a.limbs.as_slice().try_into().unwrap();
                let b_arr: &[u64; 6] = b.limbs.as_slice().try_into().unwrap();
                let q_arr: &[u64; 6] = q.limbs.as_slice().try_into().unwrap();
                let result = aarch64_asm::cios_6_limbs(a_arr, b_arr, q_arr, *mu);
                let mut limbs = [0u64; NUM_LIMBS];
                limbs.copy_from_slice(&result);
                UnsignedInteger { limbs }
            }
            // Fallback to pure Rust for other sizes
            _ => Self::cios(a, b, q, mu),
        }
    }

    /// ARM64 optimized CIOS for moduli with one spare bit (EdMSM Algorithm 2)
    /// This is more efficient because it avoids overflow tracking
    /// Uses Rust u128 implementation (LLVM generates efficient ARM64 code)
    #[inline(always)]
    pub fn cios_asm_optimized<const NUM_LIMBS: usize>(
        a: &UnsignedInteger<NUM_LIMBS>,
        b: &UnsignedInteger<NUM_LIMBS>,
        q: &UnsignedInteger<NUM_LIMBS>,
        mu: &u64,
    ) -> UnsignedInteger<NUM_LIMBS> {
        match NUM_LIMBS {
            4 => {
                let a_arr: &[u64; 4] = a.limbs.as_slice().try_into().unwrap();
                let b_arr: &[u64; 4] = b.limbs.as_slice().try_into().unwrap();
                let q_arr: &[u64; 4] = q.limbs.as_slice().try_into().unwrap();
                let result = aarch64_asm::cios_4_limbs_optimized(a_arr, b_arr, q_arr, *mu);
                let mut limbs = [0u64; NUM_LIMBS];
                limbs.copy_from_slice(&result);
                UnsignedInteger { limbs }
            }
            6 => {
                let a_arr: &[u64; 6] = a.limbs.as_slice().try_into().unwrap();
                let b_arr: &[u64; 6] = b.limbs.as_slice().try_into().unwrap();
                let q_arr: &[u64; 6] = q.limbs.as_slice().try_into().unwrap();
                let result = aarch64_asm::cios_6_limbs_optimized(a_arr, b_arr, q_arr, *mu);
                let mut limbs = [0u64; NUM_LIMBS];
                limbs.copy_from_slice(&result);
                UnsignedInteger { limbs }
            }
            // Fallback to pure Rust optimized version for other sizes
            _ => Self::cios_optimized_for_moduli_with_one_spare_bit(a, b, q, mu),
        }
    }

    /// ARM64 TRUE INLINE ASSEMBLY CIOS for moduli with one spare bit
    /// Uses hand-written ARM64 assembly with MUL/UMULH and ADDS/ADCS chains
    /// This provides maximum performance by avoiding LLVM's register allocation
    #[inline(always)]
    pub fn cios_true_asm_optimized<const NUM_LIMBS: usize>(
        a: &UnsignedInteger<NUM_LIMBS>,
        b: &UnsignedInteger<NUM_LIMBS>,
        q: &UnsignedInteger<NUM_LIMBS>,
        mu: &u64,
    ) -> UnsignedInteger<NUM_LIMBS> {
        match NUM_LIMBS {
            4 => {
                let a_arr: &[u64; 4] = a.limbs.as_slice().try_into().unwrap();
                let b_arr: &[u64; 4] = b.limbs.as_slice().try_into().unwrap();
                let q_arr: &[u64; 4] = q.limbs.as_slice().try_into().unwrap();
                let result = aarch64_asm::cios_4_limbs_asm_optimized(a_arr, b_arr, q_arr, *mu);
                let mut limbs = [0u64; NUM_LIMBS];
                limbs.copy_from_slice(&result);
                UnsignedInteger { limbs }
            }
            6 => {
                // Use corrected true inline assembly for 6 limbs
                let a_arr: &[u64; 6] = a.limbs.as_slice().try_into().unwrap();
                let b_arr: &[u64; 6] = b.limbs.as_slice().try_into().unwrap();
                let q_arr: &[u64; 6] = q.limbs.as_slice().try_into().unwrap();
                let result = aarch64_asm::cios_6_limbs_asm_optimized(a_arr, b_arr, q_arr, *mu);
                let mut limbs = [0u64; NUM_LIMBS];
                limbs.copy_from_slice(&result);
                UnsignedInteger { limbs }
            }
            // Fallback to pure Rust optimized version for other sizes
            _ => Self::cios_optimized_for_moduli_with_one_spare_bit(a, b, q, mu),
        }
    }

    /// ARM64 assembly optimized modular addition: (a + b) mod q
    #[inline(always)]
    pub fn add_asm<const NUM_LIMBS: usize>(
        a: &UnsignedInteger<NUM_LIMBS>,
        b: &UnsignedInteger<NUM_LIMBS>,
        q: &UnsignedInteger<NUM_LIMBS>,
    ) -> UnsignedInteger<NUM_LIMBS> {
        match NUM_LIMBS {
            4 => {
                let a_arr: &[u64; 4] = a.limbs.as_slice().try_into().unwrap();
                let b_arr: &[u64; 4] = b.limbs.as_slice().try_into().unwrap();
                let q_arr: &[u64; 4] = q.limbs.as_slice().try_into().unwrap();
                let result = aarch64_asm::mod_add_4_limbs(a_arr, b_arr, q_arr);
                let mut limbs = [0u64; NUM_LIMBS];
                limbs.copy_from_slice(&result);
                UnsignedInteger { limbs }
            }
            6 => {
                let a_arr: &[u64; 6] = a.limbs.as_slice().try_into().unwrap();
                let b_arr: &[u64; 6] = b.limbs.as_slice().try_into().unwrap();
                let q_arr: &[u64; 6] = q.limbs.as_slice().try_into().unwrap();
                let result = aarch64_asm::mod_add_6_limbs(a_arr, b_arr, q_arr);
                let mut limbs = [0u64; NUM_LIMBS];
                limbs.copy_from_slice(&result);
                UnsignedInteger { limbs }
            }
            // Fallback to pure Rust
            _ => {
                let (sum, overflow) = UnsignedInteger::add(a, b);
                if overflow || UnsignedInteger::const_le(q, &sum) {
                    UnsignedInteger::sub(&sum, q).0
                } else {
                    sum
                }
            }
        }
    }

    /// ARM64 assembly optimized modular subtraction: (a - b) mod q
    #[inline(always)]
    pub fn sub_asm<const NUM_LIMBS: usize>(
        a: &UnsignedInteger<NUM_LIMBS>,
        b: &UnsignedInteger<NUM_LIMBS>,
        q: &UnsignedInteger<NUM_LIMBS>,
    ) -> UnsignedInteger<NUM_LIMBS> {
        match NUM_LIMBS {
            4 => {
                let a_arr: &[u64; 4] = a.limbs.as_slice().try_into().unwrap();
                let b_arr: &[u64; 4] = b.limbs.as_slice().try_into().unwrap();
                let q_arr: &[u64; 4] = q.limbs.as_slice().try_into().unwrap();
                let result = aarch64_asm::mod_sub_4_limbs(a_arr, b_arr, q_arr);
                let mut limbs = [0u64; NUM_LIMBS];
                limbs.copy_from_slice(&result);
                UnsignedInteger { limbs }
            }
            6 => {
                let a_arr: &[u64; 6] = a.limbs.as_slice().try_into().unwrap();
                let b_arr: &[u64; 6] = b.limbs.as_slice().try_into().unwrap();
                let q_arr: &[u64; 6] = q.limbs.as_slice().try_into().unwrap();
                let result = aarch64_asm::mod_sub_6_limbs(a_arr, b_arr, q_arr);
                let mut limbs = [0u64; NUM_LIMBS];
                limbs.copy_from_slice(&result);
                UnsignedInteger { limbs }
            }
            // Fallback to pure Rust
            _ => {
                if b <= a {
                    *a - *b
                } else {
                    *q - (*b - *a)
                }
            }
        }
    }
}

// x86-64 assembly implementations for Montgomery multiplication
#[cfg(all(target_arch = "x86_64", feature = "asm"))]
mod x86_64_asm {
    use core::arch::asm;

    /// x86-64 inline assembly for 4-limb addition: r = a + b
    /// Returns (result, overflow)
    /// Uses ADD/ADC chain for efficient carry propagation
    /// Note: lambdaworks uses big-endian limb order (limbs[0] = MSB)
    #[inline(always)]
    pub fn add_4_limbs_asm(a: &[u64; 4], b: &[u64; 4]) -> ([u64; 4], bool) {
        let mut r = [0u64; 4];
        let overflow: u8;

        unsafe {
            asm!(
                // Add with carry chain starting from LSB (index 3)
                "add {r3}, {b3}",        // r[3] = a[3] + b[3], set carry
                "adc {r2}, {b2}",        // r[2] = a[2] + b[2] + carry
                "adc {r1}, {b1}",        // r[1] = a[1] + b[1] + carry
                "adc {r0}, {b0}",        // r[0] = a[0] + b[0] + carry
                "setc {ovf}",            // capture final carry

                r0 = inout(reg) a[0] => r[0],
                r1 = inout(reg) a[1] => r[1],
                r2 = inout(reg) a[2] => r[2],
                r3 = inout(reg) a[3] => r[3],
                b0 = in(reg) b[0],
                b1 = in(reg) b[1],
                b2 = in(reg) b[2],
                b3 = in(reg) b[3],
                ovf = out(reg_byte) overflow,
                options(pure, nomem, nostack),
            );
        }

        (r, overflow != 0)
    }

    /// x86-64 inline assembly for 4-limb subtraction: r = a - b
    /// Returns (result, borrow)
    /// Uses SUB/SBB chain for efficient borrow propagation
    #[inline(always)]
    pub fn sub_4_limbs_asm(a: &[u64; 4], b: &[u64; 4]) -> ([u64; 4], bool) {
        let mut r = [0u64; 4];
        let borrow: u8;

        unsafe {
            asm!(
                // Subtract with borrow chain starting from LSB (index 3)
                "sub {r3}, {b3}",        // r[3] = a[3] - b[3], set borrow
                "sbb {r2}, {b2}",        // r[2] = a[2] - b[2] - borrow
                "sbb {r1}, {b1}",        // r[1] = a[1] - b[1] - borrow
                "sbb {r0}, {b0}",        // r[0] = a[0] - b[0] - borrow
                "setc {brw}",            // capture final borrow (CF=1 means borrow)

                r0 = inout(reg) a[0] => r[0],
                r1 = inout(reg) a[1] => r[1],
                r2 = inout(reg) a[2] => r[2],
                r3 = inout(reg) a[3] => r[3],
                b0 = in(reg) b[0],
                b1 = in(reg) b[1],
                b2 = in(reg) b[2],
                b3 = in(reg) b[3],
                brw = out(reg_byte) borrow,
                options(pure, nomem, nostack),
            );
        }

        (r, borrow != 0)
    }

    /// x86-64 inline assembly for 6-limb addition: r = a + b
    /// Returns (result, overflow)
    #[inline(always)]
    pub fn add_6_limbs_asm(a: &[u64; 6], b: &[u64; 6]) -> ([u64; 6], bool) {
        let mut r = [0u64; 6];
        let overflow: u8;

        unsafe {
            asm!(
                // Add with carry chain starting from LSB (index 5)
                "add {r5}, {b5}",
                "adc {r4}, {b4}",
                "adc {r3}, {b3}",
                "adc {r2}, {b2}",
                "adc {r1}, {b1}",
                "adc {r0}, {b0}",
                "setc {ovf}",

                r0 = inout(reg) a[0] => r[0],
                r1 = inout(reg) a[1] => r[1],
                r2 = inout(reg) a[2] => r[2],
                r3 = inout(reg) a[3] => r[3],
                r4 = inout(reg) a[4] => r[4],
                r5 = inout(reg) a[5] => r[5],
                b0 = in(reg) b[0],
                b1 = in(reg) b[1],
                b2 = in(reg) b[2],
                b3 = in(reg) b[3],
                b4 = in(reg) b[4],
                b5 = in(reg) b[5],
                ovf = out(reg_byte) overflow,
                options(pure, nomem, nostack),
            );
        }

        (r, overflow != 0)
    }

    /// x86-64 inline assembly for 6-limb subtraction: r = a - b
    /// Returns (result, borrow)
    #[inline(always)]
    pub fn sub_6_limbs_asm(a: &[u64; 6], b: &[u64; 6]) -> ([u64; 6], bool) {
        let mut r = [0u64; 6];
        let borrow: u8;

        unsafe {
            asm!(
                "sub {r5}, {b5}",
                "sbb {r4}, {b4}",
                "sbb {r3}, {b3}",
                "sbb {r2}, {b2}",
                "sbb {r1}, {b1}",
                "sbb {r0}, {b0}",
                "setc {brw}",

                r0 = inout(reg) a[0] => r[0],
                r1 = inout(reg) a[1] => r[1],
                r2 = inout(reg) a[2] => r[2],
                r3 = inout(reg) a[3] => r[3],
                r4 = inout(reg) a[4] => r[4],
                r5 = inout(reg) a[5] => r[5],
                b0 = in(reg) b[0],
                b1 = in(reg) b[1],
                b2 = in(reg) b[2],
                b3 = in(reg) b[3],
                b4 = in(reg) b[4],
                b5 = in(reg) b[5],
                brw = out(reg_byte) borrow,
                options(pure, nomem, nostack),
            );
        }

        (r, borrow != 0)
    }

    #[inline(always)]
    fn const_ge_4(a: &[u64; 4], b: &[u64; 4]) -> bool {
        for i in 0..4 {
            if a[i] > b[i] {
                return true;
            }
            if a[i] < b[i] {
                return false;
            }
        }
        true // equal
    }

    #[inline(always)]
    fn const_ge_6(a: &[u64; 6], b: &[u64; 6]) -> bool {
        for i in 0..6 {
            if a[i] > b[i] {
                return true;
            }
            if a[i] < b[i] {
                return false;
            }
        }
        true // equal
    }

    #[inline(always)]
    fn sub_4(a: &mut [u64; 4], b: &[u64; 4]) {
        let (result, _) = sub_4_limbs_asm(a, b);
        *a = result;
    }

    #[inline(always)]
    fn sub_6(a: &mut [u64; 6], b: &[u64; 6]) {
        let (result, _) = sub_6_limbs_asm(a, b);
        *a = result;
    }

    /// Modular addition for 4 limbs: r = (a + b) mod q
    #[inline(always)]
    pub fn mod_add_4_limbs(a: &[u64; 4], b: &[u64; 4], q: &[u64; 4]) -> [u64; 4] {
        let (sum, overflow) = add_4_limbs_asm(a, b);
        if overflow || const_ge_4(&sum, q) {
            let (reduced, _) = sub_4_limbs_asm(&sum, q);
            reduced
        } else {
            sum
        }
    }

    /// Modular subtraction for 4 limbs: r = (a - b) mod q
    #[inline(always)]
    pub fn mod_sub_4_limbs(a: &[u64; 4], b: &[u64; 4], q: &[u64; 4]) -> [u64; 4] {
        let (diff, borrow) = sub_4_limbs_asm(a, b);
        if borrow {
            let (result, _) = add_4_limbs_asm(&diff, q);
            result
        } else {
            diff
        }
    }

    /// Modular addition for 6 limbs: r = (a + b) mod q
    #[inline(always)]
    pub fn mod_add_6_limbs(a: &[u64; 6], b: &[u64; 6], q: &[u64; 6]) -> [u64; 6] {
        let (sum, overflow) = add_6_limbs_asm(a, b);
        if overflow || const_ge_6(&sum, q) {
            let (reduced, _) = sub_6_limbs_asm(&sum, q);
            reduced
        } else {
            sum
        }
    }

    /// Modular subtraction for 6 limbs: r = (a - b) mod q
    #[inline(always)]
    pub fn mod_sub_6_limbs(a: &[u64; 6], b: &[u64; 6], q: &[u64; 6]) -> [u64; 6] {
        let (diff, borrow) = sub_6_limbs_asm(a, b);
        if borrow {
            let (result, _) = add_6_limbs_asm(&diff, q);
            result
        } else {
            diff
        }
    }

    /// x86-64 optimized CIOS Montgomery multiplication for 4 limbs (256-bit)
    /// Uses Rust u128 arithmetic which LLVM optimizes to MUL/MULX instructions
    /// Note: lambdaworks uses big-endian limb order (limbs[0] = MSB)
    #[inline(always)]
    pub fn cios_4_limbs(a: &[u64; 4], b: &[u64; 4], q: &[u64; 4], mu: u64) -> [u64; 4] {
        const N: usize = 4;
        let mut t = [0u64; N];
        let mut t_extra = [0u64; 2];

        // Iterate from LSB to MSB (i = N-1 down to 0)
        for i in (0..N).rev() {
            // t += a * b[i]
            let mut c: u128 = 0;
            for j in (0..N).rev() {
                let cs = t[j] as u128 + (a[j] as u128) * (b[i] as u128) + c;
                c = cs >> 64;
                t[j] = cs as u64;
            }
            let cs = (t_extra[1] as u128) + c;
            t_extra[0] = (cs >> 64) as u64;
            t_extra[1] = cs as u64;

            // m := t[N-1] * mu mod 2^64
            let m = t[N - 1].wrapping_mul(mu) as u128;

            // t += m * q, then shift right by 64
            let mut c: u128 = (t[N - 1] as u128 + m * (q[N - 1] as u128)) >> 64;
            for j in (0..N - 1).rev() {
                let cs = t[j] as u128 + m * (q[j] as u128) + c;
                c = cs >> 64;
                t[j + 1] = cs as u64;
            }
            let cs = (t_extra[1] as u128) + c;
            c = cs >> 64;
            t[0] = cs as u64;
            t_extra[1] = t_extra[0] + c as u64;
            t_extra[0] = 0;
        }

        // Final reduction
        let overflow = t_extra[1] > 0;
        if overflow || const_ge_4(&t, q) {
            sub_4(&mut t, q);
        }
        t
    }

    /// x86-64 optimized CIOS Montgomery multiplication for 6 limbs (384-bit)
    #[inline(always)]
    pub fn cios_6_limbs(a: &[u64; 6], b: &[u64; 6], q: &[u64; 6], mu: u64) -> [u64; 6] {
        const N: usize = 6;
        let mut t = [0u64; N];
        let mut t_extra = [0u64; 2];

        for i in (0..N).rev() {
            // t += a * b[i]
            let mut c: u128 = 0;
            for j in (0..N).rev() {
                let cs = t[j] as u128 + (a[j] as u128) * (b[i] as u128) + c;
                c = cs >> 64;
                t[j] = cs as u64;
            }
            let cs = (t_extra[1] as u128) + c;
            t_extra[0] = (cs >> 64) as u64;
            t_extra[1] = cs as u64;

            // m := t[N-1] * mu mod 2^64
            let m = t[N - 1].wrapping_mul(mu) as u128;

            // t += m * q, then shift right by 64
            let mut c: u128 = (t[N - 1] as u128 + m * (q[N - 1] as u128)) >> 64;
            for j in (0..N - 1).rev() {
                let cs = t[j] as u128 + m * (q[j] as u128) + c;
                c = cs >> 64;
                t[j + 1] = cs as u64;
            }
            let cs = (t_extra[1] as u128) + c;
            c = cs >> 64;
            t[0] = cs as u64;
            t_extra[1] = t_extra[0] + c as u64;
            t_extra[0] = 0;
        }

        // Final reduction
        let overflow = t_extra[1] > 0;
        if overflow || const_ge_6(&t, q) {
            sub_6(&mut t, q);
        }
        t
    }

    /// x86-64 optimized CIOS for 4 limbs with spare bit optimization (EdMSM Algorithm 2)
    /// For moduli where the high limb is < 2^63 - 1
    #[inline(always)]
    pub fn cios_4_limbs_optimized(a: &[u64; 4], b: &[u64; 4], q: &[u64; 4], mu: u64) -> [u64; 4] {
        const N: usize = 4;
        let mut t = [0u64; N];

        for i in (0..N).rev() {
            // t += a * b[i]
            let mut c: u128 = 0;
            for j in (0..N).rev() {
                let cs = t[j] as u128 + (a[j] as u128) * (b[i] as u128) + c;
                c = cs >> 64;
                t[j] = cs as u64;
            }
            let t_extra = c as u64;

            // m := t[N-1] * mu mod 2^64
            let m = t[N - 1].wrapping_mul(mu) as u128;

            // t += m * q, then shift right by 64
            let mut c: u128 = (t[N - 1] as u128 + m * (q[N - 1] as u128)) >> 64;
            for j in (0..N - 1).rev() {
                let cs = t[j] as u128 + m * (q[j] as u128) + c;
                c = cs >> 64;
                t[j + 1] = cs as u64;
            }
            let cs = (t_extra as u128) + c;
            t[0] = cs as u64;
        }

        // Final reduction (no overflow possible with spare bit)
        if const_ge_4(&t, q) {
            sub_4(&mut t, q);
        }
        t
    }

    /// x86-64 optimized CIOS for 6 limbs with spare bit optimization
    #[inline(always)]
    pub fn cios_6_limbs_optimized(a: &[u64; 6], b: &[u64; 6], q: &[u64; 6], mu: u64) -> [u64; 6] {
        const N: usize = 6;
        let mut t = [0u64; N];

        for i in (0..N).rev() {
            // t += a * b[i]
            let mut c: u128 = 0;
            for j in (0..N).rev() {
                let cs = t[j] as u128 + (a[j] as u128) * (b[i] as u128) + c;
                c = cs >> 64;
                t[j] = cs as u64;
            }
            let t_extra = c as u64;

            // m := t[N-1] * mu mod 2^64
            let m = t[N - 1].wrapping_mul(mu) as u128;

            // t += m * q, then shift right by 64
            let mut c: u128 = (t[N - 1] as u128 + m * (q[N - 1] as u128)) >> 64;
            for j in (0..N - 1).rev() {
                let cs = t[j] as u128 + m * (q[j] as u128) + c;
                c = cs >> 64;
                t[j + 1] = cs as u64;
            }
            let cs = (t_extra as u128) + c;
            t[0] = cs as u64;
        }

        // Final reduction (no overflow possible with spare bit)
        if const_ge_6(&t, q) {
            sub_6(&mut t, q);
        }
        t
    }
}

// x86-64 assembly dispatch methods
#[cfg(all(target_arch = "x86_64", feature = "asm"))]
impl MontgomeryAlgorithms {
    /// x86-64 assembly optimized CIOS multiplication
    /// Automatically dispatches to the appropriate implementation based on NUM_LIMBS
    ///
    /// # Safety invariant
    /// The match arms are only reached when NUM_LIMBS equals the expected value,
    /// so the slice-to-array conversions are guaranteed to succeed.
    #[inline(always)]
    pub fn cios_asm<const NUM_LIMBS: usize>(
        a: &UnsignedInteger<NUM_LIMBS>,
        b: &UnsignedInteger<NUM_LIMBS>,
        q: &UnsignedInteger<NUM_LIMBS>,
        mu: &u64,
    ) -> UnsignedInteger<NUM_LIMBS> {
        // Dispatch to specialized assembly implementations
        // SAFETY: Each match arm only executes when NUM_LIMBS equals the array size,
        // so try_into() is guaranteed to succeed. We use unwrap() to match the
        // existing aarch64_asm pattern in this file.
        match NUM_LIMBS {
            4 => {
                let a_arr: &[u64; 4] = a.limbs.as_slice().try_into().unwrap();
                let b_arr: &[u64; 4] = b.limbs.as_slice().try_into().unwrap();
                let q_arr: &[u64; 4] = q.limbs.as_slice().try_into().unwrap();
                let result = x86_64_asm::cios_4_limbs(a_arr, b_arr, q_arr, *mu);
                let mut limbs = [0u64; NUM_LIMBS];
                limbs.copy_from_slice(&result);
                UnsignedInteger { limbs }
            }
            6 => {
                let a_arr: &[u64; 6] = a.limbs.as_slice().try_into().unwrap();
                let b_arr: &[u64; 6] = b.limbs.as_slice().try_into().unwrap();
                let q_arr: &[u64; 6] = q.limbs.as_slice().try_into().unwrap();
                let result = x86_64_asm::cios_6_limbs(a_arr, b_arr, q_arr, *mu);
                let mut limbs = [0u64; NUM_LIMBS];
                limbs.copy_from_slice(&result);
                UnsignedInteger { limbs }
            }
            // Fallback to pure Rust for other sizes
            _ => Self::cios(a, b, q, mu),
        }
    }

    /// x86-64 optimized CIOS for moduli with one spare bit (EdMSM Algorithm 2)
    ///
    /// # Safety invariant
    /// The match arms are only reached when NUM_LIMBS equals the expected value.
    #[inline(always)]
    pub fn cios_asm_optimized<const NUM_LIMBS: usize>(
        a: &UnsignedInteger<NUM_LIMBS>,
        b: &UnsignedInteger<NUM_LIMBS>,
        q: &UnsignedInteger<NUM_LIMBS>,
        mu: &u64,
    ) -> UnsignedInteger<NUM_LIMBS> {
        // SAFETY: Each match arm only executes when NUM_LIMBS equals the array size.
        match NUM_LIMBS {
            4 => {
                let a_arr: &[u64; 4] = a.limbs.as_slice().try_into().unwrap();
                let b_arr: &[u64; 4] = b.limbs.as_slice().try_into().unwrap();
                let q_arr: &[u64; 4] = q.limbs.as_slice().try_into().unwrap();
                let result = x86_64_asm::cios_4_limbs_optimized(a_arr, b_arr, q_arr, *mu);
                let mut limbs = [0u64; NUM_LIMBS];
                limbs.copy_from_slice(&result);
                UnsignedInteger { limbs }
            }
            6 => {
                let a_arr: &[u64; 6] = a.limbs.as_slice().try_into().unwrap();
                let b_arr: &[u64; 6] = b.limbs.as_slice().try_into().unwrap();
                let q_arr: &[u64; 6] = q.limbs.as_slice().try_into().unwrap();
                let result = x86_64_asm::cios_6_limbs_optimized(a_arr, b_arr, q_arr, *mu);
                let mut limbs = [0u64; NUM_LIMBS];
                limbs.copy_from_slice(&result);
                UnsignedInteger { limbs }
            }
            // Fallback to pure Rust optimized version for other sizes
            _ => Self::cios_optimized_for_moduli_with_one_spare_bit(a, b, q, mu),
        }
    }

    /// x86-64 optimized modular addition
    ///
    /// # Safety invariant
    /// The match arms are only reached when NUM_LIMBS equals the expected value.
    #[inline(always)]
    pub fn add_asm<const NUM_LIMBS: usize>(
        a: &UnsignedInteger<NUM_LIMBS>,
        b: &UnsignedInteger<NUM_LIMBS>,
        q: &UnsignedInteger<NUM_LIMBS>,
    ) -> UnsignedInteger<NUM_LIMBS> {
        // SAFETY: Each match arm only executes when NUM_LIMBS equals the array size.
        match NUM_LIMBS {
            4 => {
                let a_arr: &[u64; 4] = a.limbs.as_slice().try_into().unwrap();
                let b_arr: &[u64; 4] = b.limbs.as_slice().try_into().unwrap();
                let q_arr: &[u64; 4] = q.limbs.as_slice().try_into().unwrap();
                let result = x86_64_asm::mod_add_4_limbs(a_arr, b_arr, q_arr);
                let mut limbs = [0u64; NUM_LIMBS];
                limbs.copy_from_slice(&result);
                UnsignedInteger { limbs }
            }
            6 => {
                let a_arr: &[u64; 6] = a.limbs.as_slice().try_into().unwrap();
                let b_arr: &[u64; 6] = b.limbs.as_slice().try_into().unwrap();
                let q_arr: &[u64; 6] = q.limbs.as_slice().try_into().unwrap();
                let result = x86_64_asm::mod_add_6_limbs(a_arr, b_arr, q_arr);
                let mut limbs = [0u64; NUM_LIMBS];
                limbs.copy_from_slice(&result);
                UnsignedInteger { limbs }
            }
            _ => {
                let (sum, overflow) = UnsignedInteger::add(a, b);
                if overflow || sum >= *q {
                    UnsignedInteger::sub(&sum, q).0
                } else {
                    sum
                }
            }
        }
    }

    /// x86-64 optimized modular subtraction
    ///
    /// # Safety invariant
    /// The match arms are only reached when NUM_LIMBS equals the expected value.
    #[inline(always)]
    pub fn sub_asm<const NUM_LIMBS: usize>(
        a: &UnsignedInteger<NUM_LIMBS>,
        b: &UnsignedInteger<NUM_LIMBS>,
        q: &UnsignedInteger<NUM_LIMBS>,
    ) -> UnsignedInteger<NUM_LIMBS> {
        // SAFETY: Each match arm only executes when NUM_LIMBS equals the array size.
        match NUM_LIMBS {
            4 => {
                let a_arr: &[u64; 4] = a.limbs.as_slice().try_into().unwrap();
                let b_arr: &[u64; 4] = b.limbs.as_slice().try_into().unwrap();
                let q_arr: &[u64; 4] = q.limbs.as_slice().try_into().unwrap();
                let result = x86_64_asm::mod_sub_4_limbs(a_arr, b_arr, q_arr);
                let mut limbs = [0u64; NUM_LIMBS];
                limbs.copy_from_slice(&result);
                UnsignedInteger { limbs }
            }
            6 => {
                let a_arr: &[u64; 6] = a.limbs.as_slice().try_into().unwrap();
                let b_arr: &[u64; 6] = b.limbs.as_slice().try_into().unwrap();
                let q_arr: &[u64; 6] = q.limbs.as_slice().try_into().unwrap();
                let result = x86_64_asm::mod_sub_6_limbs(a_arr, b_arr, q_arr);
                let mut limbs = [0u64; NUM_LIMBS];
                limbs.copy_from_slice(&result);
                UnsignedInteger { limbs }
            }
            _ => {
                if b <= a {
                    *a - *b
                } else {
                    *q - (*b - *a)
                }
            }
        }
    }
}

impl MontgomeryAlgorithms {
    /// Compute CIOS multiplication of `a` * `b`
    /// `q` is the modulus
    /// `mu` is the inverse of -q modulo 2^{64}
    /// Notice CIOS stands for Coarsely Integrated Operand Scanning
    /// For more information see section 2.3.2 of Tolga Acar's thesis
    /// <https://www.microsoft.com/en-us/research/wp-content/uploads/1998/06/97Acar.pdf>.
    #[inline(always)]
    pub const fn cios<const NUM_LIMBS: usize>(
        a: &UnsignedInteger<NUM_LIMBS>,
        b: &UnsignedInteger<NUM_LIMBS>,
        q: &UnsignedInteger<NUM_LIMBS>,
        mu: &u64,
    ) -> UnsignedInteger<NUM_LIMBS> {
        let mut t = [0_u64; NUM_LIMBS];
        let mut t_extra = [0_u64; 2];
        let mut i: usize = NUM_LIMBS;
        while i > 0 {
            i -= 1;
            // C := 0
            let mut c: u128 = 0;

            // for j=N-1 to 0
            //    (C,t[j]) := t[j] + a[j]*b[i] + C
            let mut cs: u128;
            let mut j: usize = NUM_LIMBS;
            while j > 0 {
                j -= 1;
                cs = t[j] as u128 + (a.limbs[j] as u128) * (b.limbs[i] as u128) + c;
                c = cs >> 64;
                t[j] = cs as u64;
            }

            // (t_extra[0],t_extra[1]) := t_extra[1] + C
            cs = (t_extra[1] as u128) + c;
            t_extra[0] = (cs >> 64) as u64;
            t_extra[1] = cs as u64;

            let mut c: u128;

            // m := t[N-1]*q'[N-1] mod D
            let m = t[NUM_LIMBS - 1].wrapping_mul(*mu) as u128;

            // (C,_) := t[N-1] + m*q[N-1]
            c = (t[NUM_LIMBS - 1] as u128 + m * (q.limbs[NUM_LIMBS - 1] as u128)) >> 64;

            // for j=N-1 to 1
            //    (C,t[j+1]) := t[j] + m*q[j] + C
            let mut j: usize = NUM_LIMBS - 1;
            while j > 0 {
                j -= 1;
                cs = t[j] as u128 + m * (q.limbs[j] as u128) + c;
                c = cs >> 64;
                t[j + 1] = cs as u64;
            }

            // (C,t[0]) := t_extra[1] + C
            cs = (t_extra[1] as u128) + c;
            c = cs >> 64;
            t[0] = cs as u64;

            // t_extra[1] := t_extra[0] + C
            t_extra[1] = t_extra[0] + c as u64;
        }
        let mut result = UnsignedInteger { limbs: t };

        let overflow = t_extra[1] > 0;

        if overflow || UnsignedInteger::const_le(q, &result) {
            (result, _) = UnsignedInteger::sub(&result, q);
        }
        result
    }

    /// Compute CIOS multiplication of `a` * `b`
    /// This is the Algorithm 2 described in the paper
    /// "EdMSM: Multi-Scalar-Multiplication for SNARKs and Faster Montgomery multiplication"
    /// <https://eprint.iacr.org/2022/1400.pdf>.
    /// It is only suited for moduli with `q[0]` smaller than `2^63 - 1`.
    /// `q` is the modulus
    /// `mu` is the inverse of -q modulo 2^{64}
    #[inline(always)]
    pub fn cios_optimized_for_moduli_with_one_spare_bit<const NUM_LIMBS: usize>(
        a: &UnsignedInteger<NUM_LIMBS>,
        b: &UnsignedInteger<NUM_LIMBS>,
        q: &UnsignedInteger<NUM_LIMBS>,
        mu: &u64,
    ) -> UnsignedInteger<NUM_LIMBS> {
        let mut t = [0_u64; NUM_LIMBS];
        let mut t_extra;
        let mut i: usize = NUM_LIMBS;
        while i > 0 {
            i -= 1;
            // C := 0
            let mut c: u128 = 0;

            // for j=N-1 to 0
            //    (C,t[j]) := t[j] + a[j]*b[i] + C
            let mut cs: u128;
            let mut j: usize = NUM_LIMBS;
            while j > 0 {
                j -= 1;
                cs = t[j] as u128 + (a.limbs[j] as u128) * (b.limbs[i] as u128) + c;
                c = cs >> 64;
                t[j] = cs as u64;
            }

            t_extra = c as u64;

            let mut c: u128;

            // m := t[N-1]*q'[N-1] mod D
            let m = t[NUM_LIMBS - 1].wrapping_mul(*mu) as u128;

            // (C,_) := t[0] + m*q[0]
            c = (t[NUM_LIMBS - 1] as u128 + m * (q.limbs[NUM_LIMBS - 1] as u128)) >> 64;

            // for j=N-1 to 1
            //    (C,t[j+1]) := t[j] + m*q[j] + C
            let mut j: usize = NUM_LIMBS - 1;
            while j > 0 {
                j -= 1;
                cs = t[j] as u128 + m * (q.limbs[j] as u128) + c;
                c = cs >> 64;
                t[j + 1] = cs as u64;
            }

            // (C,t[0]) := t_extra + C
            cs = (t_extra as u128) + c;
            t[0] = cs as u64;
        }
        let mut result = UnsignedInteger { limbs: t };

        if UnsignedInteger::const_le(q, &result) {
            (result, _) = UnsignedInteger::sub(&result, q);
        }
        result
    }

    // Separated Operand Scanning Method (2.3.1)
    #[inline(always)]
    pub fn sos_square<const NUM_LIMBS: usize>(
        a: &UnsignedInteger<NUM_LIMBS>,
        q: &UnsignedInteger<NUM_LIMBS>,
        mu: &u64,
    ) -> UnsignedInteger<NUM_LIMBS> {
        // NOTE: we use explicit `while` loops in this function because profiling pointed
        // at iterators of the form `(<x>..<y>).rev()` as the main performance bottleneck.

        // Step 1: Compute `(hi, lo) = a * a`
        let (mut hi, mut lo) = UnsignedInteger::square(a);

        // Step 2: Add terms to `(hi, lo)` until multiple it
        // is a multiple of both `2^{NUM_LIMBS * 64}` and
        // `q`.
        let mut c: u128 = 0;
        let mut i = NUM_LIMBS;
        let mut overflow = false;
        while i > 0 {
            i -= 1;
            c = 0;
            let m = (lo.limbs[i] as u128 * *mu as u128) as u64;
            let mut j = NUM_LIMBS;
            while j > 0 {
                j -= 1;
                if i + j >= NUM_LIMBS - 1 {
                    let index = i + j - (NUM_LIMBS - 1);
                    let cs = lo.limbs[index] as u128 + m as u128 * (q.limbs[j] as u128) + c;
                    c = cs >> 64;
                    lo.limbs[index] = cs as u64;
                } else {
                    let index = i + j + 1;
                    let cs = hi.limbs[index] as u128 + m as u128 * (q.limbs[j] as u128) + c;
                    c = cs >> 64;
                    hi.limbs[index] = cs as u64;
                }
            }

            // Carry propagation to `hi`
            let mut t = 0;
            while c > 0 && i >= t {
                let cs = hi.limbs[i - t] as u128 + c;
                c = cs >> 64;
                hi.limbs[i - t] = cs as u64;
                t += 1;
            }
            overflow |= c > 0;
        }

        // Step 3: At this point `overflow * 2^{2 * NUM_LIMBS * 64} + (hi, lo)` is a multiple
        // of `2^{NUM_LIMBS * 64}` and the result is obtained by dividing it by `2^{NUM_LIMBS * 64}`.
        // In other words, `lo` is zero and the result is
        // `overflow * 2^{NUM_LIMBS * 64} + hi`.
        // That number is always strictly smaller than `2 * q`. To normalize it we substract
        // `q` whenever it is larger than `q`.
        // The easy case is when `overflow` is zero. We just use the `sub` function.
        // If `overflow` is 1, then `hi` is smaller than `q`. The function `sub(hi, q)` wraps
        // around `2^{NUM_LIMBS * 64}`. This is the result we need.
        overflow |= c > 0;
        if overflow || UnsignedInteger::const_le(q, &hi) {
            (hi, _) = UnsignedInteger::sub(&hi, q);
        }
        hi
    }
}

#[cfg(test)]
mod tests {
    use crate::unsigned_integer::{element::U384, montgomery::MontgomeryAlgorithms};

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn cios_vs_cios_optimized(a in any::<[u64; 6]>(), b in any::<[u64; 6]>()) {
            let x = U384::from_limbs(a);
            let y = U384::from_limbs(b);
            let m = U384::from_hex_unchecked("cdb061954fdd36e5176f50dbdcfd349570a29ce1"); // this is prime
            let mu: u64 = 16085280245840369887; // negative of the inverse of `m` modulo 2^{64}
            assert_eq!(
                MontgomeryAlgorithms::cios(&x, &y, &m, &mu),
                MontgomeryAlgorithms::cios_optimized_for_moduli_with_one_spare_bit(&x, &y, &m, &mu)
            );
        }

        #[test]
        fn cios_vs_sos_square(a in any::<[u64; 6]>()) {
            let x = U384::from_limbs(a);
            let m = U384::from_hex_unchecked("cdb061954fdd36e5176f50dbdcfd349570a29ce1"); // this is prime
            let mu: u64 = 16085280245840369887; // negative of the inverse of `m` modulo 2^{64}
            assert_eq!(
                MontgomeryAlgorithms::cios(&x, &x, &m, &mu),
                MontgomeryAlgorithms::sos_square(&x, &m, &mu)
            );
        }
    }
    #[test]
    fn montgomery_multiplication_works_0() {
        let x = U384::from_u64(11_u64);
        let y = U384::from_u64(10_u64);
        let m = U384::from_u64(23_u64); //
        let mu: u64 = 3208129404123400281; // negative of the inverse of `m` modulo 2^{64}.
        let c = U384::from_u64(13_u64); // x * y * (r^{-1}) % m, where r = 2^{64 * 6} and r^{-1} mod m = 2.
        assert_eq!(MontgomeryAlgorithms::cios(&x, &y, &m, &mu), c);
    }

    #[test]
    fn montgomery_multiplication_works_1() {
        let x = U384::from_hex_unchecked("05ed176deb0e80b4deb7718cdaa075165f149c");
        let y = U384::from_hex_unchecked("5f103b0bd4397d4df560eb559f38353f80eeb6");
        let m = U384::from_hex_unchecked("cdb061954fdd36e5176f50dbdcfd349570a29ce1"); // this is prime
        let mu: u64 = 16085280245840369887; // negative of the inverse of `m` modulo 2^{64}
        let c = U384::from_hex_unchecked("8d65cdee621682815d59f465d2641eea8a1274dc"); // x * y * (r^{-1}) % m, where r = 2^{64 * 6}
        assert_eq!(MontgomeryAlgorithms::cios(&x, &y, &m, &mu), c);
    }

    #[test]
    fn montgomery_multiplication_works_2() {
        let x = U384::from_hex_unchecked("8d65cdee621682815d59f465d2641eea8a1274dc");
        let m = U384::from_hex_unchecked("cdb061954fdd36e5176f50dbdcfd349570a29ce1"); // this is prime
        let r_mod_m = U384::from_hex_unchecked("58dfb0e1b3dd5e674bdcde4f42eb5533b8759d33");
        let mu: u64 = 16085280245840369887; // negative of the inverse of `m` modulo 2^{64}
        let c = U384::from_hex_unchecked("8d65cdee621682815d59f465d2641eea8a1274dc");
        assert_eq!(MontgomeryAlgorithms::cios(&x, &r_mod_m, &m, &mu), c);
    }

    // ARM64 assembly tests
    #[cfg(all(target_arch = "aarch64", feature = "asm"))]
    mod asm_tests {
        use super::*;
        use crate::unsigned_integer::element::U256;

        proptest! {
            #[test]
            fn cios_asm_matches_cios_6_limbs(a in any::<[u64; 6]>(), b in any::<[u64; 6]>()) {
                let x = U384::from_limbs(a);
                let y = U384::from_limbs(b);
                let m = U384::from_hex_unchecked("cdb061954fdd36e5176f50dbdcfd349570a29ce1");
                let mu: u64 = 16085280245840369887;

                let rust_result = MontgomeryAlgorithms::cios(&x, &y, &m, &mu);
                let asm_result = MontgomeryAlgorithms::cios_asm(&x, &y, &m, &mu);

                prop_assert_eq!(rust_result, asm_result);
            }

            #[test]
            fn cios_asm_matches_cios_4_limbs(a in any::<[u64; 4]>(), b in any::<[u64; 4]>()) {
                let x = U256::from_limbs(a);
                let y = U256::from_limbs(b);
                // BN254 scalar field modulus
                let m = U256::from_hex_unchecked("30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001");
                let mu: u64 = 14042775128853446655; // computed -m^{-1} mod 2^64

                let rust_result = MontgomeryAlgorithms::cios(&x, &y, &m, &mu);
                let asm_result = MontgomeryAlgorithms::cios_asm(&x, &y, &m, &mu);

                prop_assert_eq!(rust_result, asm_result);
            }
        }

        #[test]
        fn cios_asm_4_limbs_simple() {
            let x = U256::from_u64(11_u64);
            let y = U256::from_u64(10_u64);
            let m = U256::from_u64(23_u64);
            let mu: u64 = 3208129404123400281;

            let rust_result = MontgomeryAlgorithms::cios(&x, &y, &m, &mu);
            let asm_result = MontgomeryAlgorithms::cios_asm(&x, &y, &m, &mu);

            assert_eq!(rust_result, asm_result);
        }

        #[test]
        fn cios_asm_6_limbs_simple() {
            let x = U384::from_u64(11_u64);
            let y = U384::from_u64(10_u64);
            let m = U384::from_u64(23_u64);
            let mu: u64 = 3208129404123400281;

            let rust_result = MontgomeryAlgorithms::cios(&x, &y, &m, &mu);
            let asm_result = MontgomeryAlgorithms::cios_asm(&x, &y, &m, &mu);

            assert_eq!(rust_result, asm_result);
        }

        proptest! {
            // Test cios_asm_optimized matches cios_optimized_for_moduli_with_one_spare_bit
            #[test]
            fn cios_asm_optimized_matches_rust_6_limbs(a in any::<[u64; 6]>(), b in any::<[u64; 6]>()) {
                let x = U384::from_limbs(a);
                let y = U384::from_limbs(b);
                // BLS12-381 has spare bit
                let m = U384::from_hex_unchecked("1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab");
                let mu: u64 = 9940570264628428797;

                let rust_result = MontgomeryAlgorithms::cios_optimized_for_moduli_with_one_spare_bit(&x, &y, &m, &mu);
                let asm_result = MontgomeryAlgorithms::cios_asm_optimized(&x, &y, &m, &mu);

                prop_assert_eq!(rust_result, asm_result);
            }

            #[test]
            fn cios_asm_optimized_matches_rust_4_limbs(a in any::<[u64; 4]>(), b in any::<[u64; 4]>()) {
                let x = U256::from_limbs(a);
                let y = U256::from_limbs(b);
                // BN254 scalar field (has spare bit)
                let m = U256::from_hex_unchecked("30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001");
                let mu: u64 = 14042775128853446655;

                let rust_result = MontgomeryAlgorithms::cios_optimized_for_moduli_with_one_spare_bit(&x, &y, &m, &mu);
                let asm_result = MontgomeryAlgorithms::cios_asm_optimized(&x, &y, &m, &mu);

                prop_assert_eq!(rust_result, asm_result);
            }

            // Test add_asm matches pure Rust add
            #[test]
            fn add_asm_matches_rust_4_limbs(a in any::<[u64; 4]>(), b in any::<[u64; 4]>()) {
                let x = U256::from_limbs(a);
                let y = U256::from_limbs(b);
                let m = U256::from_hex_unchecked("30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001");

                // Pure Rust implementation
                let (sum, overflow) = U256::add(&x, &y);
                let rust_result = if overflow || sum >= m {
                    U256::sub(&sum, &m).0
                } else {
                    sum
                };

                let asm_result = MontgomeryAlgorithms::add_asm(&x, &y, &m);
                prop_assert_eq!(rust_result, asm_result);
            }

            #[test]
            fn add_asm_matches_rust_6_limbs(a in any::<[u64; 6]>(), b in any::<[u64; 6]>()) {
                let x = U384::from_limbs(a);
                let y = U384::from_limbs(b);
                let m = U384::from_hex_unchecked("1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab");

                // Pure Rust implementation
                let (sum, overflow) = U384::add(&x, &y);
                let rust_result = if overflow || sum >= m {
                    U384::sub(&sum, &m).0
                } else {
                    sum
                };

                let asm_result = MontgomeryAlgorithms::add_asm(&x, &y, &m);
                prop_assert_eq!(rust_result, asm_result);
            }

            // Test sub_asm matches pure Rust sub
            #[test]
            fn sub_asm_matches_rust_4_limbs(a in any::<[u64; 4]>(), b in any::<[u64; 4]>()) {
                let x = U256::from_limbs(a);
                let y = U256::from_limbs(b);
                let m = U256::from_hex_unchecked("30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001");

                // Pure Rust implementation (using sub that returns tuple to avoid panic)
                let (diff, borrow) = U256::sub(&x, &y);
                let rust_result = if borrow {
                    U256::add(&diff, &m).0
                } else {
                    diff
                };

                let asm_result = MontgomeryAlgorithms::sub_asm(&x, &y, &m);
                prop_assert_eq!(rust_result, asm_result);
            }

            #[test]
            fn sub_asm_matches_rust_6_limbs(a in any::<[u64; 6]>(), b in any::<[u64; 6]>()) {
                let x = U384::from_limbs(a);
                let y = U384::from_limbs(b);
                let m = U384::from_hex_unchecked("1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab");

                // Pure Rust implementation (using sub that returns tuple to avoid panic)
                let (diff, borrow) = U384::sub(&x, &y);
                let rust_result = if borrow {
                    U384::add(&diff, &m).0
                } else {
                    diff
                };

                let asm_result = MontgomeryAlgorithms::sub_asm(&x, &y, &m);
                prop_assert_eq!(rust_result, asm_result);
            }
        }

        // Unit tests for edge cases
        #[test]
        fn add_asm_4_limbs_overflow() {
            // Test addition that causes overflow and needs reduction
            let m = U256::from_hex_unchecked(
                "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001",
            );
            let x = m - U256::from_u64(1); // m - 1
            let y = U256::from_u64(2);

            let result = MontgomeryAlgorithms::add_asm(&x, &y, &m);
            assert_eq!(result, U256::from_u64(1)); // (m-1) + 2 mod m = 1
        }

        #[test]
        fn sub_asm_4_limbs_borrow() {
            // Test subtraction that causes borrow and needs modulus addition
            let m = U256::from_hex_unchecked(
                "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001",
            );
            let x = U256::from_u64(1);
            let y = U256::from_u64(2);

            let result = MontgomeryAlgorithms::sub_asm(&x, &y, &m);
            assert_eq!(result, m - U256::from_u64(1)); // 1 - 2 mod m = m - 1
        }

        #[test]
        fn add_asm_6_limbs_overflow() {
            let m = U384::from_hex_unchecked("1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab");
            let x = m - U384::from_u64(1);
            let y = U384::from_u64(2);

            let result = MontgomeryAlgorithms::add_asm(&x, &y, &m);
            assert_eq!(result, U384::from_u64(1));
        }

        #[test]
        fn sub_asm_6_limbs_borrow() {
            let m = U384::from_hex_unchecked("1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab");
            let x = U384::from_u64(1);
            let y = U384::from_u64(2);

            let result = MontgomeryAlgorithms::sub_asm(&x, &y, &m);
            assert_eq!(result, m - U384::from_u64(1));
        }

        // Tests for true inline assembly implementations
        proptest! {
            #[test]
            fn cios_true_asm_optimized_matches_rust_4_limbs(a in any::<[u64; 4]>(), b in any::<[u64; 4]>()) {
                let x = U256::from_limbs(a);
                let y = U256::from_limbs(b);
                // BN254 scalar field (has spare bit)
                let m = U256::from_hex_unchecked("30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001");
                let mu: u64 = 14042775128853446655;

                let rust_result = MontgomeryAlgorithms::cios_optimized_for_moduli_with_one_spare_bit(&x, &y, &m, &mu);
                let asm_result = MontgomeryAlgorithms::cios_true_asm_optimized(&x, &y, &m, &mu);

                prop_assert_eq!(rust_result, asm_result);
            }

            #[test]
            fn cios_true_asm_optimized_matches_rust_6_limbs(a in any::<[u64; 6]>(), b in any::<[u64; 6]>()) {
                let x = U384::from_limbs(a);
                let y = U384::from_limbs(b);
                // BLS12-381 has spare bit
                let m = U384::from_hex_unchecked("1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab");
                let mu: u64 = 9940570264628428797;

                let rust_result = MontgomeryAlgorithms::cios_optimized_for_moduli_with_one_spare_bit(&x, &y, &m, &mu);
                let asm_result = MontgomeryAlgorithms::cios_true_asm_optimized(&x, &y, &m, &mu);

                prop_assert_eq!(rust_result, asm_result);
            }
        }

        // Unit tests for true inline assembly implementations
        #[test]
        fn cios_true_asm_4_limbs_simple() {
            let x = U256::from_u64(11_u64);
            let y = U256::from_u64(10_u64);
            let m = U256::from_u64(23_u64);
            let mu: u64 = 3208129404123400281;

            let rust_result =
                MontgomeryAlgorithms::cios_optimized_for_moduli_with_one_spare_bit(&x, &y, &m, &mu);
            let asm_result = MontgomeryAlgorithms::cios_true_asm_optimized(&x, &y, &m, &mu);

            assert_eq!(rust_result, asm_result);
        }

        #[test]
        fn cios_true_asm_6_limbs_simple() {
            let x = U384::from_u64(11_u64);
            let y = U384::from_u64(10_u64);
            let m = U384::from_u64(23_u64);
            let mu: u64 = 3208129404123400281;

            let rust_result =
                MontgomeryAlgorithms::cios_optimized_for_moduli_with_one_spare_bit(&x, &y, &m, &mu);
            let asm_result = MontgomeryAlgorithms::cios_true_asm_optimized(&x, &y, &m, &mu);

            assert_eq!(rust_result, asm_result);
        }

        #[test]
        fn cios_true_asm_4_limbs_bn254_modulus() {
            // Test with realistic BN254 values
            let m = U256::from_hex_unchecked(
                "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001",
            );
            let mu: u64 = 14042775128853446655;

            let x = U256::from_hex_unchecked(
                "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
            );
            let y = U256::from_hex_unchecked(
                "fedcba0987654321fedcba0987654321fedcba0987654321fedcba0987654321",
            );

            let rust_result =
                MontgomeryAlgorithms::cios_optimized_for_moduli_with_one_spare_bit(&x, &y, &m, &mu);
            let asm_result = MontgomeryAlgorithms::cios_true_asm_optimized(&x, &y, &m, &mu);

            assert_eq!(rust_result, asm_result);
        }

        #[test]
        fn cios_true_asm_6_limbs_bls12381_modulus() {
            // Test with realistic BLS12-381 values
            let m = U384::from_hex_unchecked("1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab");
            let mu: u64 = 9940570264628428797;

            let x = U384::from_hex_unchecked("123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef0");
            let y = U384::from_hex_unchecked("fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210");

            let rust_result =
                MontgomeryAlgorithms::cios_optimized_for_moduli_with_one_spare_bit(&x, &y, &m, &mu);
            let asm_result = MontgomeryAlgorithms::cios_true_asm_optimized(&x, &y, &m, &mu);

            assert_eq!(rust_result, asm_result);
        }
    }
}
