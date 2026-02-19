//! Metal GPU FFT wrapper functions for the STARK prover.
//!
//! Provides high-level `gpu_interpolate_fft` and `gpu_evaluate_offset_fft` functions
//! that use the existing Metal FFT infrastructure from `lambdaworks-math` for
//! Goldilocks-compatible fields.

#[cfg(all(target_os = "macos", feature = "metal"))]
use lambdaworks_gpu::metal::abstractions::{errors::MetalError, state::MetalState};

#[cfg(all(target_os = "macos", feature = "metal"))]
use lambdaworks_math::{
    fft::gpu::metal::ops::{fft, fft_to_buffer, gen_twiddles, gen_twiddles_to_buffer},
    field::{
        element::FieldElement,
        traits::{IsFFTField, IsSubFieldOf, RootsConfig},
    },
};

/// Interpolates polynomial coefficients from evaluations using Metal GPU FFT.
///
/// Given `evaluations` at roots of unity, this computes the polynomial coefficients
/// via inverse FFT on the GPU. Equivalent to `Polynomial::interpolate_fft` but
/// executed on Metal.
///
/// # Algorithm
///
/// 1. Generate inverse twiddle factors with `RootsConfig::BitReverseInversed`
/// 2. Run forward FFT with inverse twiddles (this performs the inverse transform)
/// 3. Normalize each coefficient by multiplying by `1/n`
///
/// # Type Parameters
///
/// - `F`: An FFT-compatible field (must provide primitive roots of unity)
///
/// # Errors
///
/// Returns `MetalError` if GPU twiddle generation or FFT execution fails.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_interpolate_fft<F>(
    evaluations: &[FieldElement<F>],
    state: &MetalState,
) -> Result<Vec<FieldElement<F>>, MetalError>
where
    F: IsFFTField + IsSubFieldOf<F>,
    F::BaseType: Copy,
{
    let order = evaluations.len().trailing_zeros() as u64;
    let inv_twiddles = gen_twiddles::<F>(order, RootsConfig::BitReverseInversed, state)?;

    let coeffs = fft(evaluations, &inv_twiddles, state)?;

    // Normalize by 1/n
    let n_inv = FieldElement::<F>::from(evaluations.len() as u64)
        .inv()
        .expect("Power-of-two length is always invertible in an FFT field");

    Ok(coeffs.iter().map(|c| c * &n_inv).collect())
}

/// Evaluates a polynomial on an offset coset domain using Metal GPU FFT.
///
/// Given polynomial `coefficients`, this computes evaluations on the domain
/// `{offset * w^i}` where `w` is a primitive root of unity of order
/// `coefficients.len() * blowup_factor`. This is the Low-Degree Extension (LDE)
/// operation used in STARK provers.
///
/// # Algorithm
///
/// 1. Multiply coefficient `k` by `offset^k` (coset shift)
/// 2. Zero-pad to `domain_size = len * blowup_factor`
/// 3. Generate forward twiddle factors with `RootsConfig::BitReverse`
/// 4. Run forward FFT on the GPU
///
/// # Type Parameters
///
/// - `F`: An FFT-compatible field (must provide primitive roots of unity)
///
/// # Arguments
///
/// - `coefficients`: The polynomial coefficients
/// - `blowup_factor`: The LDE blowup factor (must be a power of two)
/// - `offset`: The coset offset element
/// - `state`: Metal GPU state
///
/// # Errors
///
/// Returns `MetalError` if GPU twiddle generation or FFT execution fails.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_evaluate_offset_fft<F>(
    coefficients: &[FieldElement<F>],
    blowup_factor: usize,
    offset: &FieldElement<F>,
    state: &MetalState,
) -> Result<Vec<FieldElement<F>>, MetalError>
where
    F: IsFFTField + IsSubFieldOf<F>,
    F::BaseType: Copy,
{
    let domain_size = coefficients.len() * blowup_factor;

    // Step 1: Multiply coefficient k by offset^k (coset shift)
    let mut shifted = Vec::with_capacity(domain_size);
    let mut offset_power = FieldElement::<F>::one();
    for coeff in coefficients {
        shifted.push(coeff * &offset_power);
        offset_power = &offset_power * offset;
    }

    // Step 2: Zero-pad to domain_size
    shifted.resize(domain_size, FieldElement::zero());

    // Step 3 & 4: Generate forward twiddles and run FFT
    let order = domain_size.trailing_zeros() as u64;
    let twiddles = gen_twiddles::<F>(order, RootsConfig::BitReverse, state)?;
    fft(&shifted, &twiddles, state)
}

/// Evaluates a polynomial on an offset coset domain, returning a GPU Metal Buffer.
///
/// Like [`gpu_evaluate_offset_fft`] but keeps the result on GPU for direct use by
/// downstream GPU operations (e.g., Merkle tree hashing) without CPU readback.
///
/// The returned buffer contains `F::BaseType` elements in bit-reversed order
/// (natural FFT output order after bit-reverse permutation).
///
/// # Returns
///
/// A tuple of (Metal Buffer, element count) where the buffer contains the
/// bit-reversed FFT evaluations.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_evaluate_offset_fft_to_buffer<F>(
    coefficients: &[FieldElement<F>],
    blowup_factor: usize,
    offset: &FieldElement<F>,
    state: &MetalState,
) -> Result<(metal::Buffer, usize), MetalError>
where
    F: IsFFTField + IsSubFieldOf<F>,
    F::BaseType: Copy,
{
    let domain_size = coefficients.len() * blowup_factor;

    // Step 1: Multiply coefficient k by offset^k (coset shift)
    let mut shifted = Vec::with_capacity(domain_size);
    let mut offset_power = FieldElement::<F>::one();
    for coeff in coefficients {
        shifted.push(coeff * &offset_power);
        offset_power = &offset_power * offset;
    }

    // Step 2: Zero-pad to domain_size
    shifted.resize(domain_size, FieldElement::zero());

    // Step 3: Generate twiddles directly as GPU buffer (no CPU download)
    let order = domain_size.trailing_zeros() as u64;
    let twiddles_buffer = gen_twiddles_to_buffer::<F>(order, RootsConfig::BitReverse, state)?;

    // Step 4: FFT with result staying on GPU
    let result_buffer = fft_to_buffer::<F>(&shifted, &twiddles_buffer, state)?;
    Ok((result_buffer, domain_size))
}

/// Batch-evaluates multiple polynomials on the same offset coset domain, returning GPU Buffers.
///
/// Like calling [`gpu_evaluate_offset_fft_to_buffer`] for each polynomial but generates
/// twiddle factors only once and reuses them for all polynomials in the batch.
///
/// Returns a vector of (Metal Buffer, element count) pairs, one per polynomial.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub fn gpu_evaluate_offset_fft_to_buffers_batch<F>(
    polynomials: &[&[FieldElement<F>]],
    blowup_factor: usize,
    offset: &FieldElement<F>,
    state: &MetalState,
) -> Result<Vec<(metal::Buffer, usize)>, MetalError>
where
    F: IsFFTField + IsSubFieldOf<F>,
    F::BaseType: Copy,
{
    if polynomials.is_empty() {
        return Ok(Vec::new());
    }

    let domain_size = polynomials[0].len() * blowup_factor;
    let order = domain_size.trailing_zeros() as u64;

    // Generate twiddles ONCE for the shared domain
    let twiddles_buffer = gen_twiddles_to_buffer::<F>(order, RootsConfig::BitReverse, state)?;

    let mut results = Vec::with_capacity(polynomials.len());

    for coefficients in polynomials {
        let poly_domain_size = coefficients.len() * blowup_factor;

        // Coset shift: multiply coefficient k by offset^k
        let mut shifted = Vec::with_capacity(poly_domain_size);
        let mut offset_power = FieldElement::<F>::one();
        for coeff in *coefficients {
            shifted.push(coeff * &offset_power);
            offset_power = &offset_power * offset;
        }
        shifted.resize(poly_domain_size, FieldElement::zero());

        // FFT with shared twiddles, result stays on GPU
        let result_buffer = fft_to_buffer::<F>(&shifted, &twiddles_buffer, state)?;
        results.push((result_buffer, poly_domain_size));
    }

    Ok(results)
}

#[cfg(all(test, target_os = "macos", feature = "metal"))]
mod tests {
    use super::*;
    use lambdaworks_math::field::element::FieldElement;
    use lambdaworks_math::field::fields::u64_goldilocks_field::Goldilocks64Field;
    use lambdaworks_math::polynomial::Polynomial;

    type FpE = FieldElement<Goldilocks64Field>;

    #[test]
    fn gpu_fft_interpolation_matches_cpu() {
        let state = crate::metal::state::StarkMetalState::new().unwrap();
        let values: Vec<FpE> = (0..8).map(|i| FpE::from(i as u64 + 1)).collect();

        let cpu_poly = Polynomial::interpolate_fft::<Goldilocks64Field>(&values).unwrap();
        let gpu_coeffs = gpu_interpolate_fft::<Goldilocks64Field>(&values, state.inner()).unwrap();

        assert_eq!(cpu_poly.coefficients().len(), gpu_coeffs.len());
        for (cpu, gpu) in cpu_poly.coefficients().iter().zip(&gpu_coeffs) {
            assert_eq!(cpu, gpu, "FFT coefficient mismatch");
        }
    }

    #[test]
    fn gpu_fft_evaluate_offset_matches_cpu() {
        let state = crate::metal::state::StarkMetalState::new().unwrap();
        let coeffs: Vec<FpE> = (0..8).map(|i| FpE::from(i as u64 + 1)).collect();
        let poly = Polynomial::new(&coeffs);
        let offset = FpE::from(7u64);
        let blowup_factor = 4;

        let cpu_evals = Polynomial::evaluate_offset_fft::<Goldilocks64Field>(
            &poly,
            blowup_factor,
            None,
            &offset,
        )
        .unwrap();

        let gpu_evals = gpu_evaluate_offset_fft::<Goldilocks64Field>(
            &coeffs,
            blowup_factor,
            &offset,
            state.inner(),
        )
        .unwrap();

        assert_eq!(cpu_evals.len(), gpu_evals.len());
        for (cpu, gpu) in cpu_evals.iter().zip(&gpu_evals) {
            assert_eq!(cpu, gpu, "Offset FFT evaluation mismatch");
        }
    }

    #[test]
    fn gpu_fft_roundtrip() {
        let state = crate::metal::state::StarkMetalState::new().unwrap();
        let original_coeffs: Vec<FpE> = (0..16).map(|i| FpE::from(i as u64 + 1)).collect();
        let poly = Polynomial::new(&original_coeffs);

        // Evaluate on roots of unity (blowup=1, no offset => offset=1)
        let evals = Polynomial::evaluate_fft::<Goldilocks64Field>(&poly, 1, None).unwrap();

        // Interpolate back using GPU
        let recovered_coeffs =
            gpu_interpolate_fft::<Goldilocks64Field>(&evals, state.inner()).unwrap();

        assert_eq!(original_coeffs.len(), recovered_coeffs.len());
        for (orig, recov) in original_coeffs.iter().zip(&recovered_coeffs) {
            assert_eq!(orig, recov, "Roundtrip coefficient mismatch");
        }
    }
}
