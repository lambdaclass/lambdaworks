pub mod air;
pub mod cairo_run;
pub mod cairo_vm;
pub mod errors;
pub mod fri;
pub mod proof;
pub mod prover;
pub mod verifier;

use air::AIR;
use errors::StarkError;
use lambdaworks_crypto::fiat_shamir::transcript::Transcript;
use lambdaworks_fft::roots_of_unity::get_powers_of_primitive_root_coset;
use lambdaworks_math::field::{
    element::FieldElement,
    fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
    traits::{IsFFTField, IsField},
};

pub struct ProofConfig {
    pub count_queries: usize,
    pub blowup_factor: usize,
}

pub type PrimeField = Stark252PrimeField;
pub type FE = FieldElement<PrimeField>;

// TODO: change this to use more bits
pub fn transcript_to_field<F: IsField, T: Transcript>(transcript: &mut T) -> FieldElement<F> {
    // `transcript.challenge` returns an array of 32 bytes, so it's ok to get its first 8 bytes
    // without checking
    let value: u64 = u64::from_be_bytes(transcript.challenge()[..8].try_into().unwrap());
    FieldElement::from(value)
}

pub fn transcript_to_usize<T: Transcript>(transcript: &mut T) -> usize {
    const NUM_BYTES_USIZE: usize = (usize::BITS / 8) as usize;
    // `transcript.challenge` returns an array of 32 bytes and `usize` size is 8 bytes at most,
    // so it's ok to get its first `NUM_BYTES_USIZE` bytes without checking
    let value = transcript.challenge()[..NUM_BYTES_USIZE]
        .try_into()
        .unwrap();
    usize::from_be_bytes(value)
}

pub fn sample_z_ood<F: IsField, T: Transcript>(
    lde_roots_of_unity_coset: &[FieldElement<F>],
    trace_roots_of_unity: &[FieldElement<F>],
    transcript: &mut T,
) -> FieldElement<F> {
    loop {
        let value: FieldElement<F> = transcript_to_field(transcript);
        if !lde_roots_of_unity_coset.iter().any(|x| x == &value)
            && !trace_roots_of_unity.iter().any(|x| x == &value)
        {
            return value;
        }
    }
}

pub fn batch_sample_challenges<F: IsFFTField, T: Transcript>(
    size: usize,
    transcript: &mut T,
) -> Vec<FieldElement<F>> {
    (0..size).map(|_| transcript_to_field(transcript)).collect()
}

pub struct Domain<F: IsFFTField> {
    root_order: u32,
    lde_roots_of_unity_coset: Vec<FieldElement<F>>,
    lde_root_order: u32,
    trace_primitive_root: FieldElement<F>,
    trace_roots_of_unity: Vec<FieldElement<F>>,
    coset_offset: FieldElement<F>,
    blowup_factor: usize,
    interpolation_domain_size: usize,
}

impl<F: IsFFTField> Domain<F> {
    fn new<A: AIR<Field = F>>(air: &A) -> Result<Self, StarkError> {
        // Initial definitions
        let blowup_factor = air.options().blowup_factor as usize;
        let coset_offset = FieldElement::<F>::from(air.options().coset_offset);
        let interpolation_domain_size = air.context().trace_length;
        let root_order = air.context().trace_length.trailing_zeros();
        // * Generate Coset
        let trace_primitive_root = F::get_primitive_root_of_unity(root_order as u64)?;
        let trace_roots_of_unity = get_powers_of_primitive_root_coset(
            root_order as u64,
            interpolation_domain_size,
            &FieldElement::<F>::one(),
        )?;

        let lde_root_order = (air.context().trace_length * blowup_factor).trailing_zeros();
        let lde_roots_of_unity_coset = get_powers_of_primitive_root_coset(
            lde_root_order as u64,
            air.context().trace_length * blowup_factor,
            &coset_offset,
        )?;

        Ok(Self {
            root_order,
            lde_roots_of_unity_coset,
            lde_root_order,
            trace_primitive_root,
            trace_roots_of_unity,
            blowup_factor,
            coset_offset,
            interpolation_domain_size,
        })
    }
}
