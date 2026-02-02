use lambdaworks_math::{
    fft::cpu::roots_of_unity::get_powers_of_primitive_root_coset,
    field::{
        element::FieldElement,
        traits::{IsFFTField, IsField, IsSubFieldOf},
    },
};

use super::prover::ProvingError;
use super::traits::AIR;

pub struct Domain<F: IsFFTField> {
    pub(crate) root_order: u32,
    pub(crate) lde_roots_of_unity_coset: Vec<FieldElement<F>>,
    pub(crate) trace_primitive_root: FieldElement<F>,
    pub(crate) trace_roots_of_unity: Vec<FieldElement<F>>,
    pub(crate) coset_offset: FieldElement<F>,
    pub(crate) blowup_factor: usize,
    pub(crate) interpolation_domain_size: usize,
}

impl<F: IsFFTField> Domain<F> {
    pub fn new<A>(air: &A) -> Self
    where
        A: AIR<Field = F>,
    {
        // Initial definitions
        let blowup_factor = air.options().blowup_factor as usize;
        let coset_offset = FieldElement::from(air.options().coset_offset);
        let trace_length = air.trace_length();

        // Validate that trace length is a power of two (required for FFT)
        assert!(
            trace_length > 0 && trace_length.is_power_of_two(),
            "trace_length must be a positive power of two, got {trace_length}"
        );

        let interpolation_domain_size = trace_length;
        let root_order = trace_length.trailing_zeros();
        // * Generate Coset
        let trace_primitive_root = F::get_primitive_root_of_unity(root_order as u64).unwrap();
        let trace_roots_of_unity = get_powers_of_primitive_root_coset(
            root_order as u64,
            interpolation_domain_size,
            &FieldElement::one(),
        )
        .unwrap();

        let lde_root_order = (air.trace_length() * blowup_factor).trailing_zeros();
        let lde_roots_of_unity_coset = get_powers_of_primitive_root_coset(
            lde_root_order as u64,
            air.trace_length() * blowup_factor,
            &coset_offset,
        )
        .unwrap();

        Self {
            root_order,
            lde_roots_of_unity_coset,
            trace_primitive_root,
            trace_roots_of_unity,
            blowup_factor,
            coset_offset,
            interpolation_domain_size,
        }
    }
}

pub fn new_domain<Field, FieldExtension, PI>(
    air: &dyn AIR<Field = Field, FieldExtension = FieldExtension, PublicInputs = PI>,
) -> Result<Domain<Field>, ProvingError>
where
    Field: IsSubFieldOf<FieldExtension> + IsFFTField + Send + Sync,
    FieldExtension: Send + Sync + IsField,
{
    // Initial definitions
    let blowup_factor = air.options().blowup_factor as usize;
    let coset_offset = FieldElement::from(air.options().coset_offset);
    let trace_length = air.trace_length();

    // Validate that trace length is a power of two (required for FFT)
    if trace_length == 0 || !trace_length.is_power_of_two() {
        return Err(ProvingError::InvalidTraceLength(trace_length));
    }

    let interpolation_domain_size = trace_length;
    let root_order = trace_length.trailing_zeros();
    // * Generate Coset
    let trace_primitive_root = Field::get_primitive_root_of_unity(root_order as u64)
        .map_err(|_| ProvingError::PrimitiveRootNotFound(root_order as u64))?;
    let trace_roots_of_unity = get_powers_of_primitive_root_coset(
        root_order as u64,
        interpolation_domain_size,
        &FieldElement::one(),
    )?;

    let lde_root_order = (air.trace_length() * blowup_factor).trailing_zeros();
    let lde_roots_of_unity_coset = get_powers_of_primitive_root_coset(
        lde_root_order as u64,
        air.trace_length() * blowup_factor,
        &coset_offset,
    )?;

    Ok(Domain {
        root_order,
        lde_roots_of_unity_coset,
        trace_primitive_root,
        trace_roots_of_unity,
        blowup_factor,
        coset_offset,
        interpolation_domain_size,
    })
}
