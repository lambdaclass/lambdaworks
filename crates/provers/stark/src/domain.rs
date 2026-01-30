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
    /// Creates a new domain for the given AIR.
    ///
    /// # Panics
    ///
    /// Panics if primitive root of unity cannot be found for the trace length,
    /// or if coset generation fails. For fallible construction, use [`new_domain`].
    pub fn new<A>(air: &A) -> Self
    where
        A: AIR<Field = F>,
    {
        // Initial definitions
        let blowup_factor = air.options().blowup_factor as usize;
        let coset_offset = FieldElement::from(air.options().coset_offset);
        let interpolation_domain_size = air.trace_length();
        let root_order = air.trace_length().trailing_zeros();

        // Generate Coset - use expect() with descriptive messages
        let trace_primitive_root = F::get_primitive_root_of_unity(root_order as u64)
            .expect("failed to get primitive root of unity: trace length may exceed field's two-adicity");

        let trace_roots_of_unity = get_powers_of_primitive_root_coset(
            root_order as u64,
            interpolation_domain_size,
            &FieldElement::one(),
        )
        .expect("failed to generate trace roots of unity coset");

        let lde_root_order = (air.trace_length() * blowup_factor).trailing_zeros();
        let lde_roots_of_unity_coset = get_powers_of_primitive_root_coset(
            lde_root_order as u64,
            air.trace_length() * blowup_factor,
            &coset_offset,
        )
        .expect("failed to generate LDE roots of unity coset: blowup factor may be too large");

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
    let interpolation_domain_size = air.trace_length();
    let root_order = air.trace_length().trailing_zeros();
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
