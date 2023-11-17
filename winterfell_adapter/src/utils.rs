use lambdaworks_math::{field::traits::IsField, traits::ByteConversion};
use stark_platinum_prover::fri::FieldElement;
use winter_math::FieldElement as IsWinterfellFieldElement;

pub fn vec_lambda2winter<
    FE: IsField<BaseType = FE> + IsWinterfellFieldElement + ByteConversion + Unpin,
>(
    input: &[FieldElement<FE>],
) -> Vec<FE> {
    input.iter().map(|&e| *e.value()).collect()
}

pub fn vec_winter2lambda<
    FE: IsField<BaseType = FE> + IsWinterfellFieldElement + ByteConversion + Unpin,
>(
    input: &[FE],
) -> Vec<FieldElement<FE>> {
    input
        .iter()
        .map(|&e| FieldElement::<FE>::const_from_raw(e))
        .collect()
}

pub fn matrix_lambda2winter<
    FE: IsField<BaseType = FE> + IsWinterfellFieldElement + ByteConversion + Unpin,
>(
    input: &[Vec<FieldElement<FE>>],
) -> Vec<Vec<FE>> {
    input.iter().map(|v| vec_lambda2winter(v)).collect()
}

pub fn matrix_winter2lambda<
    FE: IsField<BaseType = FE> + IsWinterfellFieldElement + ByteConversion + Unpin,
>(
    input: &[Vec<FE>],
) -> Vec<Vec<FieldElement<FE>>> {
    input.iter().map(|v| vec_winter2lambda(v)).collect()
}
