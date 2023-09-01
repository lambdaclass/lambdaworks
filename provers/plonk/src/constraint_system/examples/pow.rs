use lambdaworks_math::field::{element::FieldElement as FE, traits::IsPrimeField};

use crate::constraint_system::{ConstraintSystem, Variable};

/// A square and multiply implementation.
pub fn pow<F: IsPrimeField>(
    system: &mut ConstraintSystem<F>,
    base: Variable,
    exponent: Variable,
) -> Variable {
    let exponent_bits = system.new_u32(&exponent);
    let mut result = system.new_constant(FE::one());

    assert_eq!(exponent_bits.len(), 32);
    for (i, bit) in exponent_bits.iter().enumerate() {
        if i != 0 {
            result = system.mul(&result, &result);
        }
        let result_times_base = system.mul(&result, &base);
        result = system.if_else(bit, &result_times_base, &result);
    }
    result
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;

    use crate::constraint_system::{examples::pow::pow, ConstraintSystem};
    use lambdaworks_math::field::element::FieldElement as FE;

    #[test]
    fn test_pow() {
        let system = &mut ConstraintSystem::<U64PrimeField<65537>>::new();

        let base = system.new_variable();
        let exponent = system.new_variable();
        let result = pow(system, base, exponent);
        let inputs = HashMap::from([(base, FE::from(3)), (exponent, FE::from(10))]);

        let assignments = system.solve(inputs).unwrap();
        assert_eq!(assignments.get(&result).unwrap(), &FE::from(59049));
    }
}
