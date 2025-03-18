use lambdaworks_math::field::{element::FieldElement, traits::IsField};

use super::{Constraint, ConstraintSystem, ConstraintType, Variable};

impl<F> ConstraintSystem<F>
where
    F: IsField,
{
    /// Adds a constraint to enforce that `v1` is equal to `v2`.
    pub fn assert_eq(&mut self, v1: &Variable, v2: &Variable) {
        self.add_constraint(Constraint {
            constraint_type: ConstraintType {
                ql: FieldElement::one(),
                qr: -FieldElement::one(),
                qm: FieldElement::zero(),
                qo: FieldElement::zero(),
                qc: FieldElement::zero(),
            },
            l: *v1,
            r: *v2,
            o: self.null_variable(),
            hint: None,
        });
    }

    /// Creates a new variable `w` constrained to be `v1` in case
    /// `boolean_condition` is `1` and `v2` otherwise.
    pub fn if_else(
        &mut self,
        boolean_condition: &Variable,
        v1: &Variable,
        v2: &Variable,
    ) -> Variable {
        let not_boolean_condition = self.not(boolean_condition);
        let if_branch = self.mul(v1, boolean_condition);
        let else_branch = self.mul(v2, &not_boolean_condition);
        self.add(&if_branch, &else_branch)
    }

    /// Creates a new variable `w` constrained to be `v1` in case
    /// `condition` is not zero and `v2` otherwise.
    pub fn if_nonzero_else(
        &mut self,
        condition: &Variable,
        v1: &Variable,
        v2: &Variable,
    ) -> Variable {
        let (is_zero, _) = self.inv(condition);
        self.if_else(&is_zero, v2, v1)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use lambdaworks_math::field::{
        element::FieldElement as FE, fields::u64_prime_field::U64PrimeField,
    };

    use crate::constraint_system::ConstraintSystem;

    #[test]
    fn test_assert_eq_1() {
        let system = &mut ConstraintSystem::<U64PrimeField<65537>>::new();

        let v = system.new_variable();
        let w = system.new_variable();
        let z = system.mul(&v, &w);
        let output = system.new_variable();
        system.assert_eq(&z, &output);

        let inputs = HashMap::from([(v, FE::from(2)), (w, FE::from(2).inv().unwrap())]);

        let assignments = system.solve(inputs).unwrap();
        assert_eq!(assignments.get(&output).unwrap(), &FE::one());
    }

    #[test]
    fn test_assert_eq_2() {
        let system = &mut ConstraintSystem::<U64PrimeField<65537>>::new();

        let v = system.new_variable();
        let w = system.new_variable();
        let z = system.mul(&v, &w);
        let output = system.new_variable();
        system.assert_eq(&z, &output);

        let inputs = HashMap::from([(v, FE::from(2)), (w, FE::from(2)), (output, FE::from(1))]);

        let _assignments = system.solve(inputs).unwrap_err();
    }

    #[test]
    fn test_if_nonzero_else_1() {
        let system = &mut ConstraintSystem::<U64PrimeField<65537>>::new();

        let v = system.new_variable();
        let v2 = system.mul(&v, &v);
        let v4 = system.mul(&v2, &v2);
        let w = system.add_constant(&v4, -FE::one());
        let output = system.if_nonzero_else(&w, &v, &v2);

        let inputs = HashMap::from([(v, FE::from(256))]);

        let assignments = system.solve(inputs).unwrap();
        assert_eq!(
            assignments.get(&output).unwrap(),
            assignments.get(&v2).unwrap()
        );
    }

    #[test]
    fn test_if_nonzero_else_2() {
        let system = &mut ConstraintSystem::<U64PrimeField<65537>>::new();

        let v = system.new_variable();
        let v2 = system.mul(&v, &v);
        let v4 = system.mul(&v2, &v2);
        let w = system.add_constant(&v4, -FE::one());
        let output = system.if_nonzero_else(&w, &v, &v2);

        let inputs = HashMap::from([(v, FE::from(255))]);

        let assignments = system.solve(inputs).unwrap();
        assert_eq!(
            assignments.get(&output).unwrap(),
            assignments.get(&v).unwrap()
        );
    }
}
