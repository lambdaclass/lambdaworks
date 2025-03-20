use lambdaworks_math::field::{
    element::FieldElement as FE,
    traits::{IsField, IsPrimeField},
};

use super::{Column, Constraint, ConstraintSystem, ConstraintType, Hint, Variable};

impl<F> ConstraintSystem<F>
where
    F: IsField,
{
    /// Returns a new variable `w` constrained to take the value `value`.
    pub fn new_constant(&mut self, value: FE<F>) -> Variable {
        let constant = self.new_variable();
        self.add_constraint(Constraint {
            constraint_type: ConstraintType {
                ql: -FE::one(),
                qr: FE::zero(),
                qm: FE::zero(),
                qo: FE::zero(),
                qc: value,
            },
            l: constant,
            r: self.null_variable(),
            o: self.null_variable(),
            hint: None,
        });
        constant
    }

    /// Returns a new variable `w` constrained to take either `0` or `1`
    /// values.
    pub fn new_boolean(&mut self) -> Variable {
        let boolean = self.new_variable();
        self.add_constraint(Constraint {
            constraint_type: ConstraintType {
                ql: -FE::one(),
                qr: FE::zero(),
                qm: FE::one(),
                qo: FE::zero(),
                qc: FE::zero(),
            },
            l: boolean,
            r: boolean,
            o: self.null_variable(),
            hint: None,
        });
        boolean
    }

    /// Returns 32 new variables `[b31, b30,..., b1, b0]` constrained to take either
    /// `0` or `1` values and to represent the binary decomposition
    /// of the representative of the value of `v`:
    /// `v = b0 + b1 * 2 + b2 * 2^2 + ... + b31 * 2^31`.
    pub fn new_u32(&mut self, v: &Variable) -> Vec<Variable>
    where
        F: IsPrimeField,
    {
        let bits: Vec<_> = (0..32).map(|_| self.new_boolean()).collect();
        let mut aux_vars: Vec<Variable> = Vec::new();
        let hint_function = |v: &FE<F>| {
            if v.representative() & 1.into() == 1.into() {
                FE::one()
            } else {
                FE::zero()
            }
        };

        let hint = Some(Hint {
            function: hint_function,
            input: Column::O,
            output: Column::R,
        });
        // t1 := 2 b_0 + b_1
        let t_0 = self.linear_combination(
            &bits[0],
            FE::from(2),
            &bits[1],
            FE::one(),
            FE::zero(),
            hint.clone(),
        );
        aux_vars.push(t_0);
        for bit in bits.iter().take(32).skip(2) {
            // t_i := 2 t_{i-1} + b_i
            let t_i = self.linear_combination(
                aux_vars.last().unwrap(),
                FE::from(2),
                bit,
                FE::one(),
                FE::zero(),
                hint.clone(),
            );
            aux_vars.push(t_i);
        }
        self.assert_eq(v, aux_vars.last().unwrap());
        bits
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use lambdaworks_math::field::{element::FieldElement, fields::u64_prime_field::U64PrimeField};

    use crate::constraint_system::ConstraintSystem;

    #[test]
    fn test_constant() {
        let system = &mut ConstraintSystem::<U64PrimeField<65537>>::new();
        let constant = system.new_constant(FieldElement::from(17));
        let inputs = HashMap::new();
        let assignments = system.solve(inputs).unwrap();
        assert_eq!(assignments.get(&constant).unwrap(), &FieldElement::from(17));
    }

    #[test]
    fn test_boolean_1() {
        let system = &mut ConstraintSystem::<U64PrimeField<65537>>::new();

        let boolean = system.new_boolean();
        let inputs = HashMap::from([(boolean, FieldElement::from(2))]);
        // system is inconsistent
        system.solve(inputs).unwrap_err();
    }

    #[test]
    fn test_boolean_2() {
        let system = &mut ConstraintSystem::<U64PrimeField<65537>>::new();

        let boolean = system.new_boolean();
        let inputs = HashMap::from([(boolean, FieldElement::from(1))]);
        // system is solvable
        system.solve(inputs).unwrap();
    }

    #[test]
    fn test_boolean_3() {
        let system = &mut ConstraintSystem::<U64PrimeField<65537>>::new();

        let boolean = system.new_boolean();
        let inputs = HashMap::from([(boolean, FieldElement::from(0))]);
        // system is solvable
        system.solve(inputs).unwrap();
    }

    #[test]
    fn test_u32() {
        let system = &mut ConstraintSystem::<U64PrimeField<65537>>::new();

        let input = system.new_variable();
        let u32_var = system.new_u32(&input);

        let a = 59049;
        let inputs = HashMap::from([(input, FieldElement::from(a))]);

        let assignments = system.solve(inputs).unwrap();

        #[allow(clippy::needless_range_loop)]
        for i in 0..32 {
            assert_eq!(
                assignments.get(&u32_var[i]).unwrap().representative(),
                (a >> (31 - i)) & 1
            );
        }
    }
}
