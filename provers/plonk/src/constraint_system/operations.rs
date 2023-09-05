use lambdaworks_math::field::{element::FieldElement as FE, traits::IsField};

use super::{Column, Constraint, ConstraintSystem, ConstraintType, Hint, Variable};

impl<F> ConstraintSystem<F>
where
    F: IsField,
{
    /// Creates a new variable `w` constrained to be equal to `c1 * v1 + c2 * v2 + b`.
    /// Optionally a hint can be provided to insert values in `v1`, `v2` or `w`. To do
    /// so use the `L`, `R`, and `O` input/output columns of the hint to refer to `v1`,
    /// `v2` and `w` respectively.
    pub fn linear_combination(
        &mut self,
        v1: &Variable,
        c1: FE<F>,
        v2: &Variable,
        c2: FE<F>,
        b: FE<F>,
        hint: Option<Hint<F>>,
    ) -> Variable {
        let result = self.new_variable();

        self.add_constraint(Constraint {
            constraint_type: ConstraintType {
                ql: c1,
                qr: c2,
                qm: FE::zero(),
                qo: -FE::one(),
                qc: b,
            },
            l: *v1,
            r: *v2,
            o: result,
            hint,
        });
        result
    }

    /// Creates a new variable `w` constrained to be equal to `c * v + b`.
    /// Optionally a hint can be provided to insert values in `v1`, `v2` or `w`. To do
    /// so use the `L`, `R`, and `O` input/output columns of the hint to refer to `v1`,
    /// `v2` and `w` respectively.
    pub fn linear_function(
        &mut self,
        v: &Variable,
        c: FE<F>,
        b: FE<F>,
        hint: Option<Hint<F>>,
    ) -> Variable {
        let result = self.new_variable();
        self.add_constraint(Constraint {
            constraint_type: ConstraintType {
                ql: c,
                qr: FE::zero(),
                qm: FE::zero(),
                qo: -FE::one(),
                qc: b,
            },
            l: *v,
            r: self.null_variable(),
            o: result,
            hint,
        });
        result
    }

    /// Creates a new variable `w` constrained to be equal to `v1 + v2`.
    pub fn add(&mut self, v1: &Variable, v2: &Variable) -> Variable {
        self.linear_combination(v1, FE::one(), v2, FE::one(), FE::zero(), None)
    }

    /// Creates a new variable `w` constrained to be equal to `v1 + constant`.
    pub fn add_constant(&mut self, v: &Variable, constant: FE<F>) -> Variable {
        self.linear_function(v, FE::one(), constant, None)
    }

    /// Creates a new variable `w` constrained to be equal to `v1 * v2`.
    pub fn mul(&mut self, v1: &Variable, v2: &Variable) -> Variable {
        let result = self.new_variable();
        self.add_constraint(Constraint {
            constraint_type: ConstraintType {
                ql: FE::zero(),
                qr: FE::zero(),
                qm: FE::one(),
                qo: -FE::one(),
                qc: FE::zero(),
            },
            l: *v1,
            r: *v2,
            o: result,
            hint: None,
        });
        result
    }

    /// Creates a new variable `w` constrained to be equal to `v1 / v2`.
    pub fn div(&mut self, v1: &Variable, v2: &Variable) -> Variable {
        // TODO: check 0.div(0) does not compile
        let result = self.new_variable();
        self.add_constraint(Constraint {
            constraint_type: ConstraintType {
                ql: FE::zero(),
                qr: FE::zero(),
                qm: FE::one(),
                qo: -FE::one(),
                qc: FE::zero(),
            },
            l: result,
            r: *v2,
            o: *v1,
            hint: None,
        });
        result
    }

    /// Creates two new variables `is_zero` and `v_inverse`. The former is constrained
    /// to be a boolean value holding `1` if `v` is zero and `0` otherwise. The latter
    /// is constrained to be `v^{-1}` when `v` is not zero and equal to `0` otherwise.
    pub fn inv(&mut self, v: &Variable) -> (Variable, Variable) {
        let is_zero = self.new_variable();
        let v_inverse = self.new_variable();
        let hint = Some(Hint {
            function: |v: &FE<F>| {
                if *v == FE::zero() {
                    FE::one()
                } else {
                    FE::zero()
                }
            },
            input: Column::L,
            output: Column::R,
        });
        // v * z == 0
        self.add_constraint(Constraint {
            constraint_type: ConstraintType {
                ql: FE::zero(),
                qr: FE::zero(),
                qm: FE::one(),
                qo: FE::zero(),
                qc: FE::zero(),
            },
            l: *v,
            r: is_zero,
            o: self.null_variable(),
            hint,
        });
        // v * w + z == 1
        self.add_constraint(Constraint {
            constraint_type: ConstraintType {
                ql: FE::zero(),
                qr: FE::zero(),
                qm: FE::one(),
                qo: FE::one(),
                qc: -FE::one(),
            },
            l: *v,
            r: v_inverse, // w
            o: is_zero,   // z
            hint: Some(Hint {
                function: |v: &FE<F>| {
                    if *v == FE::zero() {
                        FE::zero()
                    } else {
                        v.inv().unwrap()
                    }
                },
                input: Column::L,
                output: Column::R,
            }),
        });
        (is_zero, v_inverse)
    }

    /// Returns a new variable `w` constrained to satisfy `w = 1 - v`. When `v` is boolean
    /// this is the `not` operator.
    pub fn not(&mut self, v: &Variable) -> Variable {
        let result = self.new_variable();
        self.add_constraint(Constraint {
            constraint_type: ConstraintType {
                ql: FE::one(),
                qr: FE::one(),
                qm: FE::zero(),
                qo: FE::zero(),
                qc: -FE::one(),
            },
            l: *v,
            r: result,
            o: self.null_variable(),
            hint: None,
        });
        result
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use lambdaworks_math::field::{
        element::FieldElement as FE, fields::u64_prime_field::U64PrimeField,
    };

    #[test]
    fn test_linear_combination() {
        let system = &mut ConstraintSystem::<U64PrimeField<65537>>::new();

        let v1 = system.new_variable();
        let c1 = FE::from(15);
        let v2 = system.new_variable();
        let c2 = -FE::from(7);
        let b = FE::from(99);
        let result = system.linear_combination(&v1, c1, &v2, c2, b, None);

        let x = FE::from(17);
        let y = FE::from(29);

        let inputs = HashMap::from([(v1, x), (v2, y)]);

        let assignments = system.solve(inputs).unwrap();
        assert_eq!(assignments.get(&result).unwrap(), &(x * c1 + y * c2 + b));
    }

    #[test]
    fn test_linear_function() {
        let system = &mut ConstraintSystem::<U64PrimeField<65537>>::new();

        let v = system.new_variable();
        let c = FE::from(8);
        let b = FE::from(109);
        let result = system.linear_function(&v, c, b, None);

        let x = FE::from(17);

        let inputs = HashMap::from([(v, x)]);

        let assignments = system.solve(inputs).unwrap();
        assert_eq!(assignments.get(&result).unwrap(), &(x * c + b));
    }

    #[test]
    fn test_add() {
        let system = &mut ConstraintSystem::<U64PrimeField<65537>>::new();

        let input1 = system.new_variable();
        let input2 = system.new_variable();
        let result = system.add(&input1, &input2);

        let a = FE::from(3);
        let b = FE::from(10);

        let inputs = HashMap::from([(input1, a), (input2, b)]);

        let assignments = system.solve(inputs).unwrap();
        assert_eq!(assignments.get(&result).unwrap(), &(a + b));
    }

    #[test]
    fn test_mul() {
        let system = &mut ConstraintSystem::<U64PrimeField<65537>>::new();

        let input1 = system.new_variable();
        let input2 = system.new_variable();
        let result = system.mul(&input1, &input2);

        let a = FE::from(3);
        let b = FE::from(11);

        let inputs = HashMap::from([(input1, a), (input2, b)]);

        let assignments = system.solve(inputs).unwrap();
        assert_eq!(assignments.get(&result).unwrap(), &(a * b));
    }

    #[test]
    fn test_div() {
        let system = &mut ConstraintSystem::<U64PrimeField<65537>>::new();

        let input1 = system.new_variable();
        let input2 = system.new_variable();
        let result = system.div(&input1, &input2);

        let a = FE::from(3);
        let b = FE::from(11);

        let inputs = HashMap::from([(input1, a), (input2, b)]);

        let assignments = system.solve(inputs).unwrap();
        assert_eq!(assignments.get(&result).unwrap(), &(a / b));
    }

    #[test]
    fn test_add_constant() {
        let system = &mut ConstraintSystem::<U64PrimeField<65537>>::new();

        let input1 = system.new_variable();
        let b = FE::from(11);
        let result = system.add_constant(&input1, b);

        let a = FE::from(3);

        let inputs = HashMap::from([(input1, a)]);

        let assignments = system.solve(inputs).unwrap();
        assert_eq!(assignments.get(&result).unwrap(), &(a + b));
    }

    #[test]
    fn test_not() {
        let system = &mut ConstraintSystem::<U64PrimeField<65537>>::new();

        let boolean = system.new_boolean();
        let result1 = system.not(&boolean);
        let result2 = system.not(&result1);

        let inputs = HashMap::from([(boolean, FE::one())]);

        let assignments = system.solve(inputs).unwrap();
        assert_eq!(assignments.get(&result1).unwrap(), &FE::zero());
        assert_eq!(assignments.get(&result2).unwrap(), &FE::one());
    }

    #[test]
    fn test_inv() {
        let system = &mut ConstraintSystem::<U64PrimeField<65537>>::new();

        let v = system.new_variable();
        let w = system.new_variable();
        let (v_is_zero, v_inverse) = system.inv(&v);
        let (w_is_zero, w_inverse) = system.inv(&w);

        let inputs = HashMap::from([(v, FE::from(2)), (w, FE::from(0))]);

        let assignments = system.solve(inputs).unwrap();
        assert_eq!(
            assignments.get(&v_inverse).unwrap(),
            &FE::from(2).inv().unwrap()
        );
        assert_eq!(assignments.get(&v_is_zero).unwrap(), &FE::zero());

        assert_eq!(assignments.get(&w_inverse).unwrap(), &FE::from(0));
        assert_eq!(assignments.get(&w_is_zero).unwrap(), &FE::one());
    }
}
