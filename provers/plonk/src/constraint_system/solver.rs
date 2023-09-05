use std::collections::HashMap;

use lambdaworks_math::field::{element::FieldElement as FE, traits::IsField};

use super::{errors::SolverError, Column, Constraint, ConstraintSystem, Variable};

/// Finds a solution to the system extending the `assignments` map. It uses the
/// simple strategy of going through all the constraints trying to determine an
/// unkwown value of a variable in terms of known values. It stops when it goes
/// through every constraint and there's nothing else to be solved this way.
/// It returns an error in case there is no such solution or in case this strategy
/// is not enough.
impl<F> ConstraintSystem<F>
where
    F: IsField,
{
    pub fn solve(
        &self,
        mut assignments: HashMap<Variable, FE<F>>,
    ) -> Result<HashMap<Variable, FE<F>>, SolverError> {
        loop {
            let old_solved = assignments.keys().len();
            for constraint in self.constraints.iter() {
                assignments = solve_hint(assignments, constraint);
                assignments = solve_constraint(assignments, constraint);
            }
            if old_solved == assignments.keys().len() {
                break;
            }
        }

        // Check the system is solved
        for constraint in self.constraints.iter() {
            let a = assignments.get(&constraint.l);
            let b = assignments.get(&constraint.r);
            let c = assignments.get(&constraint.o);

            match (a, b, c) {
                (Some(a), Some(b), Some(c)) => {
                    let ct = &constraint.constraint_type;
                    let result = a * &ct.ql + b * &ct.qr + a * b * &ct.qm + c * &ct.qo + &ct.qc;
                    if result != FE::zero() {
                        return Err(SolverError::InconsistentSystem);
                    }
                }
                _ => return Err(SolverError::UnableToSolve),
            }
        }
        Ok(assignments)
    }
}

fn solve_hint<F: IsField>(
    mut assignments: HashMap<Variable, FE<F>>,
    constraint: &Constraint<F>,
) -> HashMap<Variable, FE<F>> {
    let column_to_variable = |column: &Column| match column {
        Column::L => constraint.l,
        Column::R => constraint.r,
        Column::O => constraint.o,
    };
    if let Some(hint) = &constraint.hint {
        if !assignments.contains_key(&column_to_variable(&hint.output)) {
            if let Some(input) = assignments.get(&column_to_variable(&hint.input)) {
                assignments.insert(column_to_variable(&hint.output), (hint.function)(input));
            }
        }
    }

    assignments
}

fn solve_constraint<F: IsField>(
    mut assignments: HashMap<Variable, FE<F>>,
    constraint: &Constraint<F>,
) -> HashMap<Variable, FE<F>> {
    let ct = &constraint.constraint_type;
    let a = assignments.get(&constraint.l);
    let b = assignments.get(&constraint.r);
    let c = assignments.get(&constraint.o);
    let zero = FE::zero();

    match (
        (a, b, c),
        (ct.ql == zero, ct.qr == zero, ct.qm == zero, ct.qo == zero),
    ) {
        ((Some(a), Some(b), None), _) => {
            if ct.qo != FE::zero() {
                let c = -(a * &ct.ql + b * &ct.qr + a * b * &ct.qm + &ct.qc) * ct.qo.inv().unwrap();
                assignments.insert(constraint.o, c);
            }
        }
        ((Some(a), None, Some(c)), _) => {
            let denominator = &ct.qr + a * &ct.qm;
            if denominator != FE::zero() {
                let b = -(a * &ct.ql + c * &ct.qo + &ct.qc) * denominator.inv().unwrap();
                assignments.insert(constraint.r, b);
            }
        }
        ((None, Some(b), Some(c)), _) => {
            let denominator = &ct.ql + b * &ct.qm;
            if denominator != FE::zero() {
                let a = -(b * &ct.qr + c * &ct.qo + &ct.qc) * denominator.inv().unwrap();
                assignments.insert(constraint.l, a);
            }
        }
        ((Some(a), None, None), _) => {
            let b_coefficient = &ct.qr + a * &ct.qm;
            if b_coefficient == FE::zero() && ct.qo != FE::zero() {
                let c = -(a * &ct.ql + &ct.qc) * ct.qo.inv().unwrap();
                assignments.insert(constraint.o, c);
            } else if b_coefficient != FE::zero() && ct.qo == FE::zero() {
                let b = -(a * &ct.ql + &ct.qc) * b_coefficient.inv().unwrap();
                assignments.insert(constraint.r, b);
            }
        }
        ((None, Some(b), None), _) => {
            let a_coefficient = &ct.ql + b * &ct.qm;
            if a_coefficient == FE::zero() && ct.qo != FE::zero() {
                let c = -(b * &ct.qr + &ct.qc) * ct.qo.inv().unwrap();
                assignments.insert(constraint.o, c);
            } else if a_coefficient != FE::zero() && ct.qo == FE::zero() {
                let a = -(b * &ct.qr + &ct.qc) * a_coefficient.inv().unwrap();
                assignments.insert(constraint.l, a);
            }
        }
        ((None, None, Some(c)), (false, true, true, _)) => {
            let a = -(c * &ct.qo + &ct.qc) * ct.ql.inv().unwrap();
            assignments.insert(constraint.l, a);
        }
        ((None, None, Some(c)), (true, false, true, _)) => {
            let b = -(c * &ct.qo + &ct.qc) * ct.qr.inv().unwrap();
            assignments.insert(constraint.r, b);
        }
        ((None, None, None), (true, true, true, false)) => {
            let c = -&ct.qc * ct.qo.inv().unwrap();
            assignments.insert(constraint.o, c);
        }
        ((None, None, None), (true, false, true, true)) => {
            let b = -&ct.qc * ct.qr.inv().unwrap();
            assignments.insert(constraint.r, b);
        }
        ((None, None, None), (false, true, true, true)) => {
            let a = -&ct.qc * ct.ql.inv().unwrap();
            assignments.insert(constraint.l, a);
        }
        _ => {}
    }
    assignments
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::constraint_system::{
        errors::SolverError, Constraint, ConstraintSystem, ConstraintType,
    };
    use lambdaworks_math::field::{
        element::FieldElement as FE, fields::u64_prime_field::U64PrimeField,
    };

    #[test]
    fn test_case_all_values_are_known() {
        let mut system = ConstraintSystem::<U64PrimeField<65537>>::new();
        let a = system.new_variable();
        let b = system.new_variable();
        let c = system.new_variable();
        let constraint = Constraint {
            constraint_type: ConstraintType {
                ql: FE::one(),
                qr: FE::one(),
                qm: FE::one(),
                qo: -FE::one(),
                qc: FE::one(),
            },
            hint: None,
            l: a,
            r: b,
            o: c,
        };
        system.add_constraint(constraint);
        let inputs = HashMap::from([(a, FE::from(2)), (b, FE::from(3)), (c, FE::from(12))]);
        system.solve(inputs).unwrap();
    }

    #[test]
    fn test_case_b_and_c_are_known() {
        let mut system = ConstraintSystem::<U64PrimeField<65537>>::new();
        let a = system.new_variable();
        let b = system.new_variable();
        let c = system.new_variable();
        let constraint = Constraint {
            constraint_type: ConstraintType {
                ql: FE::one(),
                qr: FE::one(),
                qm: FE::one(),
                qo: -FE::one(),
                qc: FE::one(),
            },
            hint: None,
            l: a,
            r: b,
            o: c,
        };
        system.add_constraint(constraint);
        let inputs = HashMap::from([(b, FE::from(3)), (c, FE::from(12))]);
        let assignments = system.solve(inputs).unwrap();
        assert_eq!(assignments.get(&a).unwrap(), &FE::from(2));
    }

    #[test]
    fn test_case_b_and_c_are_known_but_as_coefficient_is_zero() {
        let mut system = ConstraintSystem::<U64PrimeField<65537>>::new();
        let a = system.new_variable();
        let b = system.new_variable();
        let c = system.new_variable();
        let constraint = Constraint {
            constraint_type: ConstraintType {
                ql: FE::from(3),
                qr: FE::one(),
                qm: -FE::one(),
                qo: -FE::one(),
                qc: FE::one(),
            },
            hint: None,
            l: a,
            r: b,
            o: c,
        };
        system.add_constraint(constraint);
        let inputs = HashMap::from([(b, FE::from(3)), (c, FE::from(12))]);
        assert_eq!(
            system.solve(inputs).unwrap_err(),
            SolverError::UnableToSolve
        );
    }

    #[test]
    fn test_case_a_and_c_are_known() {
        let mut system = ConstraintSystem::<U64PrimeField<65537>>::new();
        let a = system.new_variable();
        let b = system.new_variable();
        let c = system.new_variable();
        let constraint = Constraint {
            constraint_type: ConstraintType {
                ql: FE::one(),
                qr: FE::one(),
                qm: FE::one(),
                qo: -FE::one(),
                qc: FE::one(),
            },
            hint: None,
            l: a,
            r: b,
            o: c,
        };
        system.add_constraint(constraint);
        let inputs = HashMap::from([(a, FE::from(2)), (c, FE::from(12))]);
        let assignments = system.solve(inputs).unwrap();
        assert_eq!(assignments.get(&b).unwrap(), &FE::from(3));
    }

    #[test]
    fn test_case_a_and_c_are_known_but_bs_coefficient_is_zero() {
        let mut system = ConstraintSystem::<U64PrimeField<65537>>::new();
        let a = system.new_variable();
        let b = system.new_variable();
        let c = system.new_variable();
        let constraint = Constraint {
            constraint_type: ConstraintType {
                ql: FE::one(),
                qr: FE::from(2),
                qm: -FE::one(),
                qo: FE::one(),
                qc: FE::one(),
            },
            hint: None,
            l: a,
            r: b,
            o: c,
        };
        system.add_constraint(constraint);
        let inputs = HashMap::from([(a, FE::from(2)), (c, FE::from(12))]);
        assert_eq!(
            system.solve(inputs).unwrap_err(),
            SolverError::UnableToSolve
        );
    }

    #[test]
    fn test_case_a_and_b_are_known() {
        let mut system = ConstraintSystem::<U64PrimeField<65537>>::new();
        let a = system.new_variable();
        let b = system.new_variable();
        let c = system.new_variable();
        let constraint = Constraint {
            constraint_type: ConstraintType {
                ql: FE::one(),
                qr: FE::one(),
                qm: FE::one(),
                qo: -FE::one(),
                qc: FE::one(),
            },
            hint: None,
            l: a,
            r: b,
            o: c,
        };
        system.add_constraint(constraint);
        let inputs = HashMap::from([(a, FE::from(2)), (b, FE::from(3))]);
        let assignments = system.solve(inputs).unwrap();
        assert_eq!(assignments.get(&c).unwrap(), &FE::from(12));
    }

    #[test]
    fn test_case_a_and_b_are_known_but_cs_coefficient_is_zero() {
        let mut system = ConstraintSystem::<U64PrimeField<65537>>::new();
        let a = system.new_variable();
        let b = system.new_variable();
        let c = system.new_variable();
        let constraint = Constraint {
            constraint_type: ConstraintType {
                ql: FE::one(),
                qr: FE::one(),
                qm: FE::one(),
                qo: FE::zero(),
                qc: FE::one(),
            },
            hint: None,
            l: a,
            r: b,
            o: c,
        };
        system.add_constraint(constraint);
        let inputs = HashMap::from([(a, FE::from(2)), (b, FE::from(3))]);
        assert_eq!(
            system.solve(inputs).unwrap_err(),
            SolverError::UnableToSolve
        );
    }

    #[test]
    fn test_case_only_a_is_known_but_bs_coeffient_is_zero_and_cs_coefficient_is_nonzero() {
        let mut system = ConstraintSystem::<U64PrimeField<65537>>::new();
        let a = system.new_variable();
        let b = system.new_variable();
        let c = system.new_variable();
        let constraint1 = Constraint {
            constraint_type: ConstraintType {
                ql: FE::one(),
                qr: -FE::from(2),
                qm: FE::one(),
                qo: FE::one(),
                qc: FE::one(),
            },
            hint: None,
            l: a,
            r: b,
            o: c,
        };
        let constraint2 = Constraint {
            constraint_type: ConstraintType {
                ql: FE::one(),
                qr: FE::one(),
                qm: FE::zero(),
                qo: FE::zero(),
                qc: FE::zero(),
            },
            hint: None,
            l: b,
            r: c,
            o: system.null_variable(),
        };
        system.add_constraint(constraint1);
        system.add_constraint(constraint2);
        let inputs = HashMap::from([(a, FE::from(2))]);
        let assignments = system.solve(inputs).unwrap();
        assert_eq!(assignments.get(&b).unwrap(), &FE::from(3));
        assert_eq!(assignments.get(&c).unwrap(), &-FE::from(3));
    }

    #[test]
    fn test_case_only_a_is_known_but_bs_coefficient_is_nonzero_and_cs_coeffient_is_zero() {
        let mut system = ConstraintSystem::<U64PrimeField<65537>>::new();
        let a = system.new_variable();
        let b = system.new_variable();
        let c = system.new_variable();
        let constraint1 = Constraint {
            constraint_type: ConstraintType {
                ql: FE::one(),
                qr: FE::one(),
                qm: FE::one(),
                qo: FE::zero(),
                qc: -FE::from(5),
            },
            hint: None,
            l: a,
            r: b,
            o: c,
        };
        let constraint2 = Constraint {
            constraint_type: ConstraintType {
                ql: FE::one(),
                qr: FE::one(),
                qm: FE::zero(),
                qo: FE::zero(),
                qc: FE::zero(),
            },
            hint: None,
            l: b,
            r: c,
            o: system.null_variable(),
        };
        system.add_constraint(constraint1);
        system.add_constraint(constraint2);
        let inputs = HashMap::from([(a, FE::from(1))]);
        let assignments = system.solve(inputs).unwrap();
        assert_eq!(assignments.get(&b).unwrap(), &FE::from(2));
        assert_eq!(assignments.get(&c).unwrap(), &-FE::from(2));
    }

    #[test]
    fn test_case_only_a_is_known_but_bs_cofficient_is_zero_and_cs_coeffient_is_zero() {
        let mut system = ConstraintSystem::<U64PrimeField<65537>>::new();
        let a = system.new_variable();
        let b = system.new_variable();
        let c = system.new_variable();
        let constraint1 = Constraint {
            constraint_type: ConstraintType {
                ql: FE::one(),
                qr: FE::one(),
                qm: -FE::one(),
                qo: FE::zero(),
                qc: -FE::from(5),
            },
            hint: None,
            l: a,
            r: b,
            o: c,
        };
        let constraint2 = Constraint {
            constraint_type: ConstraintType {
                ql: FE::one(),
                qr: FE::one(),
                qm: FE::zero(),
                qo: FE::zero(),
                qc: FE::zero(),
            },
            hint: None,
            l: b,
            r: c,
            o: system.null_variable(),
        };
        system.add_constraint(constraint1);
        system.add_constraint(constraint2);
        let inputs = HashMap::from([(a, FE::from(1))]);
        assert_eq!(
            system.solve(inputs).unwrap_err(),
            SolverError::UnableToSolve
        );
    }

    #[test]
    // TODO: This system is actually solvable but not with our current solver logic
    fn test_case_only_a_is_known_but_bs_cofficient_is_nonzero_and_cs_coeffient_is_nonzero() {
        let mut system = ConstraintSystem::<U64PrimeField<65537>>::new();
        let a = system.new_variable();
        let b = system.new_variable();
        let c = system.new_variable();
        let constraint1 = Constraint {
            constraint_type: ConstraintType {
                ql: FE::one(),
                qr: FE::one(),
                qm: FE::one(),
                qo: FE::one(),
                qc: -FE::from(5),
            },
            hint: None,
            l: a,
            r: b,
            o: c,
        };
        let constraint2 = Constraint {
            constraint_type: ConstraintType {
                ql: FE::one(),
                qr: FE::one(),
                qm: FE::zero(),
                qo: FE::zero(),
                qc: FE::zero(),
            },
            hint: None,
            l: b,
            r: c,
            o: system.null_variable(),
        };
        system.add_constraint(constraint1);
        system.add_constraint(constraint2);
        let inputs = HashMap::from([(a, FE::from(1))]);
        assert_eq!(
            system.solve(inputs).unwrap_err(),
            SolverError::UnableToSolve
        );
    }

    #[test]
    fn test_case_only_b_is_known_but_as_coeffient_is_zero_and_cs_coefficient_is_nonzero() {
        let mut system = ConstraintSystem::<U64PrimeField<65537>>::new();
        let a = system.new_variable();
        let b = system.new_variable();
        let c = system.new_variable();
        let constraint1 = Constraint {
            constraint_type: ConstraintType {
                ql: -FE::from(3),
                qr: FE::one(),
                qm: FE::one(),
                qo: FE::one(),
                qc: FE::one(),
            },
            hint: None,
            l: a,
            r: b,
            o: c,
        };
        let constraint2 = Constraint {
            constraint_type: ConstraintType {
                ql: FE::one(),
                qr: FE::one(),
                qm: FE::zero(),
                qo: FE::zero(),
                qc: FE::zero(),
            },
            hint: None,
            l: a,
            r: c,
            o: system.null_variable(),
        };
        system.add_constraint(constraint1);
        system.add_constraint(constraint2);
        let inputs = HashMap::from([(b, FE::from(3))]);
        let assignments = system.solve(inputs).unwrap();
        assert_eq!(assignments.get(&a).unwrap(), &FE::from(4));
        assert_eq!(assignments.get(&c).unwrap(), &-FE::from(4));
    }

    #[test]
    fn test_case_only_b_is_known_but_as_coefficient_is_nonzero_and_cs_coeffient_is_zero() {
        let mut system = ConstraintSystem::<U64PrimeField<65537>>::new();
        let a = system.new_variable();
        let b = system.new_variable();
        let c = system.new_variable();
        let constraint1 = Constraint {
            constraint_type: ConstraintType {
                ql: FE::one(),
                qr: FE::one(),
                qm: FE::one(),
                qo: FE::zero(),
                qc: -FE::from(5),
            },
            hint: None,
            l: a,
            r: b,
            o: system.null_variable(),
        };
        let constraint2 = Constraint {
            constraint_type: ConstraintType {
                ql: FE::one(),
                qr: FE::one(),
                qm: FE::zero(),
                qo: FE::zero(),
                qc: FE::zero(),
            },
            hint: None,
            l: a,
            r: c,
            o: system.null_variable(),
        };
        system.add_constraint(constraint1);
        system.add_constraint(constraint2);
        let inputs = HashMap::from([(b, FE::from(1))]);
        let assignments = system.solve(inputs).unwrap();
        assert_eq!(assignments.get(&a).unwrap(), &FE::from(2));
        assert_eq!(assignments.get(&c).unwrap(), &-FE::from(2));
    }

    #[test]
    fn test_case_only_b_is_known_but_as_coefficient_is_zero_and_cs_coeffient_is_zero() {
        let mut system = ConstraintSystem::<U64PrimeField<65537>>::new();
        let a = system.new_variable();
        let b = system.new_variable();
        let c = system.new_variable();
        let constraint1 = Constraint {
            constraint_type: ConstraintType {
                ql: FE::one(),
                qr: FE::one(),
                qm: -FE::one(),
                qo: FE::zero(),
                qc: -FE::from(5),
            },
            hint: None,
            l: a,
            r: b,
            o: system.null_variable(),
        };
        let constraint2 = Constraint {
            constraint_type: ConstraintType {
                ql: FE::one(),
                qr: FE::one(),
                qm: FE::zero(),
                qo: FE::zero(),
                qc: FE::zero(),
            },
            hint: None,
            l: a,
            r: c,
            o: system.null_variable(),
        };
        system.add_constraint(constraint1);
        system.add_constraint(constraint2);
        let inputs = HashMap::from([(b, FE::from(1))]);
        assert_eq!(
            system.solve(inputs).unwrap_err(),
            SolverError::UnableToSolve
        );
    }

    #[test]
    fn test_case_only_c_is_known_but_bs_coeffient_is_zero_and_qm_is_zero() {
        let mut system = ConstraintSystem::<U64PrimeField<65537>>::new();
        let a = system.new_variable();
        let b = system.new_variable();
        let c = system.new_variable();
        let constraint1 = Constraint {
            constraint_type: ConstraintType {
                ql: FE::from(2),
                qr: FE::zero(),
                qm: FE::zero(),
                qo: FE::one(),
                qc: FE::zero(),
            },
            hint: None,
            l: a,
            r: b,
            o: c,
        };
        let constraint2 = Constraint {
            constraint_type: ConstraintType {
                ql: FE::one(),
                qr: FE::one(),
                qm: FE::zero(),
                qo: FE::zero(),
                qc: FE::zero(),
            },
            hint: None,
            l: a,
            r: b,
            o: system.null_variable(),
        };
        system.add_constraint(constraint1);
        system.add_constraint(constraint2);
        let inputs = HashMap::from([(c, FE::from(2))]);
        let assignments = system.solve(inputs).unwrap();
        assert_eq!(assignments.get(&a).unwrap(), &-FE::from(1));
        assert_eq!(assignments.get(&b).unwrap(), &FE::from(1));
    }

    #[test]
    fn test_case_only_c_is_known_and_bs_coeffient_is_zero_but_qm_is_nonzero() {
        let mut system = ConstraintSystem::<U64PrimeField<65537>>::new();
        let a = system.new_variable();
        let b = system.new_variable();
        let c = system.new_variable();
        let constraint1 = Constraint {
            constraint_type: ConstraintType {
                ql: FE::from(2),
                qr: FE::zero(),
                qm: FE::one(),
                qo: FE::one(),
                qc: FE::zero(),
            },
            hint: None,
            l: a,
            r: b,
            o: c,
        };
        let constraint2 = Constraint {
            constraint_type: ConstraintType {
                ql: FE::one(),
                qr: FE::one(),
                qm: FE::zero(),
                qo: FE::zero(),
                qc: FE::zero(),
            },
            hint: None,
            l: a,
            r: b,
            o: system.null_variable(),
        };
        system.add_constraint(constraint1);
        system.add_constraint(constraint2);
        let inputs = HashMap::from([(c, FE::from(2))]);
        assert_eq!(
            system.solve(inputs).unwrap_err(),
            SolverError::UnableToSolve
        );
    }

    #[test]
    fn test_case_only_c_is_known_but_as_coeffient_is_zero_and_qm_is_zero() {
        let mut system = ConstraintSystem::<U64PrimeField<65537>>::new();
        let a = system.new_variable();
        let b = system.new_variable();
        let c = system.new_variable();
        let constraint1 = Constraint {
            constraint_type: ConstraintType {
                ql: FE::zero(),
                qr: FE::from(2),
                qm: FE::zero(),
                qo: FE::one(),
                qc: FE::zero(),
            },
            hint: None,
            l: a,
            r: b,
            o: c,
        };
        let constraint2 = Constraint {
            constraint_type: ConstraintType {
                ql: FE::one(),
                qr: FE::one(),
                qm: FE::zero(),
                qo: FE::zero(),
                qc: FE::zero(),
            },
            hint: None,
            l: a,
            r: b,
            o: system.null_variable(),
        };
        system.add_constraint(constraint1);
        system.add_constraint(constraint2);
        let inputs = HashMap::from([(c, FE::from(2))]);
        let assignments = system.solve(inputs).unwrap();
        assert_eq!(assignments.get(&a).unwrap(), &FE::from(1));
        assert_eq!(assignments.get(&b).unwrap(), &-FE::from(1));
    }

    #[test]
    fn test_case_only_c_is_known_but_as_coeffient_is_nonzero_and_qm_is_nonzero() {
        let mut system = ConstraintSystem::<U64PrimeField<65537>>::new();
        let a = system.new_variable();
        let b = system.new_variable();
        let c = system.new_variable();
        let constraint1 = Constraint {
            constraint_type: ConstraintType {
                ql: FE::one(),
                qr: FE::from(2),
                qm: FE::one(),
                qo: FE::one(),
                qc: FE::zero(),
            },
            hint: None,
            l: a,
            r: b,
            o: c,
        };
        let constraint2 = Constraint {
            constraint_type: ConstraintType {
                ql: FE::one(),
                qr: FE::one(),
                qm: FE::zero(),
                qo: FE::zero(),
                qc: FE::zero(),
            },
            hint: None,
            l: a,
            r: b,
            o: system.null_variable(),
        };
        system.add_constraint(constraint1);
        system.add_constraint(constraint2);
        let inputs = HashMap::from([(c, FE::from(2))]);
        assert_eq!(
            system.solve(inputs).unwrap_err(),
            SolverError::UnableToSolve
        );
    }

    #[test]
    fn test_case_all_values_are_unknown() {
        let mut system = ConstraintSystem::<U64PrimeField<65537>>::new();
        let a = system.new_variable();
        let b = system.new_variable();
        let c = system.new_variable();
        let constraint1 = Constraint {
            constraint_type: ConstraintType {
                ql: FE::from(2),
                qr: FE::zero(),
                qm: FE::zero(),
                qo: FE::zero(),
                qc: -FE::from(2),
            },
            hint: None,
            l: a,
            r: b,
            o: c,
        };
        let constraint2 = Constraint {
            constraint_type: ConstraintType {
                ql: FE::zero(),
                qr: FE::from(2),
                qm: FE::zero(),
                qo: FE::zero(),
                qc: -FE::from(4),
            },
            hint: None,
            l: a,
            r: b,
            o: c,
        };
        let constraint3 = Constraint {
            constraint_type: ConstraintType {
                ql: FE::zero(),
                qr: FE::zero(),
                qm: FE::zero(),
                qo: FE::from(2),
                qc: -FE::from(6),
            },
            hint: None,
            l: a,
            r: b,
            o: c,
        };
        system.add_constraint(constraint1);
        system.add_constraint(constraint2);
        system.add_constraint(constraint3);
        let inputs = HashMap::from([]);
        let assignments = system.solve(inputs).unwrap();
        assert_eq!(assignments.get(&a).unwrap(), &FE::from(1));
        assert_eq!(assignments.get(&b).unwrap(), &FE::from(2));
        assert_eq!(assignments.get(&c).unwrap(), &FE::from(3));
    }

    #[test]
    fn test_inconsistent_system() {
        let mut system = ConstraintSystem::<U64PrimeField<65537>>::new();
        let a = system.new_variable();
        let b = system.new_variable();
        let constraint1 = Constraint {
            constraint_type: ConstraintType {
                ql: FE::one(),
                qr: FE::one(),
                qm: FE::zero(),
                qo: FE::zero(),
                qc: FE::one(),
            },
            hint: None,
            l: a,
            r: b,
            o: system.null_variable(),
        };
        system.add_constraint(constraint1);
        let constraint2 = Constraint {
            constraint_type: ConstraintType {
                ql: -FE::one(),
                qr: FE::one(),
                qm: FE::zero(),
                qo: FE::zero(),
                qc: FE::one(),
            },
            hint: None,
            l: a,
            r: b,
            o: system.null_variable(),
        };
        let inputs = HashMap::from([(a, FE::from(2))]);
        system.add_constraint(constraint2);
        assert_eq!(
            system.solve(inputs).unwrap_err(),
            SolverError::InconsistentSystem
        );
    }

    #[test]
    fn test_indeterminate_system() {
        let mut system = ConstraintSystem::<U64PrimeField<65537>>::new();
        let a = system.new_variable();
        let b = system.new_variable();
        let constraint = Constraint {
            constraint_type: ConstraintType {
                ql: FE::one(),
                qr: FE::one(),
                qm: FE::zero(),
                qo: FE::zero(),
                qc: FE::one(),
            },
            hint: None,
            l: a,
            r: b,
            o: system.null_variable(),
        };
        let inputs = HashMap::from([]);
        system.add_constraint(constraint);
        assert_eq!(
            system.solve(inputs).unwrap_err(),
            SolverError::UnableToSolve
        );
    }
}
