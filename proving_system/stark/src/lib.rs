pub mod air;
pub mod fri;
pub mod prover;
pub mod verifier;
use air::frame::Frame;

use fri::fri_decommit::FriDecommitment;

use lambdaworks_crypto::fiat_shamir::transcript::Transcript;
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::fields::fft_friendly::u256_two_adic_prime_field::U256MontgomeryTwoAdicPrimeField;
use lambdaworks_math::field::traits::IsField;

pub struct ProofConfig {
    pub count_queries: usize,
    pub blowup_factor: usize,
}

pub type PrimeField = U256MontgomeryTwoAdicPrimeField;
pub type FE = FieldElement<PrimeField>;

// TODO: change this to use more bits
pub fn transcript_to_field<F: IsField>(transcript: &mut Transcript) -> FieldElement<F> {
    let value: u64 = u64::from_be_bytes(transcript.challenge()[..8].try_into().unwrap());
    FieldElement::from(value)
}

pub fn transcript_to_usize(transcript: &mut Transcript) -> usize {
    const CANT_BYTES_USIZE: usize = (usize::BITS / 8) as usize;
    let value = transcript.challenge()[..CANT_BYTES_USIZE]
        .try_into()
        .unwrap();
    usize::from_be_bytes(value)
}

pub fn sample_z_ood<F: IsField>(
    lde_roots_of_unity_coset: &[FieldElement<F>],
    trace_roots_of_unity: &[FieldElement<F>],
    transcript: &mut Transcript,
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

#[derive(Debug, Clone)]
pub struct StarkQueryProof<F: IsField> {
    pub fri_layers_merkle_roots: Vec<FieldElement<F>>,
    pub fri_decommitment: FriDecommitment<F>,
}

pub struct StarkProof<F: IsField> {
    pub fri_layers_merkle_roots: Vec<FieldElement<F>>,
    pub trace_ood_frame_evaluations: Frame<F>,
    pub composition_poly_evaluations: Vec<FieldElement<F>>,
    pub query_list: Vec<StarkQueryProof<F>>,
}

pub use lambdaworks_crypto::merkle_tree::merkle::MerkleTree;
pub use lambdaworks_crypto::merkle_tree::DefaultHasher;

#[cfg(test)]
mod tests {
    use lambdaworks_math::field::fields::u64_prime_field::FE17;

    use crate::test_utils::{Fibonacci17AIR, FibonacciAIR};

    use crate::{
        air::{
            context::{AirContext, ProofOptions},
            trace::TraceTable,
            AIR,
        },
        prover::prove,
        test_utils::fibonacci_trace,
        verifier::verify,
        FE,
    };

    #[test]
    fn test_prove_fib() {
        let trace = fibonacci_trace([FE::from(1), FE::from(1)], 4);

        let context = AirContext {
            options: ProofOptions {
                blowup_factor: 2,
                fri_number_of_queries: 1,
                coset_offset: 3,
            },
            trace_length: trace.len(),
            trace_info: (trace.len(), 1),
            transition_degrees: vec![1],
            transition_exemptions: vec![trace.len() - 2, trace.len() - 1],
            transition_offsets: vec![0, 1, 2],
            num_transition_constraints: 1,
        };

        let trace_table = TraceTable {
            table: trace.clone(),
            num_cols: 1,
        };

        let fibonacci_air = FibonacciAIR::new(trace_table, context);

        let result = prove(&trace, &fibonacci_air);
        assert!(verify(&result, &fibonacci_air));
    }

    #[ignore]
    #[test]
    fn test_prove_fib17() {
        let trace = fibonacci_trace([FE17::new(1), FE17::new(1)], 4);

        let context = AirContext {
            options: ProofOptions {
                blowup_factor: 2,
                fri_number_of_queries: 1,
                coset_offset: 3,
            },
            trace_length: trace.len(),
            trace_info: (trace.len(), 1),
            transition_degrees: vec![1],
            transition_exemptions: vec![trace.len() - 2, trace.len() - 1],
            transition_offsets: vec![0, 1, 2],
            num_transition_constraints: 1,
        };

        let trace_table = TraceTable {
            table: trace.clone(),
            num_cols: 1,
        };

        let fibonacci_air = Fibonacci17AIR::new(trace_table, context);

        let result = prove(&trace, &fibonacci_air);
        assert!(verify(&result, &fibonacci_air));
    }
}

#[cfg(test)]
mod test_utils {
    use super::*;
    use crate::air::{
        constraints::boundary::{BoundaryConstraint, BoundaryConstraints},
        context::AirContext,
        trace::TraceTable,
        AIR,
    };
    use lambdaworks_math::field::fields::u64_prime_field::F17;
    use lambdaworks_math::field::traits::IsTwoAdicField;
    use lambdaworks_math::{field::element::FieldElement, polynomial::Polynomial};

    pub fn fibonacci_trace<F: IsField>(
        initial_values: [FieldElement<F>; 2],
        trace_length: usize,
    ) -> Vec<FieldElement<F>> {
        let mut ret: Vec<FieldElement<F>> = vec![];

        ret.push(initial_values[0].clone());
        ret.push(initial_values[1].clone());

        for i in 2..(trace_length) {
            ret.push(ret[i - 1].clone() + ret[i - 2].clone());
        }

        ret
    }

    #[derive(Clone)]
    pub struct FibonacciAIR {
        context: AirContext,
        pub trace: TraceTable<PrimeField>,
    }

    impl AIR for FibonacciAIR {
        type Field = PrimeField;

        fn new(trace: TraceTable<Self::Field>, context: air::context::AirContext) -> Self {
            Self { context, trace }
        }

        fn compute_transition(
            &self,
            frame: &air::frame::Frame<Self::Field>,
        ) -> Vec<FieldElement<Self::Field>> {
            let first_row = frame.get_row(0);
            let second_row = frame.get_row(1);
            let third_row = frame.get_row(2);

            vec![third_row[0].clone() - second_row[0].clone() - first_row[0].clone()]
        }

        fn compute_boundary_constraints(&self) -> BoundaryConstraints<Self::Field> {
            let a0 = BoundaryConstraint::new_simple(0, FieldElement::<Self::Field>::one());
            let a1 = BoundaryConstraint::new_simple(1, FieldElement::<Self::Field>::one());
            let result = BoundaryConstraint::new_simple(3, FieldElement::<Self::Field>::from(3));

            BoundaryConstraints::from_constraints(vec![a0, a1, result])
        }

        fn transition_divisors(&self) -> Vec<Polynomial<FieldElement<Self::Field>>> {
            let roots_of_unity_order = self.context().trace_length.trailing_zeros();
            let roots_of_unity = Self::Field::get_powers_of_primitive_root_coset(
                u64::try_from(roots_of_unity_order).unwrap(),
                self.context().trace_length,
                &FieldElement::<Self::Field>::one(),
            )
            .unwrap();

            let mut result = vec![];

            for _ in 0..self.context().num_transition_constraints {
                // X^(roots_of_unity_order) - 1
                let roots_of_unity_vanishing_polynomial =
                    Polynomial::new_monomial(
                        FieldElement::<Self::Field>::one(),
                        usize::try_from(roots_of_unity_order).unwrap(),
                    ) - Polynomial::new_monomial(FieldElement::<Self::Field>::one(), 0);

                let mut exemptions_polynomial =
                    Polynomial::new_monomial(FieldElement::<Self::Field>::one(), 0);

                for exemption_index in self.context().transition_exemptions {
                    exemptions_polynomial = exemptions_polynomial
                        * (Polynomial::new_monomial(FieldElement::<Self::Field>::one(), 1)
                            - Polynomial::new_monomial(roots_of_unity[exemption_index].clone(), 0));
                }

                result.push(roots_of_unity_vanishing_polynomial / exemptions_polynomial);
            }

            result
        }

        fn context(&self) -> air::context::AirContext {
            self.context.clone()
        }
    }

    #[derive(Clone)]
    pub struct Fibonacci17AIR {
        context: AirContext,
        pub trace: TraceTable<F17>,
    }

    impl AIR for Fibonacci17AIR {
        type Field = F17;

        fn new(trace: TraceTable<Self::Field>, context: air::context::AirContext) -> Self {
            Self { context, trace }
        }

        fn compute_transition(
            &self,
            frame: &air::frame::Frame<Self::Field>,
        ) -> Vec<FieldElement<Self::Field>> {
            let first_row = frame.get_row(0);
            let second_row = frame.get_row(1);
            let third_row = frame.get_row(2);

            vec![third_row[0] - second_row[0] - first_row[0]]
        }

        fn compute_boundary_constraints(&self) -> BoundaryConstraints<Self::Field> {
            let a0 = BoundaryConstraint::new_simple(0, FieldElement::<Self::Field>::one());
            let a1 = BoundaryConstraint::new_simple(1, FieldElement::<Self::Field>::one());
            let result = BoundaryConstraint::new_simple(3, FieldElement::<Self::Field>::from(3));

            BoundaryConstraints::from_constraints(vec![a0, a1, result])
        }

        fn transition_divisors(&self) -> Vec<Polynomial<FieldElement<Self::Field>>> {
            let roots_of_unity_order = self.context().trace_length.trailing_zeros();
            let roots_of_unity = Self::Field::get_powers_of_primitive_root_coset(
                roots_of_unity_order as u64,
                self.context().trace_length,
                &FieldElement::<Self::Field>::one(),
            )
            .unwrap();

            let mut result = vec![];

            for _ in 0..self.context().num_transition_constraints {
                // X^(roots_of_unity_order) - 1
                let roots_of_unity_vanishing_polynomial =
                    Polynomial::new_monomial(
                        FieldElement::<Self::Field>::one(),
                        roots_of_unity_order as usize,
                    ) - Polynomial::new_monomial(FieldElement::<Self::Field>::one(), 0);

                let mut exemptions_polynomial =
                    Polynomial::new_monomial(FieldElement::<Self::Field>::one(), 0);

                for exemption_index in self.context().transition_exemptions {
                    exemptions_polynomial = exemptions_polynomial
                        * (Polynomial::new_monomial(FieldElement::<Self::Field>::one(), 1)
                            - Polynomial::new_monomial(roots_of_unity[exemption_index], 0));
                }

                result.push(roots_of_unity_vanishing_polynomial / exemptions_polynomial);
            }

            result
        }

        fn context(&self) -> air::context::AirContext {
            self.context.clone()
        }
    }
}
