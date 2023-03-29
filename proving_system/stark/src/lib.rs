pub mod air;
pub mod cairo_vm;
pub mod fri;
pub mod prover;
pub mod verifier;
use air::frame::Frame;

use fri::fri_decommit::FriDecommitment;

use lambdaworks_crypto::fiat_shamir::transcript::Transcript;
use lambdaworks_math::field::{
    element::FieldElement,
    fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
    traits::{IsField, IsTwoAdicField},
};

pub struct ProofConfig {
    pub count_queries: usize,
    pub blowup_factor: usize,
}

pub type PrimeField = Stark252PrimeField;
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

#[derive(Debug)]
pub struct StarkProof<F: IsTwoAdicField> {
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

    use crate::test_utils::{Fibonacci17AIR, Fibonacci2ColsAIR, FibonacciAIR, QuadraticAIR};

    use crate::{
        air::{
            context::{AirContext, ProofOptions},
            trace::TraceTable,
            AIR,
        },
        prover::prove,
        test_utils::{fibonacci_trace, fibonacci_trace_2_columns, quadratic_trace},
        verifier::verify,
        FE,
    };

    #[test]
    fn test_prove_fib() {
        let trace = fibonacci_trace([FE::from(1), FE::from(1)], 8);
        let trace_length = trace[0].len();
        let trace_table = TraceTable::new_from_cols(&trace);

        let context = AirContext {
            options: ProofOptions {
                blowup_factor: 2,
                fri_number_of_queries: 1,
                coset_offset: 3,
            },
            trace_length,
            trace_columns: trace_table.n_cols,
            transition_degrees: vec![1],
            transition_exemptions: vec![2],
            transition_offsets: vec![0, 1, 2],
            num_transition_constraints: 1,
        };

        let fibonacci_air = FibonacciAIR::new(context);

        let result = prove(&trace_table, &fibonacci_air);
        assert!(verify(&result, &fibonacci_air));
    }

    #[test]
    fn test_prove_fib17() {
        let trace = fibonacci_trace([FE17::from(1), FE17::from(1)], 4);
        let trace_table = TraceTable::new_from_cols(&trace);

        let context = AirContext {
            options: ProofOptions {
                blowup_factor: 2,
                fri_number_of_queries: 1,
                coset_offset: 3,
            },
            trace_length: trace_table.n_rows(),
            trace_columns: trace_table.n_cols,
            transition_degrees: vec![1],
            transition_exemptions: vec![2],
            transition_offsets: vec![0, 1, 2],
            num_transition_constraints: 1,
        };

        let fibonacci_air = Fibonacci17AIR::new(context);

        let result = prove(&trace_table, &fibonacci_air);
        assert!(verify(&result, &fibonacci_air));
    }

    #[test]
    fn test_prove_fib_2_cols() {
        let trace_columns = fibonacci_trace_2_columns([FE::from(1), FE::from(1)], 16);

        let trace_table = TraceTable::new_from_cols(&trace_columns);

        let context = AirContext {
            options: ProofOptions {
                blowup_factor: 2,
                fri_number_of_queries: 1,
                coset_offset: 3,
            },
            trace_length: trace_table.n_rows(),
            transition_degrees: vec![1, 1],
            transition_exemptions: vec![1, 1],
            transition_offsets: vec![0, 1],
            num_transition_constraints: 2,
            trace_columns: 2,
        };

        let fibonacci_air = Fibonacci2ColsAIR::new(context);

        let result = prove(&trace_table, &fibonacci_air);
        assert!(verify(&result, &fibonacci_air));
    }

    #[test]
    fn test_prove_quadratic() {
        let trace = quadratic_trace(FE::from(3), 4);
        let trace_table = TraceTable {
            table: trace.clone(),
            n_cols: 1,
        };

        let context = AirContext {
            options: ProofOptions {
                blowup_factor: 2,
                fri_number_of_queries: 1,
                coset_offset: 3,
            },
            trace_length: trace.len(),
            trace_columns: trace_table.n_cols,
            transition_degrees: vec![2],
            transition_exemptions: vec![1],
            transition_offsets: vec![0, 1],
            num_transition_constraints: 1,
        };

        let fibonacci_air = QuadraticAIR::new(context);

        let result = prove(&trace_table, &fibonacci_air);
        assert!(verify(&result, &fibonacci_air));
    }
}

#[cfg(test)]
mod test_utils {
    use super::*;
    use crate::air::{
        constraints::boundary::{BoundaryConstraint, BoundaryConstraints},
        context::AirContext,
        AIR,
    };
    use lambdaworks_math::field::{element::FieldElement, fields::u64_prime_field::F17};

    pub fn fibonacci_trace<F: IsField>(
        initial_values: [FieldElement<F>; 2],
        trace_length: usize,
    ) -> Vec<Vec<FieldElement<F>>> {
        let mut ret: Vec<FieldElement<F>> = vec![];

        ret.push(initial_values[0].clone());
        ret.push(initial_values[1].clone());

        for i in 2..(trace_length) {
            ret.push(ret[i - 1].clone() + ret[i - 2].clone());
        }

        vec![ret]
    }

    pub fn fibonacci_trace_2_columns<F: IsField>(
        initial_values: [FieldElement<F>; 2],
        trace_length: usize,
    ) -> Vec<Vec<FieldElement<F>>> {
        let mut ret1: Vec<FieldElement<F>> = vec![];
        let mut ret2: Vec<FieldElement<F>> = vec![];

        ret1.push(initial_values[0].clone());
        ret2.push(initial_values[1].clone());

        for i in 1..(trace_length) {
            let new_val = ret1[i - 1].clone() + ret2[i - 1].clone();
            ret1.push(new_val.clone());
            ret2.push(new_val + ret2[i - 1].clone());
        }

        vec![ret1, ret2]
    }

    pub fn quadratic_trace<F: IsField>(
        initial_value: FieldElement<F>,
        trace_length: usize,
    ) -> Vec<FieldElement<F>> {
        let mut ret: Vec<FieldElement<F>> = vec![];

        ret.push(initial_value.clone());

        for i in 1..(trace_length) {
            ret.push(ret[i - 1].clone() * ret[i - 1].clone());
        }

        ret
    }

    #[derive(Clone)]
    pub struct FibonacciAIR {
        context: AirContext,
    }

    impl AIR for FibonacciAIR {
        type Field = PrimeField;

        fn new(context: air::context::AirContext) -> Self {
            Self { context }
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

        fn boundary_constraints(&self) -> BoundaryConstraints<Self::Field> {
            let a0 = BoundaryConstraint::new_simple(0, FieldElement::<Self::Field>::one());
            let a1 = BoundaryConstraint::new_simple(1, FieldElement::<Self::Field>::one());

            BoundaryConstraints::from_constraints(vec![a0, a1])
        }

        fn context(&self) -> air::context::AirContext {
            self.context.clone()
        }
    }

    #[derive(Clone)]
    pub struct Fibonacci17AIR {
        context: AirContext,
    }

    impl AIR for Fibonacci17AIR {
        type Field = F17;

        fn new(context: air::context::AirContext) -> Self {
            Self { context }
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

        fn boundary_constraints(&self) -> BoundaryConstraints<Self::Field> {
            let a0 = BoundaryConstraint::new_simple(0, FieldElement::<Self::Field>::one());
            let a1 = BoundaryConstraint::new_simple(1, FieldElement::<Self::Field>::one());
            let result = BoundaryConstraint::new_simple(3, FieldElement::<Self::Field>::from(3));

            BoundaryConstraints::from_constraints(vec![a0, a1, result])
        }

        fn context(&self) -> air::context::AirContext {
            self.context.clone()
        }
    }

    #[derive(Clone, Debug)]
    pub struct Fibonacci2ColsAIR {
        context: AirContext,
    }

    impl AIR for Fibonacci2ColsAIR {
        type Field = PrimeField;

        fn new(context: air::context::AirContext) -> Self {
            Self { context }
        }

        fn compute_transition(
            &self,
            frame: &air::frame::Frame<Self::Field>,
        ) -> Vec<FieldElement<Self::Field>> {
            let first_row = frame.get_row(0);
            let second_row = frame.get_row(1);

            // constraints of Fibonacci sequence (2 terms per step):
            // s_{0, i+1} = s_{0, i} + s_{1, i}
            // s_{1, i+1} = s_{1, i} + s_{0, i+1}
            let first_transition = &second_row[0] - &first_row[0] - &first_row[1];
            let second_transition = &second_row[1] - &first_row[1] - &second_row[0];

            vec![first_transition, second_transition]
        }

        fn boundary_constraints(&self) -> BoundaryConstraints<Self::Field> {
            let a0 = BoundaryConstraint::new(0, 0, FieldElement::<Self::Field>::one());
            let a1 = BoundaryConstraint::new(1, 0, FieldElement::<Self::Field>::one());

            BoundaryConstraints::from_constraints(vec![a0, a1])
        }

        fn context(&self) -> air::context::AirContext {
            self.context.clone()
        }
    }

    #[derive(Clone)]
    pub struct QuadraticAIR {
        context: AirContext,
    }

    impl AIR for QuadraticAIR {
        type Field = PrimeField;

        fn new(context: air::context::AirContext) -> Self {
            Self { context }
        }

        fn compute_transition(
            &self,
            frame: &air::frame::Frame<Self::Field>,
        ) -> Vec<FieldElement<Self::Field>> {
            let first_row = frame.get_row(0);
            let second_row = frame.get_row(1);

            vec![second_row[0].clone() - first_row[0].clone() * first_row[0].clone()]
        }

        fn boundary_constraints(&self) -> BoundaryConstraints<Self::Field> {
            let a0 = BoundaryConstraint::new_simple(0, FieldElement::<Self::Field>::from(3));
            let result = BoundaryConstraint::new_simple(3, FieldElement::<Self::Field>::from(16));

            BoundaryConstraints::from_constraints(vec![a0, result])
        }

        fn context(&self) -> air::context::AirContext {
            self.context.clone()
        }
    }
}
