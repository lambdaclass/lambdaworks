use lambdaworks_math::traits::ByteConversion;
use miden_core::{Felt, ProgramInfo, StackOutputs};
use miden_processor::{AuxTraceHints, ExecutionTrace, TraceLenSummary};
use stark_platinum_prover::{
    fri::FieldElement, prover::IsStarkProver, transcript::IsStarkTranscript,
    verifier::IsStarkVerifier,
};
use winter_air::TraceLayout;
use winter_math::StarkField;
use winter_prover::ColMatrix;

use crate::adapter::air::FromColumns;
use sha3::{Digest, Keccak256};

pub struct MidenProver;
impl IsStarkProver for MidenProver {
    type Field = Felt;
}

pub struct MidenVerifier {}
impl IsStarkVerifier for MidenVerifier {
    type Field = Felt;
}

#[derive(Clone)]
pub struct ExecutionTraceMetadata {
    meta: Vec<u8>,
    layout: TraceLayout,
    aux_trace_hints: AuxTraceHints,
    program_info: ProgramInfo,
    stack_outputs: StackOutputs,
    trace_len_summary: TraceLenSummary,
}

impl From<ExecutionTrace> for ExecutionTraceMetadata {
    fn from(value: ExecutionTrace) -> Self {
        Self {
            meta: value.meta,
            layout: value.layout,
            aux_trace_hints: value.aux_trace_hints,
            program_info: value.program_info,
            stack_outputs: value.stack_outputs,
            trace_len_summary: value.trace_len_summary,
        }
    }
}

impl FromColumns<Felt, ExecutionTraceMetadata> for ExecutionTrace {
    fn from_cols(columns: Vec<Vec<Felt>>, metadata: &ExecutionTraceMetadata) -> Self {
        ExecutionTrace {
            meta: metadata.meta.clone(),
            layout: metadata.layout.clone(),
            main_trace: ColMatrix::new(columns),
            aux_trace_hints: metadata.aux_trace_hints.clone(),
            program_info: metadata.program_info.clone(),
            stack_outputs: metadata.stack_outputs.clone(),
            trace_len_summary: metadata.trace_len_summary,
        }
    }
}

pub struct MidenProverTranscript {
    hasher: Keccak256,
}

impl MidenProverTranscript {
    pub fn new(data: &[u8]) -> Self {
        let mut res = Self {
            hasher: Keccak256::new(),
        };
        res.append_bytes(data);
        res
    }
}

impl IsStarkTranscript<Felt> for MidenProverTranscript {
    fn append_field_element(&mut self, element: &FieldElement<Felt>) {
        self.append_bytes(&element.value().to_bytes_be());
    }

    fn append_bytes(&mut self, new_bytes: &[u8]) {
        self.hasher.update(&mut new_bytes.to_owned());
    }

    fn state(&self) -> [u8; 32] {
        self.hasher.clone().finalize().into()
    }

    fn sample_field_element(&mut self) -> FieldElement<Felt> {
        let mut bytes = self.state()[..8].try_into().unwrap();
        let mut x = u64::from_be_bytes(bytes);
        while x >= Felt::MODULUS {
            self.append_bytes(&bytes);
            bytes = self.state()[..8].try_into().unwrap();
            x = u64::from_be_bytes(bytes);
        }
        FieldElement::const_from_raw(Felt::new(x))
    }

    fn sample_u64(&mut self, upper_bound: u64) -> u64 {
        u64::from_be_bytes(self.state()[..8].try_into().unwrap()) % upper_bound
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adapter::air::AirAdapter;
    use crate::adapter::public_inputs::AirAdapterPublicInputs;
    use crate::examples::cubic::{self, Cubic};
    use crate::examples::fibonacci_2_terms::{self, FibAir2Terms};
    use crate::examples::fibonacci_rap::{self, FibonacciRAP, RapTraceTable};
    use crate::examples::miden_vm::MidenProver;
    use miden_air::{ProcessorAir, ProvingOptions, PublicInputs};
    use miden_assembly::Assembler;
    use miden_core::{Felt, StackInputs};
    use miden_processor::DefaultHost;
    use miden_processor::{self as processor};
    use processor::ExecutionTrace;
    use stark_platinum_prover::{
        proof::options::ProofOptions, prover::IsStarkProver, verifier::IsStarkVerifier,
    };
    use winter_air::{TraceInfo, TraceLayout};
    use winter_math::FieldElement;
    use winter_prover::{Trace, TraceTable};

    #[test]
    fn prove_and_verify_miden_readme_example() {
        let mut lambda_proof_options = ProofOptions::default_test_options();
        lambda_proof_options.blowup_factor = 32;
        let assembler = Assembler::default();

        let program = assembler.compile("begin push.3 push.5 add end").unwrap();

        let winter_trace = processor::execute(
            &program,
            StackInputs::default(),
            DefaultHost::default(),
            *ProvingOptions::default().execution_options(),
        )
        .unwrap();
        let program_info = winter_trace.program_info().clone();
        let stack_outputs = winter_trace.stack_outputs().clone();

        let pub_inputs = PublicInputs::new(program_info, StackInputs::default(), stack_outputs);

        let pub_inputs = AirAdapterPublicInputs {
            winterfell_public_inputs: pub_inputs,
            transition_exemptions: vec![2; 182],
            transition_offsets: vec![0, 1],
            trace_info: winter_trace.get_info(),
            metadata: winter_trace.clone().into(),
        };

        let trace =
            AirAdapter::<FibAir2Terms, ExecutionTrace, Felt, _>::convert_winterfell_trace_table(
                winter_trace.main_segment().clone(),
            );

        let proof = MidenProver::prove::<AirAdapter<ProcessorAir, ExecutionTrace, Felt, _>>(
            &trace,
            &pub_inputs,
            &lambda_proof_options,
            MidenProverTranscript::new(&[]),
        )
        .unwrap();

        assert!(MidenVerifier::verify::<
            AirAdapter<ProcessorAir, ExecutionTrace, Felt, _>,
        >(
            &proof,
            &pub_inputs,
            &lambda_proof_options,
            MidenProverTranscript::new(&[]),
        ));
    }

    fn compute_fibonacci(n: usize) -> Felt {
        let mut t0 = Felt::ZERO;
        let mut t1 = Felt::ONE;

        for _ in 0..n {
            t1 = t0 + t1;
            core::mem::swap(&mut t0, &mut t1);
        }
        t0
    }

    #[test]
    fn prove_and_verify_miden_fibonacci() {
        let fibonacci_number = 16;
        let program = format!(
            "begin
                repeat.{}
                    swap dup.1 add
                end
            end",
            fibonacci_number - 1
        );
        let program = Assembler::default().compile(program).unwrap();
        let expected_result = vec![compute_fibonacci(fibonacci_number).as_int()];
        let stack_inputs = StackInputs::try_from_values([0, 1]).unwrap();

        let mut lambda_proof_options = ProofOptions::default_test_options();
        lambda_proof_options.blowup_factor = 8;

        let winter_trace = processor::execute(
            &program,
            stack_inputs.clone(),
            DefaultHost::default(),
            *ProvingOptions::default().execution_options(),
        )
        .unwrap();
        let program_info = winter_trace.program_info().clone();
        let stack_outputs = winter_trace.stack_outputs().clone();

        let pub_inputs = PublicInputs::new(program_info, stack_inputs, stack_outputs.clone());

        assert_eq!(
            expected_result,
            stack_outputs.clone().stack_truncated(1),
            "Program result was computed incorrectly"
        );

        let pub_inputs = AirAdapterPublicInputs {
            winterfell_public_inputs: pub_inputs,
            transition_exemptions: vec![2; 182],
            transition_offsets: vec![0, 1],
            trace_info: winter_trace.get_info(),
            metadata: winter_trace.clone().into(),
        };

        let trace =
            AirAdapter::<ProcessorAir, ExecutionTrace, Felt, _>::convert_winterfell_trace_table(
                winter_trace.main_segment().clone(),
            );

        let proof = MidenProver::prove::<AirAdapter<ProcessorAir, ExecutionTrace, Felt, _>>(
            &trace,
            &pub_inputs,
            &lambda_proof_options,
            MidenProverTranscript::new(&[]),
        )
        .unwrap();

        assert!(MidenVerifier::verify::<
            AirAdapter<ProcessorAir, ExecutionTrace, Felt, _>,
        >(
            &proof,
            &pub_inputs,
            &lambda_proof_options,
            MidenProverTranscript::new(&[]),
        ));
    }

    #[test]
    fn prove_and_verify_a_winterfell_fibonacci_2_terms_air() {
        let lambda_proof_options = ProofOptions::default_test_options();
        let winter_trace = fibonacci_2_terms::build_trace(16);
        let trace =
            AirAdapter::<FibAir2Terms, TraceTable<_>, Felt, ()>::convert_winterfell_trace_table(
                winter_trace.main_segment().clone(),
            );
        let pub_inputs = AirAdapterPublicInputs {
            winterfell_public_inputs: *trace.columns()[1][7].value(),
            transition_exemptions: vec![1, 1],
            transition_offsets: vec![0, 1],
            trace_info: TraceInfo::new(2, 8),
            metadata: (),
        };

        let proof = MidenProver::prove::<AirAdapter<FibAir2Terms, TraceTable<_>, Felt, _>>(
            &trace,
            &pub_inputs,
            &lambda_proof_options,
            MidenProverTranscript::new(&[]),
        )
        .unwrap();

        assert!(MidenVerifier::verify::<
            AirAdapter<FibAir2Terms, TraceTable<_>, Felt, _>,
        >(
            &proof,
            &pub_inputs,
            &lambda_proof_options,
            MidenProverTranscript::new(&[]),
        ));
    }

    #[test]
    fn prove_and_verify_a_winterfell_fibonacci_rap_air() {
        let lambda_proof_options = ProofOptions::default_test_options();
        let winter_trace = fibonacci_rap::build_trace(16);
        let trace =
            AirAdapter::<FibonacciRAP, RapTraceTable<_>, Felt, ()>::convert_winterfell_trace_table(
                winter_trace.main_segment().clone(),
            );
        let trace_layout = TraceLayout::new(3, [1], [1]);
        let trace_info = TraceInfo::new_multi_segment(trace_layout, 16, vec![]);
        let fibonacci_result = trace.columns()[1][15];
        let pub_inputs = AirAdapterPublicInputs::<FibonacciRAP, _> {
            winterfell_public_inputs: *fibonacci_result.value(),
            transition_exemptions: vec![1, 1, 1],
            transition_offsets: vec![0, 1],
            trace_info,
            metadata: (),
        };

        let proof = MidenProver::prove::<AirAdapter<FibonacciRAP, RapTraceTable<_>, Felt, _>>(
            &trace,
            &pub_inputs,
            &lambda_proof_options,
            MidenProverTranscript::new(&[]),
        )
        .unwrap();
        assert!(MidenVerifier::verify::<
            AirAdapter<FibonacciRAP, RapTraceTable<_>, Felt, _>,
        >(
            &proof,
            &pub_inputs,
            &lambda_proof_options,
            MidenProverTranscript::new(&[]),
        ));
    }

    #[test]
    fn prove_and_verify_a_winterfell_cubic_air() {
        let lambda_proof_options = ProofOptions::default_test_options();
        let winter_trace = cubic::build_trace(16);
        let trace = AirAdapter::<Cubic, TraceTable<_>, Felt, ()>::convert_winterfell_trace_table(
            winter_trace.main_segment().clone(),
        );
        let pub_inputs = AirAdapterPublicInputs {
            winterfell_public_inputs: *trace.columns()[0][15].value(),
            transition_exemptions: vec![1],
            transition_offsets: vec![0, 1],
            trace_info: TraceInfo::new(1, 16),
            metadata: (),
        };

        let proof = MidenProver::prove::<AirAdapter<Cubic, TraceTable<_>, Felt, _>>(
            &trace,
            &pub_inputs,
            &lambda_proof_options,
            MidenProverTranscript::new(&[]),
        )
        .unwrap();
        assert!(MidenVerifier::verify::<
            AirAdapter<Cubic, TraceTable<_>, Felt, _>,
        >(
            &proof,
            &pub_inputs,
            &lambda_proof_options,
            MidenProverTranscript::new(&[]),
        ));
    }
}
