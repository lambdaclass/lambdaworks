
use lambdaworks_crypto::fiat_shamir::default_transcript::DefaultTranscript;
use stark_platinum_prover::{
    constraints::{
        boundary::BoundaryConstraints,
        transition::TransitionConstraint,
    }, context::AirContext, frame::Frame, proof::options::ProofOptions, prover::{IsStarkProver, Prover}, trace::TraceTable, traits::AIR, verifier::{IsStarkVerifier, Verifier}
};
use lambdaworks_math::field::{element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField, traits::IsFFTField};
use std::marker::PhantomData;


const STEP_SIZE: usize = 1;
const NUMBER_OF_COLUMNS: usize = 7;

#[derive(Clone)]
struct FlagIsBinary<F: IsFFTField> {
    phantom: PhantomData<F>,
}

impl<F: IsFFTField> FlagIsBinary<F> {
    pub fn new() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

impl<F> TransitionConstraint<F, F> for FlagIsBinary<F>
where
    F: IsFFTField + Send + Sync,
{
    fn degree(&self) -> usize {
        todo!()
    }

    fn constraint_idx(&self) -> usize {
        0
    }

    fn end_exemptions(&self) -> usize {
        todo!()
    }

    fn evaluate(
        &self,
        frame: &Frame<F, F>,
        transition_evaluations: &mut [FieldElement<F>],
        _periodic_values: &[FieldElement<F>],
        _rap_challenges: &[FieldElement<F>],
    ) {
       todo!()
    }
}


#[derive(Clone)]
struct OperationIsCorrect<F: IsFFTField> {
    phantom: PhantomData<F>,
}

impl<F: IsFFTField> OperationIsCorrect<F> {
    pub fn new() -> Self {
        Self {
            phantom: PhantomData,
        }
    }
}

impl<F> TransitionConstraint<F, F> for OperationIsCorrect<F>
where
    F: IsFFTField + Send + Sync,
{
    fn degree(&self) -> usize {
        todo!()
    }

    fn constraint_idx(&self) -> usize {
        todo!()
    }

    fn end_exemptions(&self) -> usize {
        todo!()
    }

    fn evaluate(
        &self,
        _frame: &Frame<F, F>,
        _transition_evaluations: &mut [FieldElement<F>],
        _periodic_values: &[FieldElement<F>],
        _rap_challenges: &[FieldElement<F>],
    ) {
       todo!()
    }
}

pub struct VM0Air<F>
where
    F: IsFFTField,
{
    context: AirContext,
    trace_length: usize,
    constraints: Vec<Box<dyn TransitionConstraint<F, F>>>,
    public_inputs: VM0PubInputs<F>
}


// We won't use this, but the general AIR assumes there will be public inputs
#[derive(Clone, Debug)]
pub struct VM0PubInputs<F> 
where F: IsFFTField {
    dummy:  FieldElement<F>
}

impl<F> VM0PubInputs<F> where F: IsFFTField {
    pub fn default() -> Self {
        Self {
            dummy: FieldElement::<F>::zero()
        }
    }

}

impl<F> AIR for VM0Air<F>
where
    F: IsFFTField + Send + Sync + 'static,
{
    type Field = F;
    type FieldExtension = F;
    // This is not used in this example
    type PublicInputs = VM0PubInputs<F>;

    const STEP_SIZE: usize = 1;

    fn new(
        trace_length: usize,
        // unused
        pub_inputs: &Self::PublicInputs,
        proof_options: &ProofOptions,
    ) -> Self {
        
        let constraints: Vec<Box<dyn TransitionConstraint<F, F>>> =
            vec![
                Box::new(OperationIsCorrect::new()), 
                Box::new(FlagIsBinary::new())
            ];

        let context = AirContext {
            proof_options: proof_options.clone(),
            trace_columns: NUMBER_OF_COLUMNS,
            // These are the relative indexes of rows needed to evaluate the constraints, in this case we need two consecutives rows so 0 and 1 is good.  
            transition_offsets: vec![0, 1],
            num_transition_constraints: constraints.len(),
        };

        Self {
            context,
            trace_length,
            constraints,
            public_inputs: pub_inputs.clone()
        }
    }

    fn composition_poly_degree_bound(&self) -> usize {
        self.trace_length()
    }

    fn transition_constraints(&self) -> &Vec<Box<dyn TransitionConstraint<F, F>>> {
        &self.constraints
    }

    fn context(&self) -> &AirContext {
        &self.context
    }

    fn trace_length(&self) -> usize {
        self.trace_length
    }

    // For the first example
    fn trace_layout(&self) -> (usize, usize) {
        (NUMBER_OF_COLUMNS, 0)
    }

    fn pub_inputs(&self) -> &Self::PublicInputs {
        &self.public_inputs
    }

    // Rap is not used in this example
    fn boundary_constraints(
        &self,
        _rap_challenges: &[FieldElement<Self::FieldExtension>],
    ) -> BoundaryConstraints<Self::FieldExtension> {
        todo!()
    }

    // This function can just call compute_transition_prover for this examples. It will be unified in just one compute_transition in the future.
    fn compute_transition_verifier(
        &self,
        frame: &Frame<Self::FieldExtension, Self::FieldExtension>,
        periodic_values: &[FieldElement<Self::FieldExtension>],
        rap_challenges: &[FieldElement<Self::FieldExtension>],
    ) -> Vec<FieldElement<Self::Field>> {
        self.compute_transition_prover(frame, periodic_values, rap_challenges)
    }
}


// Flag, A0, V0, A1, V1, A DST, V DST
pub fn vm0_example_trace<F: IsFFTField>(
) -> TraceTable<F, F> {

    let row0: Vec<FieldElement<F>> = vec![
        FieldElement::<F>::from(0),
        FieldElement::<F>::from(0),
        FieldElement::<F>::from(3),
        FieldElement::<F>::from(1),
        FieldElement::<F>::from(3),
        FieldElement::<F>::from(2),
        FieldElement::<F>::from(6)
    ];

    let row1: Vec<FieldElement<F>> = vec![
        FieldElement::<F>::from(1),
        FieldElement::<F>::from(4),
        FieldElement::<F>::from(3),
        FieldElement::<F>::from(5),
        FieldElement::<F>::from(3),
        FieldElement::<F>::from(6),
        FieldElement::<F>::from(9)
    ];


    let row2: Vec<FieldElement<F>> = vec![
        FieldElement::<F>::from(0),
        FieldElement::<F>::from(6),
        FieldElement::<F>::from(9),
        FieldElement::<F>::from(2),
        FieldElement::<F>::from(6),
        FieldElement::<F>::from(7),
        FieldElement::<F>::from(15)
    ];

    // Trace length needs to be a power of 2 for FFT to work
    // We pad it with another equal to the last one
    // In a real VM this can be solved by adding a jmp rel 0 for example so the padding is valid, or some extra NOPs
    let row3 = row2.clone();

    let mut trace_data: Vec<FieldElement<F>> = Vec::new();
    trace_data.extend(row0);
    trace_data.extend(row1);
    trace_data.extend(row2);
    trace_data.extend(row3);

    TraceTable::new_main(trace_data, NUMBER_OF_COLUMNS, STEP_SIZE)
}

fn main() {
    let mut trace: TraceTable<Stark252PrimeField, Stark252PrimeField> = vm0_example_trace();

    let row0_string: Vec<String> = trace.get_column_main(0).iter().map(|x| x.to_string()).collect();
    let row1_string: Vec<String> = trace.get_column_main(1).iter().map(|x| x.to_string()).collect();
    let row6_string: Vec<String> = trace.get_column_main(6).iter().map(|x| x.to_string()).collect();

    println!("First row of trace: {:?}", row0_string);
    println!("Second row of trace: {:?}", row1_string);
    println!("...");
    println!("Sixth row row of trace: {:?}", row6_string);

    // This can always be 3
    let coset_offset = 3;
    let proof_options = ProofOptions::new_secure(stark_platinum_prover::proof::options::SecurityLevel::Conjecturable100Bits, coset_offset);

    let pub_inputs = VM0PubInputs::default();

    println!("Generating proof ...");
    let proof_result = Prover::<VM0Air<Stark252PrimeField>>::prove(
        &mut trace,
        &pub_inputs,
        &proof_options,
        DefaultTranscript::default()
    );

    let proof = match proof_result {
        Ok(x)  => x,
        Err(e) => {
            println!("Error while generating the proof");
            return  
        },
    };

    println!("Done!");
    println!("Verifying proof ...");

    assert!(Verifier::<VM0Air<Stark252PrimeField>>::verify(
        &proof,
        &pub_inputs,
        &proof_options,
        DefaultTranscript::default()
    ));
}
