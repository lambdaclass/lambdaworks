# STARKs lambdaworks implementation

The goal of this section will be to go over the details of the implementation of the proving system. To this end, we will follow the flow the example in the `recap` chapter, diving deeper into the code when necessary and explaining how it fits into a more general case.

The proving system revolves around  `prove` function, that takes a trace and an AIR as inputs to generate a proof, and a `verify` function that takes the proof and the AIR as inputs, outputing `true` when the proof is verified correctly and `false` otherwise.

## AIR
In order to create a proof of the integrity of a computation, we define a trait that must be satisfied to encode the properties of a particular computation.

```rust
pub trait AIR: Clone {
    type Field: IsField;

    fn new(trace: TraceTable<Self::Field>, context: AirContext) -> Self;
    fn boundary_constraints(&self) -> BoundaryConstraints<Self::Field>;
    fn compute_transition(&self, frame: &Frame<Self::Field>) -> Vec<FieldElement<Self::Field>>;
    fn transition_divisors(&self) -> Vec<Polynomial<FieldElement<Self::Field>>>;
    fn context(&self) -> AirContext;
    fn options(&self) -> ProofOptions {
        self.context().options
    }
    fn blowup_factor(&self) -> u8 {
        self.options().blowup_factor
    }
}
```

The most fundamental methods to implement are the `boundary_constraints` and `evaluate_transition`, which encode the boundary and transition constraints of the computation.

### Fibonacci implementation example
Let's start with the `new()` method. The implementation is trivial, but it should receive a `TraceTable` and an `AirContext`. 

* `TraceTable`: A struct representing the execution trace. In essence, a table with *N* rows and *M* columns.
* `AirContext`: A struct that bundles STARK parameters and trace properties for a particular execution of a computation.

    ```rust
    pub struct AirContext {
        pub options: ProofOptions,
        pub trace_length: usize,
        pub trace_colums: usize,
        pub transition_degrees: Vec<usize>,
        pub transition_offsets: Vec<usize>,
        pub transition_exemptions: Vec<usize>,
        pub num_transition_constraints: usize,
    }
    ```
    * **options**: Holds a `ProofOptions` struct. Parameters for the generation of the proof are held here.
    * **trace_length**: Stores the number of steps of the particular computation.



