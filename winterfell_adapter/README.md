# Winterfell adapter
This package helps converting your winterfell AIR into a lambdaworks AIR via an adapter. This `AIRAdapter` can then be used with the lambdaworks prover.

# Examples
## Fibonacci
Suppose you want to run the lambdawors prover with a `WinterfellFibonacciAIR`. 

```rust
use winterfell::Air;

struct WinterfellFibonacciAIR {
    /// ...
}

impl Air for WinterfellFibonacciAIR {
    /// ...
}
```

### Step 1: Convert your winterfell trace table
With the help of the lambdaworks `AirAdapter`, convert your winterfell trace:
```rust
let trace = &AirAdapter::convert_winterfell_trace_table(winterfell_trace)
```

### Step 2: Convert your public inputs
Create the `AirAdapterPublicInputs` by passing yout `winterfell_public_inputs` and the additional parameters required by the lambdaworks prover:

```rust
let pub_inputs = AirAdapterPublicInputs {
    winterfell_public_inputs: AdapterFieldElement(trace.columns()[1][7]),
    transition_degrees: vec![1, 1],    /// The degrees of each transition
    transition_exemptions: vec![1, 1], /// The
    transition_offsets: vec![0, 1],    /// The size of the frame. This is probably [0, 1] for every Winterfell AIR.
    composition_poly_degree_bound: 8,  /// A bound over the composition degree polynomial, used for choosing the number of parts for H(x).
    trace_info: TraceInfo::new(2, 8),  /// Your winterfell trace info.
};
```

Note that you might have to also convert your field elements to `AdapterFieldElement`.

### Step 3: Make the proof 

```rust
let proof = Prover::prove::<AirAdapter<FibonacciAIR, TraceTable<_>>>(
    &trace,
    &pub_inputs, /// Public inputs
    &proof_options,
    StoneProverTranscript::new(&[]),
);
```

To check more examples you can see the `examples` folder inside this crate.
