# Winterfell adapter
This crate helps running the lambdaworks prover with AIR's written for winterfell. In both libraries the AIR's are created by implementing a trait to a structuture. This package helps converting your winterfell AIR into a lambdaworks AIR via an adapter. This AIRAdapter can then be used with the lambdaworks prover.

# Examples
## Fibonacci
Suppose you have a winterfell AIR that checks valid Fibonacci computations called `FibonacciAIR`.

```rust
struct FibonacciAIR {
    /// ...
}

impl Air for FibonacciAIR {
    /// ...
}
```

You already have a lambdaworks AIR by combining the winterfell AIR and the winterfell `TraceTable` using the type `AirAdapter<FibonacciAIR, TraceTable>`. Note that different trace tables can be used in case you have a custom RAP. 

```rust
let proof = Prover::prove::<AirAdapter<FibonacciAIR, TraceTable<_>>>(
    &AirAdapter::convert_winterfell_trace_table(winterfell_trace),
    &adapted_pub_inputs, /// Public inputs
    &proof_options,
    StoneProverTranscript::new(&[]),
);
```

Your winterfell trace can be converted to a lambdaworks trace by using the helper method `AirAdapter::convert_winterfell_trace_table`. The last thing is to adapt the public inputs. In the case of the adapter, the public inputs contain the winterfell public inputs and also any additional parameters that are needed by the lambdaworks prover. You can create the public inputs by using:

```rust
let adapted_pub_inputs = AirAdapterPublicInputs {
    winterfell_public_inputs: AdapterFieldElement(trace.columns()[1][7]),
    transition_degrees: vec![1, 1],
    transition_exemptions: vec![1, 1],
    transition_offsets: vec![0, 1],
    composition_poly_degree_bound: 8,
    trace_info: TraceInfo::new(2, 8),
};
```

To check more examples you can see the `examples` folder inside this crate.
