# High level API: Fibonacci example

Let's go over the main test we use for our prover, where we compute a STARK proof for a fibonacci trace with 4 rows and then verify it.

```rust
fn test_prove_fib() {
    let trace = simple_fibonacci::fibonacci_trace([FE::from(1), FE::from(1)], 8);
    let proof_options = ProofOptions::default_test_options();

    let pub_inputs = FibonacciPublicInputs {
        a0: FE::one(),
        a1: FE::one(),
    };

    let proof = prove::<F, FibonacciAIR<F>>(&trace, &pub_inputs, &proof_options).unwrap();
    assert!(verify::<F, FibonacciAIR<F>>(&proof, &pub_inputs, &proof_options));
}
```

The proving system revolves around the `prove` function, that takes a trace, public inputs and proof options as inputs to generate a proof, and a `verify` function that takes the generated proof, the public inputs and the proof options as inputs, outputting `true` when the proof is verified correctly and `false` otherwise. Note that the public inputs and proof options should be the same for both. Public inputs should be shared by the Cairo runner to prover and verifier, and the proof options should have been agreed on beforehand by the two entities beforehand.

Below we go over the main things involved in this code.

## AIR

To prove the integrity of a fibonacci trace, we first need to define what it means for a trace to be valid. As we've talked about in the recap, this involves defining an `AIR` for our computation where we specify both the boundary and transition constraints for a fibonacci sequence.

In code, this is done through the `AIR` trait. Implementing `AIR` requires defining a couple methods, but the two most important ones are `boundary_constraints` and `compute_transition`, which encode the boundary and transition constraints of our computation.


### Boundary Constraints
For our Fibonacci `AIR`, boundary constraints look like this:

```rust
fn boundary_constraints(
    &self,
    _rap_challenges: &Self::RAPChallenges,
) -> BoundaryConstraints<Self::Field> {
    let a0 = BoundaryConstraint::new_simple(0, self.pub_inputs.a0.clone());
    let a1 = BoundaryConstraint::new_simple(1, self.pub_inputs.a1.clone());

    BoundaryConstraints::from_constraints(vec![a0, a1])
}
```

The `BoundaryConstraint` struct represents a specific boundary constraint, meaning "column `i` at row `j` should be equal to `x`". In this case, because we have only one column, we are using the `new_simple` method to simply say 

- Row `0` should equal the public input `a0`, which in the typical fibonacci is set to 1.
- Row `1` should equal the public input `a1`, which in the typical fibonacci is set to 1.

In the case of multiple columns, the `new` method exists so you can also specify column number.

After instantiating each of these constraints, we return all of them through the struct `BoundaryConstraints`.

### Transition Constraints

The way we specify our fibonacci transition constraint looks like this:

```rust
fn compute_transition(
    &self,
    frame: &air::frame::Frame<Self::Field>,
    _rap_challenges: &Self::RAPChallenges,
) -> Vec<FieldElement<Self::Field>> {
    let first_row = frame.get_row(0);
    let second_row = frame.get_row(1);
    let third_row = frame.get_row(2);

    vec![third_row[0] - second_row[0] - first_row[0]]
}
```

It's not completely obvious why this is how we chose to express transition constraints, so let's talk a little about it. 

What we need to specify in this method is the relationship that has to hold between the current step of computation and the previous ones. For this, we get a `Frame` as an argument. This is a struct holding the current step (i.e. the current row of the trace) and all previous ones needed to encode our constraint. In our case, this is the current row and the two previous ones. To access rows we use the `get_row` method. The current step is always the last row (in our case `2`), with the others coming before it.

In our `compute_transition` method we get the three rows we need and return

```rust
third_row[0] - second_row[0] - first_row[0]
```

which is the value that needs to be zero for our constraint to hold. Because we support multiple transition constraints, we actually return a vector with one value per constraint, so the first element holds the first constraint value and so on.

## TraceTable

After defining our AIR, we create our specific trace to prove against it. 

```rust
let trace = fibonacci_trace([FE17::new(1), FE17::new(1)], 4);

let trace_table = TraceTable {
    table: trace.clone(),
    num_cols: 1,
};
```

`TraceTable` is the struct holding execution traces; the `num_cols` says how many columns the trace has, the `table` field is a `vec` holding the actual values of the trace in row-major form, meaning if the trace looks like this

```
| 1  | 2  |
| 3  | 4  |
| 5  | 6  |
```

then its corresponding `TraceTable` is 

```rust
let trace_table = TraceTable {
    table: vec![1, 2, 3, 4, 5, 6],
    num_cols: 2,
};
```

In our example, `fibonacci_trace` is just a helper function we use to generate the fibonacci trace with `4` rows and `[1, 1]` as the first two values.

## AIR Context

After specifying our constraints and trace, the only thing left to do is provide a few parameters related to the STARK protocol and our `AIR`. These specify things such as the number of columns of the trace and proof configuration, among others. They are all encapsulated in the `AirContext` struct, which in our example we instantiate like this:

```rust
let context = AirContext {
    options: ProofOptions {
        blowup_factor: 2,
        fri_number_of_queries: 1,
        coset_offset: 3,
    },
    trace_columns: trace_table.n_cols,
    transition_degrees: vec![1],
    transition_exemptions: vec![2],
    transition_offsets: vec![0, 1, 2],
    num_transition_constraints: 1,
};
```

Let's go over each of them:

- `options` requires a `ProofOptions` struct holding specific parameters related to the STARK protocol to be used when proving. They are:
    - The `blowup_factor` used for the trace LDE extension, a parameter related to the security of the protocol.
    - The number of queries performed by the verifier when doing `FRI`, also related to security.
    - The `offset` used for the LDE coset. This depends on the field being used for the STARK proof.
- `trace_columns` are the number of columns of the trace, respectively.
- `transition_degrees` holds the degree of each transition constraint.
- `transition_exemptions` is a `Vec` which tells us, for each column, the number of rows the transition constraints should not apply, starting from the end of the trace. In the example, the transition constraints won't apply on the last two rows of the trace.
- `transition_offsets` holds the indexes that define a frame for our `AIR`. In our fibonacci case, these are `[0, 1, 2]` because we need the current row and the two previous one to define our transition constraint.
- `num_transition_constraints` simply says how many transition constraints our `AIR` has.

## Proving execution

Having defined all of the above, proving our fibonacci example amounts to instantiating the necessary structs and then calling `prove` passing the trace, public inputs and proof options. We use a simple implementation of a hasher called `TestHasher` to handle merkle proof building.

```rust 
let proof = prove(&trace_table, &pub_inputs, &proof_options);
```

Verifying is then done by passing the proof of execution along with the same `AIR` to the `verify` function.

```rust
assert!(verify(&proof, &pub_inputs, &proof_options));
```
