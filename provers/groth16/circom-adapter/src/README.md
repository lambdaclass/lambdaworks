# Circom - LambdaWorks Groth16 Adapter

This package allows one to perform trusted setup, prove, and verify constraints generated by [SnarkJS](https://github.com/iden3/snarkjs) from a [Circom](https://github.com/iden3/circom) circuit.

## Retrieving required files

0. Have Circom and SnarkJS installed,

1. Have a "\*.circom" file and an "input.json" file. Let's say circom file has name **test.circom**.
2. ```bash
   circom test.circom --r1cs --wasm -p bls12381
   ```

   This will create a **test_js** directory, and a **test.r1cs** file. Do not skip the **-p bls12381** flag as this is the only field supported by this adapter right now.

3. ```bash
   node test_js/generate_witness.js test_js/test.wasm input.json witness.wtns
   ```

   This will generate a **witness.wtns** file.

4. ```bash
   snarkjs wtns export json witness.wtns
   ```

   This will generate a **witness.json** file.

5. ```bash
   snarkjs r1cs export json test.r1cs test.r1cs.json
   ```
   This will generate an **test.r1cs.json** file.

<br>

All at once, you can copy-paste the following to the terminal in the same directory as your circuit. Please do remember changing 'test' with your .circom file name.

```bash
circom test.circom --r1cs --wasm -p bls12381;
node test_js/generate_witness.js test_js/test.wasm input.json witness.wtns;
snarkjs wtns export json witness.wtns;
snarkjs r1cs export json test.r1cs test.r1cs.json;
rm -rf test_js test.r1cs witness.wtns; # Delete unnecessary artifacts
```

Now we need **test.r1cs.json** and **witness.json** files.

## Using with LambdaWorks Circom Adapter

This package exposes a **circom_to_lambda** function that accepts two parameters:

1. Stringified content of the .r1cs.json file
2. Stringified content of the .witness.json file

Relative path to a file named **test** will be just **"test"** if it's placed in the same level as Cargo.toml of this package. In this case, this method can be invoked as follows:

```rust
// ...
let (qap, w) = circom_to_lambda(
   &fs::read_to_string("test.r1cs.json").expect("Error reading file"),
   &fs::read_to_string("witness.json").expect("Error reading file"),
);
```

As seen, this function returns a Lambdaworks-compatible QAP and the witness assignments. Then one should perform setup, prove, and verify. Here's the complete procedure:

```rust
fn poseidon_parse_prove_verify() {
   let (qap, w) = circom_to_lambda(
      &fs::read_to_string("test.r1cs.json").expect("Error reading file"),
      &fs::read_to_string("witness.json").expect("Error reading file"),
   );

   let (pk, vk) = setup(&qap);
   let accept = verify(
      &vk,
      &Prover::prove(&w, &qap, &pk),
      &w[..qap.num_of_public_inputs],
   );
   assert!(accept);
}
```

For convenience, one can look up to [integration_tests.rs](integration_tests.rs) file and see an example where the Poseidon hash of "100" is proven and verified. The **vitalik_w_and_qap** example issues an example one can investigate with pen and paper, giving a clearer idea what the adapter does.
