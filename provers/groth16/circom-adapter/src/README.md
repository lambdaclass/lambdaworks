## Retrieving required files from Circom & SnarkJS

1. Have a "\*.circom" file and an "input.json" file. Let's say circom file has name **test.circom**.
2. ```bash
   circom test.circom --r1cs --wasm -p bls12381
   ```

   This will create a **test_js** directory, and a **test.r1cs** file.

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

```bash
circom test.circom --r1cs --wasm -p bls12381;
node test_js/generate_witness.js test_js/test.wasm input.json witness.wtns;
snarkjs wtns export json witness.wtns;
snarkjs r1cs export json test.r1cs test.r1cs.json;
```

<br>

Now we need **test.r1cs.json** and **witness.json** files.

## Using with LambdaWorks Circom Adapter
