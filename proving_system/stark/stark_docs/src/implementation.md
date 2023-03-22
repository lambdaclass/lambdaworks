# STARKs lambdaworks implementation

The goal of this section will be to go over the details of the implementation of the proving system. To this end, we will follow the flow the example in the `recap` chapter, diving deeper into the code when necessary and explaining how it fits into a more general case.

The proving system revolves around  `prove` function, that takes a trace and an AIR as inputs to generate a proof, and a `verify` function that takes the proof and the AIR as inputs, outputing `true` when the proof is verified correctly and `false` otherwise.

## AIR