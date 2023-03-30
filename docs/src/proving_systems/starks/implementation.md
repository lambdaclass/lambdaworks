# STARKs Prover Lambdaworks Implementation

The goal of this section will be to go over the details of the implementation of the proving system. To this end, we will follow the flow the example in the `recap` chapter, diving deeper into the code when necessary and explaining how it fits into a more general case.

This implementation couldn't be done without checking Facebook's [Winterfell](https://github.com/facebook/winterfell) and Max Gillett's [Giza](https://github.com/maxgillett/giza). We want to thank everyone involved in them, along with Shahar Papini and Lior Goldberg from Starkware who also provided us valuable insight.
