pragma circom 2.0.0;

include "circomlib/mimc.circom";

template Test() {
	signal input in;
	signal input k;

	signal output out;

	component hash = MiMC7(10);

	hash.x_in <== in;
	hash.k <== k;

	out <== hash.out;
}

component main = Test();
