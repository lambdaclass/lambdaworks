#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "fp.h"

extern "C" __global__ void main_fft (int *input, int *twiddles, const int n) {
    // divide input in groups, starting with 1, duplicating the number of groups in each stage.
    int group_count = 1;
    int group_size = n;

    // for each group, there'll be group_size / 2 butterflies.
    // a butterfly is the atomic operation of a FFT, e.g: (a, b) = (a + wb, a - wb).
    // The 0.5 factor is what gives FFT its performance, it recursively halves the problem size
    // (group size).

    while (group_count < n) {
        int distance = group_size / 2;
        for (int group = 0; group < group_count; group++) {
            int first_in_group = group * group_size;
            int first_in_next_group = first_in_group + distance;
            Fp w = twiddles[group];
            for(int i = first_in_group; i < first_in_next_group; i++) {
                Fp a = input[i];
                Fp b = input[i + distance];
                input[i] = (a + w*b).asUInt32();
                input[i + distance] = (a - w*b).asUInt32();
            };
        };
        group_count *= 2;
        group_size /= 2;
    };
};
