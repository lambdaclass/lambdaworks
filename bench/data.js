window.BENCHMARK_DATA = {
  "lastUpdate": 1683045508396,
  "repoUrl": "https://github.com/lambdaclass/lambdaworks",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "estefano.bargas@fing.edu.uy",
            "name": "Estéfano Bargas",
            "username": "xqft"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "657f6b6cfe7e67f25483dc63d6345908e8d978c2",
          "message": "Added support to all FFT implementations for inputs of all sizes via zero padding (#275)\n\n* Changed param name, coeffs to input\n\n* Added non-pow-2/edge cases support for CPU FFT\n\n* Removed next_pow_of_two helper, added tests\n\n* Removed null input edge case handling\n\n* Added zero-padding to Metal FFT API\n\n* Fix clippy\n\n* Use prop_assert in proptests\n\n* Added FFT special considerations doc\n\n* Changed get_twiddles to accept u64\n\n* Update doc",
          "timestamp": "2023-05-02T16:32:38Z",
          "tree_id": "c8f6ee9879a8fe57dacf95cb68d507277d1565e8",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/657f6b6cfe7e67f25483dc63d6345908e8d978c2"
        },
        "date": 1683045506030,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 136815744,
            "range": "± 3620787",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 268674208,
            "range": "± 1220051",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 527235917,
            "range": "± 2278131",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 1037766812,
            "range": "± 11439154",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 33780581,
            "range": "± 198004",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 69367016,
            "range": "± 740200",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 132752914,
            "range": "± 1011997",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 275326114,
            "range": "± 3253874",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 31082369,
            "range": "± 197805",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 59602708,
            "range": "± 985765",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 120897781,
            "range": "± 3509269",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 253219635,
            "range": "± 22944187",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 169710917,
            "range": "± 773257",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 329420343,
            "range": "± 1570501",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 647299854,
            "range": "± 4170879",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1310941417,
            "range": "± 10305694",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 467693489,
            "range": "± 2256462",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 931061583,
            "range": "± 5023126",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1832943812,
            "range": "± 4048020",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3578632666,
            "range": "± 71867245",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}