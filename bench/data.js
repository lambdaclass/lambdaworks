window.BENCHMARK_DATA = {
  "lastUpdate": 1687979686910,
  "repoUrl": "https://github.com/lambdaclass/lambdaworks",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "12560266+MauroToscano@users.noreply.github.com",
            "name": "Mauro Toscano",
            "username": "MauroToscano"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "a21d2c5ede062ad7758c86c4ddea7bdb934055d8",
          "message": "Generalize hashes in trees (#456)\n\n* 512 hash\n\n* 512 hash\n\n* Final version\n\n* Fixed comment\n\n* Add batch tree\n\n* Add batch trees\n\n* Fmt\n\n* Add generalization\n\n* Add generalization\n\n* Fmt\n\n* Add boundary to avoid panics\n\n* Add boundary to avoid panics\n\n* Fix benches\n\n* Remove unused backends\n\n* Add default types\n\n* Fmt",
          "timestamp": "2023-06-28T19:08:16Z",
          "tree_id": "ab4a6132aa13617af162c8e5d3bfa4edf917a962",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/a21d2c5ede062ad7758c86c4ddea7bdb934055d8"
        },
        "date": 1687979681192,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 130167385,
            "range": "± 2291252",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 253703146,
            "range": "± 1551360",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 495952385,
            "range": "± 1279920",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 998107583,
            "range": "± 16180488",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 34127261,
            "range": "± 167473",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 69046514,
            "range": "± 686752",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 135591189,
            "range": "± 1339870",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 282218510,
            "range": "± 1340148",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 30152559,
            "range": "± 681425",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 58542369,
            "range": "± 1544921",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 122856121,
            "range": "± 3381771",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 213587208,
            "range": "± 21444312",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 166213416,
            "range": "± 5210842",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 398292593,
            "range": "± 54074506",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 859934541,
            "range": "± 19015617",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1669076854,
            "range": "± 80500614",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 539787916,
            "range": "± 66418598",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 1338705708,
            "range": "± 131814405",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 2692747792,
            "range": "± 417944260",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3622238958,
            "range": "± 70178142",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}