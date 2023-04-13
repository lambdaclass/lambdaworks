window.BENCHMARK_DATA = {
  "lastUpdate": 1681406044247,
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
          "distinct": true,
          "id": "4918d4ca1a976fe9d73493ab3a24c83cbdd2d0df",
          "message": "Added polynomial interpolation benchs and markdown table comparison, and other improvements (#245)\n\n* Added poly interpolation benchs\n\n* Changed benchs to StarkField, other improves\n\n* Reorg benchs, changed names\n\n* Added gpu README with poly interpol. benchs\n\n* Reverted removal of math all_benchmarks\n\n* Reverted Makefile change\n\n* Reduced input sets\n\n* Fix clippy\n\n* Changed times to 1 precision\n\n* Separated FFT benchmarks from polynomial evaluation\n\n* Forgot to add new benches for exec\n\n* Formatting nits\n\n* Moved benchmarks to repo's book",
          "timestamp": "2023-04-13T17:04:33Z",
          "tree_id": "3fd35d4db99e242f7d97c80fc5f4d6cf1952d7f4",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/4918d4ca1a976fe9d73493ab3a24c83cbdd2d0df"
        },
        "date": 1681406042548,
        "tool": "cargo",
        "benches": [
          {
            "name": "u64/add",
            "value": 4,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "u64/mul",
            "value": 4,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "u64/pow",
            "value": 21,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "u64/sub",
            "value": 4,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "u64/inv",
            "value": 197,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "u64/div",
            "value": 206,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "u64/eq",
            "value": 2,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/evaluate",
            "value": 2471,
            "range": "± 1265",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/evaluate_slice",
            "value": 155960,
            "range": "± 298471",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/add",
            "value": 1676,
            "range": "± 414",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/neg",
            "value": 780,
            "range": "± 223",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/sub",
            "value": 2379,
            "range": "± 534",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/mul",
            "value": 1724,
            "range": "± 431",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/div",
            "value": 1779,
            "range": "± 424",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 2244433741,
            "range": "± 1877560",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 3817204115,
            "range": "± 21825654",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 4697523052,
            "range": "± 1262181",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 8643328635,
            "range": "± 24283246",
            "unit": "ns/iter"
          },
          {
            "name": "starks/simple_fibonacci",
            "value": 32686388,
            "range": "± 126246",
            "unit": "ns/iter"
          },
          {
            "name": "starks/fibonacci_2_columns",
            "value": 46652765,
            "range": "± 31808",
            "unit": "ns/iter"
          },
          {
            "name": "starks/fibonacci_f17",
            "value": 157583,
            "range": "± 138",
            "unit": "ns/iter"
          },
          {
            "name": "starks/quadratic_air",
            "value": 33166769,
            "range": "± 17943",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}