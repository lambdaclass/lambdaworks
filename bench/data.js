window.BENCHMARK_DATA = {
  "lastUpdate": 1681501503595,
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
      },
      {
        "commit": {
          "author": {
            "email": "pdeymon@fi.uba.ar",
            "name": "Pablo Deymonnaz",
            "username": "pablodeymo"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "28f6d51be187d94ec2babf8c1f289e3669eb0a56",
          "message": "Revert \"Prover refactor (#231)\" (#256)\n\nThis reverts commit 0d65974d6097554242cedc384d4d3b3a471f8907.",
          "timestamp": "2023-04-14T19:35:36Z",
          "tree_id": "3fd35d4db99e242f7d97c80fc5f4d6cf1952d7f4",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/28f6d51be187d94ec2babf8c1f289e3669eb0a56"
        },
        "date": 1681501502842,
        "tool": "cargo",
        "benches": [
          {
            "name": "u64/add",
            "value": 10,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "u64/mul",
            "value": 10,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "u64/pow",
            "value": 53,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "u64/sub",
            "value": 10,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "u64/inv",
            "value": 277,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "u64/div",
            "value": 290,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "u64/eq",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/evaluate",
            "value": 3240,
            "range": "± 1744",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/evaluate_slice",
            "value": 353819,
            "range": "± 341764",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/add",
            "value": 3091,
            "range": "± 832",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/neg",
            "value": 764,
            "range": "± 238",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/sub",
            "value": 3745,
            "range": "± 957",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/mul",
            "value": 3038,
            "range": "± 810",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/div",
            "value": 3074,
            "range": "± 745",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 2018780398,
            "range": "± 4221375",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 4450113636,
            "range": "± 12262863",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 4212034953,
            "range": "± 2692222",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 9711861602,
            "range": "± 77781619",
            "unit": "ns/iter"
          },
          {
            "name": "starks/simple_fibonacci",
            "value": 32580020,
            "range": "± 13197",
            "unit": "ns/iter"
          },
          {
            "name": "starks/fibonacci_2_columns",
            "value": 46573681,
            "range": "± 10917",
            "unit": "ns/iter"
          },
          {
            "name": "starks/fibonacci_f17",
            "value": 186644,
            "range": "± 44",
            "unit": "ns/iter"
          },
          {
            "name": "starks/quadratic_air",
            "value": 33076226,
            "range": "± 14285",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}