window.BENCHMARK_DATA = {
  "lastUpdate": 1682460795962,
  "repoUrl": "https://github.com/lambdaclass/lambdaworks",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "38794644+gabrielbosio@users.noreply.github.com",
            "name": "Gabriel Bosio",
            "username": "gabrielbosio"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "71ceb85c5fc9798a4568bfd3f6ff86910dd41244",
          "message": "[STARKs] Add consistency check for the DEEP composition polynomial (#233)\n\n* [WIP] Add deep consistency check struct\n\n* Fix num of challenges to skip\n\n* Add DEEP consistency check: merkle proofs\n\n* Refactor DEEP consistency check\n\n* Make trace evaluations a frame\n\n* Add frame evaluations verification\n\n* [WIP] Add DEEP composition poly verification\n\n* [WIP] Fix fib and fib17 tests\n\n* Fix DEEP composition poly build\n\n* Fix clippy issues\n\nUse add assign in trace evaluations sum.\nReduce number of arguments in some functions.\nRemove unnecessary use of reference operators.\n\n* Document consistency check of DEEP composition poly\n\n* Remove Prover and Verifier structs\n\nGroup arguments of functions with many of them into structs.\n\n* [WIP] Fix consistency check\n\n* Fix fib17 test\n\n* Fix fib tests\n\n* Fix unnecessary let warning\n\n* Add trace evaluation verification\n\n* Update docs\n\n* Update benchmarks\n\n* Refactor consistency verification\n\n* Address trace coeffs TODO\n\n* Remove clone() in docs\n\n* Remove clones from code\n\n* Check consistency only for one query\n\n* Remove unnecessary enumerate()\n\n---------\n\nCo-authored-by: MauroFab <maurotoscano2@gmail.com>",
          "timestamp": "2023-04-25T22:06:27Z",
          "tree_id": "cab654d50f26291f4a99ec96426b1a9916f82e73",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/71ceb85c5fc9798a4568bfd3f6ff86910dd41244"
        },
        "date": 1682460794643,
        "tool": "cargo",
        "benches": [
          {
            "name": "u64/add",
            "value": 12,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "u64/mul",
            "value": 12,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "u64/pow",
            "value": 65,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "u64/sub",
            "value": 12,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "u64/inv",
            "value": 452,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "u64/div",
            "value": 467,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "u64/eq",
            "value": 3,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/evaluate",
            "value": 3667,
            "range": "± 2065",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/evaluate_slice",
            "value": 282692,
            "range": "± 407160",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/add",
            "value": 3567,
            "range": "± 934",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/neg",
            "value": 715,
            "range": "± 302",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/sub",
            "value": 4672,
            "range": "± 1218",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/mul",
            "value": 3808,
            "range": "± 960",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/div",
            "value": 3707,
            "range": "± 1077",
            "unit": "ns/iter"
          },
          {
            "name": "starks/simple_fibonacci",
            "value": 40696024,
            "range": "± 96222",
            "unit": "ns/iter"
          },
          {
            "name": "starks/fibonacci_2_columns",
            "value": 56998129,
            "range": "± 56265",
            "unit": "ns/iter"
          },
          {
            "name": "starks/fibonacci_f17",
            "value": 381595,
            "range": "± 305",
            "unit": "ns/iter"
          },
          {
            "name": "starks/quadratic_air",
            "value": 40835416,
            "range": "± 65441",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}