window.BENCHMARK_DATA = {
  "lastUpdate": 1682531327567,
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
          "id": "1d776c54f9134805af8b64fd2707ea8b5e9a8a8b",
          "message": "Example reading file with cairo-rs (#269)\n\n* Start cairo execution trace builder\n\n* Refactor cairo memory\n\n* Implemented flags_and_offsets function for CairoTrace\n\n* Add method for trace representation of instruction flags\n\n* Implement trace representation function for instruction offsets\n\n* Add method to compute dst_addrs column for trace\n\n* cairo module using FE\n\n* Use aux_get_last_nim_of_FE function in all instruction flags TryFrom's\n\n* Refactor CairoMemory to hold FE values\n\n* Add some comments\n\n* compute_res\n\n* compute_res completed\n\n* compute_op1 WIP\n\n* Finish first implementation of build_cairo_trace function\n\n* Start cairo execution trace testing\n\n* Add test for build_cairo_execution_trace\n\n* Save work in progress: debugging cairo trace\n\n* Save work in progress: more debugging\n\n* Fix some bugs\n\n* Remove print\n\n* Finish trace test for simple_program, start call_func test\n\n* Finish second test\n\n* Add documentation and comments in the code\n\n* Make code more idiomatic\n\n* Change auxiliary function name\n\n* Add some more documentation\n\n* Example reading file with cairo-rs\n\n* cairo VM files moved inside starks\n\n* integration_cairo directory removed\n\n* default value for entrypoint is \"main\"\n\n* function name changed to run_program\n\n* parser.rs file renamed to run.rs\n\n* documentation in run_program\n\n* fmt\n\n---------\n\nCo-authored-by: Mariano Nicolini <mariano.nicolini.91@gmail.com>\nCo-authored-by: Pablo Deymonnaz <deymonnaz@gmail.com>",
          "timestamp": "2023-04-26T17:41:28Z",
          "tree_id": "4cb6b68cd1340b7636cbba3549e0f82b8b28880f",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/1d776c54f9134805af8b64fd2707ea8b5e9a8a8b"
        },
        "date": 1682531326751,
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
            "value": 234,
            "range": "± 9",
            "unit": "ns/iter"
          },
          {
            "name": "u64/div",
            "value": 248,
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
            "value": 2184,
            "range": "± 1221",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/evaluate_slice",
            "value": 241052,
            "range": "± 271916",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/add",
            "value": 1566,
            "range": "± 413",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/neg",
            "value": 746,
            "range": "± 232",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/sub",
            "value": 2355,
            "range": "± 590",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/mul",
            "value": 1729,
            "range": "± 444",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/div",
            "value": 1771,
            "range": "± 425",
            "unit": "ns/iter"
          },
          {
            "name": "starks/simple_fibonacci",
            "value": 34226828,
            "range": "± 11935",
            "unit": "ns/iter"
          },
          {
            "name": "starks/fibonacci_2_columns",
            "value": 48031718,
            "range": "± 19501",
            "unit": "ns/iter"
          },
          {
            "name": "starks/fibonacci_f17",
            "value": 276017,
            "range": "± 78",
            "unit": "ns/iter"
          },
          {
            "name": "starks/quadratic_air",
            "value": 34472978,
            "range": "± 16870",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}