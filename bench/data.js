window.BENCHMARK_DATA = {
  "lastUpdate": 1682546879408,
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
      },
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
          "id": "064f51ca23bce97a483946eeace28bb7630fbba3",
          "message": "Add iai support and benchmarks for performance comparison between new PRs and main (#255)\n\n* Added poly interpolation benchs\n\n* Changed benchs to StarkField, other improves\n\n* Reorg benchs, changed names\n\n* Added gpu README with poly interpol. benchs\n\n* Reverted removal of math all_benchmarks\n\n* Reverted Makefile change\n\n* Reduced input sets\n\n* Fix clippy\n\n* Changed times to 1 precision\n\n* Remove fft from `math` module.\n\n* Make a separate fft crate to avoid circular dependencies.\n\n* Separated FFT benchmarks from polynomial evaluation\n\n* Forgot to add new benches for exec\n\n* Formatting nits\n\n* Moved benchmarks to repo's book\n\n* Bring back benches.\n\n* Move tests to gpu.\n\n* Restructured metal benchs, added iai\n\n* Fix error handling.\n\n* Remove more repeated code.\n\n* Reverted iai benchs on metal\n\n* Added criterion and iai benchs for fft crate\n\n* Create iai CI job, update criterion's\n\n* Added polynomial benches\n\n* Added iai poly benches\n\n* Added field benches\n\n* Added stark benchs\n\n* Fixed merge\n\n* Reverted small change\n\n* Fixed jobs\n\n* Fix jobs again\n\n* Fix jobs again\n\n* Fix jobs again\n\n* Format iai tests, prevent inline\n\n* Remove u64 tests from CI\n\n* Changed iai job to restore/save in cache\n\n* Fix clippy, change job name\n\n* Fix iai job\n\n* Added job for caching main's iai, other fixes\n\n* Fix iai job, only run criterion on main\n\n* Removed unused fft benchs, changed iai_fft order\n\n* Nit\n\n* Added criterion macos tests, refactored actions\n\n* Replaced actions-rs/ with dtolnay/ toolchain to match other jobs\n\n* Small fixes in names, also another action-rs\n\n* Fix iai job\n\n* Update .github/workflows/iai_benchs_main.yml\n\nCo-authored-by: Mario Rugiero <mario.rugiero@nextroll.com>\n\n* Update .github/workflows/iai_benchs_main.yml\n\nCo-authored-by: Mario Rugiero <mario.rugiero@nextroll.com>\n\n* Update .github/workflows/iai_benchs_main.yml\n\nCo-authored-by: Mario Rugiero <mario.rugiero@nextroll.com>\n\n---------\n\nCo-authored-by: Martin Paulucci <martin.paulucci@lambdaclass.com>\nCo-authored-by: Mario Rugiero <mario.rugiero@nextroll.com>",
          "timestamp": "2023-04-26T17:43:32Z",
          "tree_id": "d1187743220df5e758525817d827c89e87f8315b",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/064f51ca23bce97a483946eeace28bb7630fbba3"
        },
        "date": 1682543431166,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 2562684915,
            "range": "± 1001712",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 4153390394,
            "range": "± 22356098",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 5362155406,
            "range": "± 1135247",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 9217541673,
            "range": "± 21854871",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 11187651606,
            "range": "± 8608242",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 19942652620,
            "range": "± 32110911",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 23320769406,
            "range": "± 14539031",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 42083781153,
            "range": "± 105465950",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential",
            "value": 1295147041,
            "range": "± 509829",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #2",
            "value": 2717860974,
            "range": "± 787598",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #3",
            "value": 5691714891,
            "range": "± 2174935",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #4",
            "value": 11893295430,
            "range": "± 5714666",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 74341639,
            "range": "± 2603956",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 146887457,
            "range": "± 570292",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 291082638,
            "range": "± 1017619",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 595722263,
            "range": "± 4055584",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 3881211162,
            "range": "± 7138350",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 8122762830,
            "range": "± 2181855",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 16974742841,
            "range": "± 2885582",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 35412223309,
            "range": "± 14168914",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 58695881632,
            "range": "± 15393260",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 117643349327,
            "range": "± 25849853",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 235862328793,
            "range": "± 162707307",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 473650036745,
            "range": "± 84305194",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 34,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 152,
            "range": "± 10",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 86,
            "range": "± 12",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 15,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 159,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 119,
            "range": "± 13",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 119,
            "range": "± 24",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci",
            "value": 33753475,
            "range": "± 16817",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/2 column Fibonacci",
            "value": 47252242,
            "range": "± 22379",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Fibonacci F17",
            "value": 274893,
            "range": "± 42",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Quadratic AIR",
            "value": 33895682,
            "range": "± 18800",
            "unit": "ns/iter"
          }
        ]
      },
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
          "id": "96326495313b973aa6d2416c711743b5fc0c5b1b",
          "message": "Fix macos criterion benchs job (#286)",
          "timestamp": "2023-04-26T18:40:32Z",
          "tree_id": "3b547f0e42a2e87c0b189ffd0f60f2d4c11d3a94",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/96326495313b973aa6d2416c711743b5fc0c5b1b"
        },
        "date": 1682546878737,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 2568118439,
            "range": "± 6103105",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 4091750189,
            "range": "± 34884656",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 5367554139,
            "range": "± 2172487",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 9152371337,
            "range": "± 42333040",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 11199446173,
            "range": "± 3657029",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 19680654323,
            "range": "± 21671501",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 23353134792,
            "range": "± 4315003",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 41855206642,
            "range": "± 197548859",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential",
            "value": 1296058256,
            "range": "± 1083019",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #2",
            "value": 2720351934,
            "range": "± 1973609",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #3",
            "value": 5692696978,
            "range": "± 3060522",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #4",
            "value": 11905278929,
            "range": "± 6034351",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 71409096,
            "range": "± 2270378",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 148131020,
            "range": "± 1895226",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 290829024,
            "range": "± 1473484",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 593736504,
            "range": "± 2388779",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 3885176726,
            "range": "± 4033087",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 8132551474,
            "range": "± 7690352",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 16987610769,
            "range": "± 9240638",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 35429920868,
            "range": "± 13800842",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 58786345509,
            "range": "± 20955220",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 117836079705,
            "range": "± 29705991",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 236178068353,
            "range": "± 101662538",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 477627935841,
            "range": "± 1697066361",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 1079,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 68983,
            "range": "± 40",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 713,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 21,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 1127,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 745,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 754,
            "range": "± 15",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci",
            "value": 33781808,
            "range": "± 12378",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/2 column Fibonacci",
            "value": 47391778,
            "range": "± 34149",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Fibonacci F17",
            "value": 272896,
            "range": "± 380",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Quadratic AIR",
            "value": 34006807,
            "range": "± 19986",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}