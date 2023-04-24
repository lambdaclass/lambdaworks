window.BENCHMARK_DATA = {
  "lastUpdate": 1682371327200,
  "repoUrl": "https://github.com/lambdaclass/lambdaworks",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "67517699+ilitteri@users.noreply.github.com",
            "name": "Ivan Litteri",
            "username": "ilitteri"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d0b0bc96abccd0ef571069a2ad29b905f3c1e461",
          "message": "Fix typos in `api.md` (#264)",
          "timestamp": "2023-04-21T14:18:16Z",
          "tree_id": "8d8c80e4608dd9df1d4e731b545e0152a9db58e3",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/d0b0bc96abccd0ef571069a2ad29b905f3c1e461"
        },
        "date": 1682087055350,
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
            "value": 315,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "u64/div",
            "value": 323,
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
            "value": 2666,
            "range": "± 1796",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/evaluate_slice",
            "value": 312190,
            "range": "± 405615",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/add",
            "value": 3093,
            "range": "± 707",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/neg",
            "value": 571,
            "range": "± 252",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/sub",
            "value": 3869,
            "range": "± 1036",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/mul",
            "value": 3277,
            "range": "± 754",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/div",
            "value": 2923,
            "range": "± 872",
            "unit": "ns/iter"
          },
          {
            "name": "starks/simple_fibonacci",
            "value": 35102375,
            "range": "± 73451",
            "unit": "ns/iter"
          },
          {
            "name": "starks/fibonacci_2_columns",
            "value": 51253500,
            "range": "± 2756844",
            "unit": "ns/iter"
          },
          {
            "name": "starks/fibonacci_f17",
            "value": 417070,
            "range": "± 56294",
            "unit": "ns/iter"
          },
          {
            "name": "starks/quadratic_air",
            "value": 35695836,
            "range": "± 277328",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mariano.nicolini.91@gmail.com",
            "name": "Mariano A. Nicolini",
            "username": "entropidelic"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c92e63ed5ef3e30bec4365f117e47a99d6ee4e37",
          "message": "feat: add cairo trace (#251)\n\n* Start cairo execution trace builder\n\n* Refactor cairo memory\n\n* Implemented flags_and_offsets function for CairoTrace\n\n* Add method for trace representation of instruction flags\n\n* Implement trace representation function for instruction offsets\n\n* Add method to compute dst_addrs column for trace\n\n* cairo module using FE\n\n* Use aux_get_last_nim_of_FE function in all instruction flags TryFrom's\n\n* Refactor CairoMemory to hold FE values\n\n* Add some comments\n\n* compute_res\n\n* compute_res completed\n\n* compute_op1 WIP\n\n* Finish first implementation of build_cairo_trace function\n\n* Start cairo execution trace testing\n\n* Add test for build_cairo_execution_trace\n\n* Save work in progress: debugging cairo trace\n\n* Save work in progress: more debugging\n\n* Fix some bugs\n\n* Remove print\n\n* Finish trace test for simple_program, start call_func test\n\n* Finish second test\n\n* Add documentation and comments in the code\n\n* Make code more idiomatic\n\n* Change auxiliary function name\n\n* Add some more documentation\n\n---------\n\nCo-authored-by: Pablo Deymonnaz <deymonnaz@gmail.com>",
          "timestamp": "2023-04-24T21:15:44Z",
          "tree_id": "b92478b8cc50740637a27c7a7238c4a9ab524504",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/c92e63ed5ef3e30bec4365f117e47a99d6ee4e37"
        },
        "date": 1682371325921,
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
            "value": 11,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "u64/pow",
            "value": 64,
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
            "value": 32,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "u64/div",
            "value": 45,
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
            "value": 3456,
            "range": "± 1958",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/evaluate_slice",
            "value": 323904,
            "range": "± 423730",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/add",
            "value": 3337,
            "range": "± 1004",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/neg",
            "value": 920,
            "range": "± 250",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/sub",
            "value": 4345,
            "range": "± 1283",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/mul",
            "value": 3470,
            "range": "± 996",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/div",
            "value": 3551,
            "range": "± 947",
            "unit": "ns/iter"
          },
          {
            "name": "starks/simple_fibonacci",
            "value": 41173886,
            "range": "± 258609",
            "unit": "ns/iter"
          },
          {
            "name": "starks/fibonacci_2_columns",
            "value": 63183904,
            "range": "± 1912909",
            "unit": "ns/iter"
          },
          {
            "name": "starks/fibonacci_f17",
            "value": 498554,
            "range": "± 63470",
            "unit": "ns/iter"
          },
          {
            "name": "starks/quadratic_air",
            "value": 41897029,
            "range": "± 341674",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}