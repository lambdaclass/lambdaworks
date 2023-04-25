window.BENCHMARK_DATA = {
  "lastUpdate": 1682449603871,
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
          "id": "3e5a9a69c39252b3b1e34ee5158151249c2acf78",
          "message": "Flamegraph (#270)\n\n* flamegraph in Makefile\n\n* flamegraph_stark in Makefile\n\n* typo fixed\n\n* Ignoring flamegraph.svg\n\n---------\n\nCo-authored-by: Pablo Deymonnaz <deymonnaz@gmail.com>",
          "timestamp": "2023-04-25T13:29:27Z",
          "tree_id": "e02d84ceb4c9501119ee17923cb8795f622124e2",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/3e5a9a69c39252b3b1e34ee5158151249c2acf78"
        },
        "date": 1682429725988,
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
            "value": 54,
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
            "value": 30,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "u64/div",
            "value": 48,
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
            "value": 3480,
            "range": "± 1806",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/evaluate_slice",
            "value": 230624,
            "range": "± 362863",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/add",
            "value": 2845,
            "range": "± 878",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/neg",
            "value": 777,
            "range": "± 223",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/sub",
            "value": 3744,
            "range": "± 891",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/mul",
            "value": 2974,
            "range": "± 794",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/div",
            "value": 3008,
            "range": "± 850",
            "unit": "ns/iter"
          },
          {
            "name": "starks/simple_fibonacci",
            "value": 35258949,
            "range": "± 1277074",
            "unit": "ns/iter"
          },
          {
            "name": "starks/fibonacci_2_columns",
            "value": 54840892,
            "range": "± 2989743",
            "unit": "ns/iter"
          },
          {
            "name": "starks/fibonacci_f17",
            "value": 537213,
            "range": "± 192790",
            "unit": "ns/iter"
          },
          {
            "name": "starks/quadratic_air",
            "value": 35988850,
            "range": "± 466676",
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
          "id": "fa1dcdeb75240fb1f478569e9d9c738b22caad9b",
          "message": "feat: add debug info - trace validation (#272)\n\n* Build validate method skeleton\n\n* Add validation function for trace\n\n* Add clippy suggestions\n\n* Remove unused variables",
          "timestamp": "2023-04-25T14:18:42Z",
          "tree_id": "971bc614e4f034d551fa2210483a3d8f7d88c1bd",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/fa1dcdeb75240fb1f478569e9d9c738b22caad9b"
        },
        "date": 1682432691872,
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
            "value": 159,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "u64/div",
            "value": 168,
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
            "value": 2250,
            "range": "± 1319",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/evaluate_slice",
            "value": 178253,
            "range": "± 267687",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/add",
            "value": 1565,
            "range": "± 359",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/neg",
            "value": 504,
            "range": "± 260",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/sub",
            "value": 2247,
            "range": "± 564",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/mul",
            "value": 1801,
            "range": "± 448",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/div",
            "value": 1720,
            "range": "± 414",
            "unit": "ns/iter"
          },
          {
            "name": "starks/simple_fibonacci",
            "value": 33423506,
            "range": "± 133316",
            "unit": "ns/iter"
          },
          {
            "name": "starks/fibonacci_2_columns",
            "value": 47420084,
            "range": "± 15353",
            "unit": "ns/iter"
          },
          {
            "name": "starks/fibonacci_f17",
            "value": 229642,
            "range": "± 80",
            "unit": "ns/iter"
          },
          {
            "name": "starks/quadratic_air",
            "value": 33895265,
            "range": "± 13298",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "jpcenteno@users.noreply.github.com",
            "name": "Joaquín Centeno",
            "username": "jpcenteno"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "6a08abb98d1562f683e9998643e79058af5d472b",
          "message": "Remove constructor from AIR trait. Impl From as replacement (#240)",
          "timestamp": "2023-04-25T19:00:37Z",
          "tree_id": "9114c09e4c9e0a08260c2157cb6d85c7e57317ac",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/6a08abb98d1562f683e9998643e79058af5d472b"
        },
        "date": 1682449603201,
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
            "value": 54,
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
            "value": 284,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "u64/div",
            "value": 292,
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
            "value": 3029,
            "range": "± 1843",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/evaluate_slice",
            "value": 428103,
            "range": "± 379436",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/add",
            "value": 3111,
            "range": "± 744",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/neg",
            "value": 749,
            "range": "± 227",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/sub",
            "value": 3850,
            "range": "± 1021",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/mul",
            "value": 3193,
            "range": "± 824",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/div",
            "value": 2830,
            "range": "± 884",
            "unit": "ns/iter"
          },
          {
            "name": "starks/simple_fibonacci",
            "value": 32937426,
            "range": "± 30885",
            "unit": "ns/iter"
          },
          {
            "name": "starks/fibonacci_2_columns",
            "value": 46938631,
            "range": "± 14083",
            "unit": "ns/iter"
          },
          {
            "name": "starks/fibonacci_f17",
            "value": 268950,
            "range": "± 111",
            "unit": "ns/iter"
          },
          {
            "name": "starks/quadratic_air",
            "value": 33421914,
            "range": "± 7846",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}