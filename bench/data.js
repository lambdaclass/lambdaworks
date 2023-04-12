window.BENCHMARK_DATA = {
  "lastUpdate": 1681335444884,
  "repoUrl": "https://github.com/lambdaclass/lambdaworks",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "49622509+jrchatruc@users.noreply.github.com",
            "name": "Javier Rodríguez Chatruc",
            "username": "jrchatruc"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ae10473aa3c4cd0fdf380a86c5d55e658795cf1a",
          "message": "Benchmarks for stark prover (#239)\n\n* [WIP] Benchmarks for stark prover\n\n* Clippy\n\n* Test workflow\n\n* Fix bench command\n\n* Fix command\n\n* Serve benchmarks under the /bench path\n\n* Add one more power size to the fft benchmarks\n\n* Update .github/workflows/benchmarks.yml\n\nCo-authored-by: Martin Paulucci <martin.paulucci@lambdaclass.com>\n\n* Merge everything into a single yaml\n\n---------\n\nCo-authored-by: Martin Paulucci <martin.paulucci@lambdaclass.com>",
          "timestamp": "2023-04-11T18:30:56Z",
          "tree_id": "b25d44a528c83c4e532e2f0f168333a482b56a9b",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/ae10473aa3c4cd0fdf380a86c5d55e658795cf1a"
        },
        "date": 1681238241332,
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
            "value": 22,
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
            "value": 22,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "u64/div",
            "value": 27,
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
            "value": 2013,
            "range": "± 1338",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/evaluate_slice",
            "value": 249321,
            "range": "± 249884",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/add",
            "value": 1612,
            "range": "± 369",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/neg",
            "value": 735,
            "range": "± 241",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/sub",
            "value": 2325,
            "range": "± 542",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/mul",
            "value": 1765,
            "range": "± 405",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/div",
            "value": 1677,
            "range": "± 451",
            "unit": "ns/iter"
          },
          {
            "name": "FFT/Sequential NR radix2 for 2^21 elements",
            "value": 477101038,
            "range": "± 2479469",
            "unit": "ns/iter"
          },
          {
            "name": "FFT/Sequential RN radix2 for 2^21 elements",
            "value": 668745738,
            "range": "± 7060688",
            "unit": "ns/iter"
          },
          {
            "name": "FFT/Sequential NR radix2 for 2^22 elements",
            "value": 1005910505,
            "range": "± 6043701",
            "unit": "ns/iter"
          },
          {
            "name": "FFT/Sequential RN radix2 for 2^22 elements",
            "value": 1538814398,
            "range": "± 28139873",
            "unit": "ns/iter"
          },
          {
            "name": "starks/simple_fibonacci",
            "value": 32521480,
            "range": "± 11894",
            "unit": "ns/iter"
          },
          {
            "name": "starks/fibonacci_2_columns",
            "value": 46429517,
            "range": "± 19583",
            "unit": "ns/iter"
          },
          {
            "name": "starks/fibonacci_f17",
            "value": 99445,
            "range": "± 68",
            "unit": "ns/iter"
          },
          {
            "name": "starks/quadratic_air",
            "value": 33018131,
            "range": "± 71763",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49622509+jrchatruc@users.noreply.github.com",
            "name": "Javier Rodríguez Chatruc",
            "username": "jrchatruc"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "df56e0f08a5bb8d69bee5a922d38d2c0534d6e97",
          "message": "Add feature flag to disable Fiat-Shamir (#242)\n\n* Add feature flag to disable Fiat-Shamir\n\n* Fix conditional compilation so make clippy can run with all features, rename dummy_fiat_shamir to test_fiat_shamir\n\n* Remove comment",
          "timestamp": "2023-04-12T21:27:25Z",
          "tree_id": "b114e6d86c0017959af63424bff5797c3991e43b",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/df56e0f08a5bb8d69bee5a922d38d2c0534d6e97"
        },
        "date": 1681335444173,
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
            "value": 66,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "u64/sub",
            "value": 13,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "u64/inv",
            "value": 482,
            "range": "± 31",
            "unit": "ns/iter"
          },
          {
            "name": "u64/div",
            "value": 507,
            "range": "± 30",
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
            "value": 3836,
            "range": "± 2257",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/evaluate_slice",
            "value": 254458,
            "range": "± 432501",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/add",
            "value": 3685,
            "range": "± 1098",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/neg",
            "value": 1051,
            "range": "± 433",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/sub",
            "value": 5087,
            "range": "± 1339",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/mul",
            "value": 3958,
            "range": "± 1118",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/div",
            "value": 4019,
            "range": "± 1172",
            "unit": "ns/iter"
          },
          {
            "name": "FFT/Sequential NR radix2 for 2^21 elements",
            "value": 3021176993,
            "range": "± 26965779",
            "unit": "ns/iter"
          },
          {
            "name": "FFT/Sequential RN radix2 for 2^21 elements",
            "value": 3481723122,
            "range": "± 21974960",
            "unit": "ns/iter"
          },
          {
            "name": "FFT/Sequential NR radix2 for 2^22 elements",
            "value": 6172091713,
            "range": "± 59776280",
            "unit": "ns/iter"
          },
          {
            "name": "FFT/Sequential RN radix2 for 2^22 elements",
            "value": 7421662656,
            "range": "± 124258226",
            "unit": "ns/iter"
          },
          {
            "name": "starks/simple_fibonacci",
            "value": 38518184,
            "range": "± 1399994",
            "unit": "ns/iter"
          },
          {
            "name": "starks/fibonacci_2_columns",
            "value": 55901667,
            "range": "± 1794322",
            "unit": "ns/iter"
          },
          {
            "name": "starks/fibonacci_f17",
            "value": 224785,
            "range": "± 8254",
            "unit": "ns/iter"
          },
          {
            "name": "starks/quadratic_air",
            "value": 39492737,
            "range": "± 1377132",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}