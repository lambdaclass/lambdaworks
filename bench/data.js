window.BENCHMARK_DATA = {
  "lastUpdate": 1681730702085,
  "repoUrl": "https://github.com/lambdaclass/lambdaworks",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "41742639+schouhy@users.noreply.github.com",
            "name": "Sergio Chouhy",
            "username": "schouhy"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a5182dafd1dc61eeb810c25e8bea22ed7e410ff9",
          "message": "Add documentation for Plonk (#228)\n\n* add intro\r\n\r\n* expand on arithmetization\r\n\r\n* minor typos\r\n\r\n* expand on arithmetization\r\n\r\n* init implementation.md\r\n\r\n* rename\r\n\r\n* expand on arithmetization\r\n\r\n* continue arithmetization\r\n\r\n* from matrices to polynomials\r\n\r\n* fix expression\r\n\r\n* fix expression\r\n\r\n* explain copy constraints\r\n\r\n* fix index\r\n\r\n* fix typos\r\n\r\n* improve copy constraints explanation\r\n\r\n* minor improvements\r\n\r\n* add prover rounds\r\n\r\n* minor additions\r\n\r\n* fix indices\r\n\r\n* minor changes\r\n\r\n* remove astherisks\r\n\r\n* format\r\n\r\n* add verifier first version\r\n\r\n* improve verification section\r\n\r\n* add setup and blindings sections\r\n\r\n* add linearization trick section\r\n\r\n* format\r\n\r\n* complete sections\r\n\r\n* minor change\r\n\r\n* fix math\r\n\r\n* add links\r\n\r\n* expand usage and implementation details\r\n\r\n* minor changes\r\n\r\n* move protocol part to its own file. Improve implementation section\r\n\r\n* minor changes\r\n\r\n* add fiat-shamir section\r\n\r\n* correction\r\n\r\n* fix\r\n\r\n---------\r\n\r\nCo-authored-by: Federico Carrone <mail@fcarrone.com>",
          "timestamp": "2023-04-15T10:55:27+02:00",
          "tree_id": "e6354cddd455aeb795ad10b3d4489850ad396791",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/a5182dafd1dc61eeb810c25e8bea22ed7e410ff9"
        },
        "date": 1681549498564,
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
            "value": 39,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "u64/div",
            "value": 51,
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
            "value": 3200,
            "range": "± 1866",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/evaluate_slice",
            "value": 384997,
            "range": "± 357735",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/add",
            "value": 2933,
            "range": "± 844",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/neg",
            "value": 496,
            "range": "± 238",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/sub",
            "value": 3490,
            "range": "± 1120",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/mul",
            "value": 2951,
            "range": "± 867",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/div",
            "value": 3102,
            "range": "± 871",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 2026162827,
            "range": "± 2072253",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 4775885733,
            "range": "± 20153844",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 4239599060,
            "range": "± 2817662",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 10442090253,
            "range": "± 53016978",
            "unit": "ns/iter"
          },
          {
            "name": "starks/simple_fibonacci",
            "value": 32582977,
            "range": "± 15055",
            "unit": "ns/iter"
          },
          {
            "name": "starks/fibonacci_2_columns",
            "value": 46548705,
            "range": "± 11936",
            "unit": "ns/iter"
          },
          {
            "name": "starks/fibonacci_f17",
            "value": 188381,
            "range": "± 46",
            "unit": "ns/iter"
          },
          {
            "name": "starks/quadratic_air",
            "value": 33078080,
            "range": "± 10327",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "60488569+tcoratger@users.noreply.github.com",
            "name": "Thomas Coratger",
            "username": "tcoratger"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "4a6500a1a508f7c25c1f809279c3215acee8de66",
          "message": "Tests for Montgomery representative method (#196)\n\n* upstream update\n\n* tests for Montgomery representative\n\n* clean up",
          "timestamp": "2023-04-17T11:11:06Z",
          "tree_id": "58ee768c40f40184475520ddeee894636127e530",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/4a6500a1a508f7c25c1f809279c3215acee8de66"
        },
        "date": 1681730430525,
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
            "value": 270,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "u64/div",
            "value": 279,
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
            "value": 2170,
            "range": "± 1290",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/evaluate_slice",
            "value": 249168,
            "range": "± 246805",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/add",
            "value": 1594,
            "range": "± 385",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/neg",
            "value": 752,
            "range": "± 230",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/sub",
            "value": 2275,
            "range": "± 528",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/mul",
            "value": 1659,
            "range": "± 442",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/div",
            "value": 1723,
            "range": "± 414",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 2259951559,
            "range": "± 2611142",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 3913566558,
            "range": "± 29613956",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 4719331292,
            "range": "± 2068444",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 8770122950,
            "range": "± 18017435",
            "unit": "ns/iter"
          },
          {
            "name": "starks/simple_fibonacci",
            "value": 32663299,
            "range": "± 19726",
            "unit": "ns/iter"
          },
          {
            "name": "starks/fibonacci_2_columns",
            "value": 46631126,
            "range": "± 21506",
            "unit": "ns/iter"
          },
          {
            "name": "starks/fibonacci_f17",
            "value": 156991,
            "range": "± 52",
            "unit": "ns/iter"
          },
          {
            "name": "starks/quadratic_air",
            "value": 33155337,
            "range": "± 13913",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "60488569+tcoratger@users.noreply.github.com",
            "name": "Thomas Coratger",
            "username": "tcoratger"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "be6b800cc339582041890886943ee174476a4450",
          "message": "Change actions-rs in CI for dtolnay/rust-toolchain@stable (#222)\n\n* upstream update\n\n* migrate to dtolnay/rust-toolchain\n\n* clean up",
          "timestamp": "2023-04-17T11:15:30Z",
          "tree_id": "635dc4878959dbef734592d34cf8e68bf50c4e08",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/be6b800cc339582041890886943ee174476a4450"
        },
        "date": 1681730701234,
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
            "value": 375,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "u64/div",
            "value": 386,
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
            "value": 2729,
            "range": "± 1674",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/evaluate_slice",
            "value": 372603,
            "range": "± 366067",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/add",
            "value": 2935,
            "range": "± 757",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/neg",
            "value": 761,
            "range": "± 236",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/sub",
            "value": 3437,
            "range": "± 1066",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/mul",
            "value": 2804,
            "range": "± 889",
            "unit": "ns/iter"
          },
          {
            "name": "polynomial/div",
            "value": 2802,
            "range": "± 800",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 2029376888,
            "range": "± 2377619",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 4664076536,
            "range": "± 13470203",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 4246429616,
            "range": "± 9140968",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 10174876303,
            "range": "± 22081027",
            "unit": "ns/iter"
          },
          {
            "name": "starks/simple_fibonacci",
            "value": 32586018,
            "range": "± 8694",
            "unit": "ns/iter"
          },
          {
            "name": "starks/fibonacci_2_columns",
            "value": 46571429,
            "range": "± 211317",
            "unit": "ns/iter"
          },
          {
            "name": "starks/fibonacci_f17",
            "value": 187727,
            "range": "± 68",
            "unit": "ns/iter"
          },
          {
            "name": "starks/quadratic_air",
            "value": 33086731,
            "range": "± 12712",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}