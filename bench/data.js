window.BENCHMARK_DATA = {
  "lastUpdate": 1698431171489,
  "repoUrl": "https://github.com/lambdaclass/lambdaworks",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "43053772+diegokingston@users.noreply.github.com",
            "name": "Diego K",
            "username": "diegokingston"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "a724038424ac5b7d615646ef927737148dc7206b",
          "message": "Update protocol.md (#624)\n\n* Update protocol.md\n\n* Update protocol.md\n\n* Update protocol.md\n\n* Update protocol.md\n\n* Update protocol.md\n\n* Update protocol.md\n\n* Update protocol.md\n\n* Update protocol.md\n\n* Update protocol.md",
          "timestamp": "2023-10-27T17:42:02Z",
          "tree_id": "c059c23f273b46a27ccd293121b31f4651b852ea",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/a724038424ac5b7d615646ef927737148dc7206b"
        },
        "date": 1698429972559,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 99266467,
            "range": "± 6089322",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 166108879,
            "range": "± 10361463",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 315883302,
            "range": "± 3670731",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 619120645,
            "range": "± 23313684",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 34752003,
            "range": "± 504602",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 69214818,
            "range": "± 582646",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 136640183,
            "range": "± 1697758",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 285523760,
            "range": "± 5467502",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 27385392,
            "range": "± 1085893",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 55801508,
            "range": "± 2540727",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 111288205,
            "range": "± 5720093",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 231165965,
            "range": "± 8616730",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 118209224,
            "range": "± 1157357",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 239354180,
            "range": "± 1019383",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 488057323,
            "range": "± 6919025",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 986546395,
            "range": "± 16390055",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 400667906,
            "range": "± 6506464",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 793358604,
            "range": "± 5347917",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1568446937,
            "range": "± 19237065",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3127816375,
            "range": "± 11716595",
            "unit": "ns/iter"
          }
        ]
      },
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
          "id": "6645ea62de31e4ee47f58232db92a1f08a10cd9c",
          "message": "Stark: Implement Stone compatible serialization (#625)\n\n* add serialization to stark proof and two tests\n\n* test pub\n\n* remove pub use\n\n* add grinding nonce to serialized proof\n\n* pow in little endian\n\n* nonce in be bytes\n\n* clippy, fmt\n\n* add code comment\n\n* fix position of nonce in proof\n\n* poc merge auth paths\n\n* fix state\n\n* add missing sym trace paths\n\n* fmt\n\n* clippy\n\n* add test with 3 queries\n\n* minor refactor\n\n* remove repeated query openings\n\n* fix repeated composition poly openings\n\n* minor change\n\n* use strong fiat shamir for fibonacci test cases\n\n* make public parameters input to serializer\n\n* make serialize method public\n\n* minor refactor\n\n* fmt\n\n* clippy\n\n* clippy\n\n* make serialize_proof pub\n\n* add docs\n\n* add docs\n\n* Add small comment\n\n* Add small comment\n\n* Add small comment\n\n* Update provers/stark/src/proof/stark.rs\n\n* Fmt\n\n---------\n\nCo-authored-by: Agustin <agustin@pop-os.localdomain>\nCo-authored-by: Mauro Toscano <12560266+MauroToscano@users.noreply.github.com>\nCo-authored-by: MauroFab <maurotoscano2@gmail.com>",
          "timestamp": "2023-10-27T17:47:00Z",
          "tree_id": "ce0c30d896ac1f347e47712c9bb829cb8ee72d77",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/6645ea62de31e4ee47f58232db92a1f08a10cd9c"
        },
        "date": 1698430355732,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 104924844,
            "range": "± 6015134",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 165028479,
            "range": "± 11603937",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 311115937,
            "range": "± 3837196",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 639613208,
            "range": "± 22817790",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 35325891,
            "range": "± 585157",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 69739085,
            "range": "± 956488",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 134156472,
            "range": "± 688184",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 285310104,
            "range": "± 3553699",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 27858462,
            "range": "± 1027361",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 51033271,
            "range": "± 3481346",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 105303529,
            "range": "± 1917699",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 212933673,
            "range": "± 6188992",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 118092299,
            "range": "± 639290",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 243781514,
            "range": "± 906786",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 499958666,
            "range": "± 8912867",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 979724104,
            "range": "± 19302589",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 410983447,
            "range": "± 2755532",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 798221000,
            "range": "± 7187535",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1576827708,
            "range": "± 12671043",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3145755458,
            "range": "± 16520936",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "43053772+diegokingston@users.noreply.github.com",
            "name": "Diego K",
            "username": "diegokingston"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "a724038424ac5b7d615646ef927737148dc7206b",
          "message": "Update protocol.md (#624)\n\n* Update protocol.md\n\n* Update protocol.md\n\n* Update protocol.md\n\n* Update protocol.md\n\n* Update protocol.md\n\n* Update protocol.md\n\n* Update protocol.md\n\n* Update protocol.md\n\n* Update protocol.md",
          "timestamp": "2023-10-27T17:42:02Z",
          "tree_id": "c059c23f273b46a27ccd293121b31f4651b852ea",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/a724038424ac5b7d615646ef927737148dc7206b"
        },
        "date": 1698431169567,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 918339689,
            "range": "± 823337",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 1923642171,
            "range": "± 22625870",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 1921036017,
            "range": "± 554417",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 4387368138,
            "range": "± 24481500",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 4014954829,
            "range": "± 9307874",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 10026293096,
            "range": "± 21966885",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 8367103062,
            "range": "± 1232607",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 21630008725,
            "range": "± 65827591",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 33958591,
            "range": "± 82255",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 33949998,
            "range": "± 183114",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 50767946,
            "range": "± 815415",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 50503905,
            "range": "± 1527007",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 67769727,
            "range": "± 385733",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 67808845,
            "range": "± 295611",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 112017715,
            "range": "± 1595249",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 111015284,
            "range": "± 2863585",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 135201756,
            "range": "± 381750",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 134928970,
            "range": "± 378071",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 230365948,
            "range": "± 276887",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 230326235,
            "range": "± 583368",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 268851113,
            "range": "± 720528",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 269053833,
            "range": "± 1092114",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 459017932,
            "range": "± 1031966",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 459227797,
            "range": "± 900554",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 44531956,
            "range": "± 358311",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 96106070,
            "range": "± 298119",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 191785939,
            "range": "± 747434",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 383984317,
            "range": "± 2724071",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 1051430966,
            "range": "± 2017305",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 2210506788,
            "range": "± 2988774",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 4586746841,
            "range": "± 4377196",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 9495394875,
            "range": "± 9547623",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 1117276507,
            "range": "± 3358923",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 2332043356,
            "range": "± 2742893",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 4841106172,
            "range": "± 2402004",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 10010523252,
            "range": "± 6410005",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 8,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 17,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 57,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 27,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 86,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 36,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 493,
            "range": "± 11",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 61,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 8,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}