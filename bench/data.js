window.BENCHMARK_DATA = {
  "lastUpdate": 1698437491878,
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
        "date": 1698431907747,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 1003524406,
            "range": "± 1472288",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 3214312916,
            "range": "± 11698637",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 2098036495,
            "range": "± 1683405",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 7032994883,
            "range": "± 13364360",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 4385247159,
            "range": "± 2869491",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 15182378392,
            "range": "± 30383002",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 9140238954,
            "range": "± 5921424",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 32213508505,
            "range": "± 124424667",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 35527660,
            "range": "± 97736",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 35928145,
            "range": "± 208017",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 54772384,
            "range": "± 908258",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 54585484,
            "range": "± 1245992",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 72715938,
            "range": "± 255696",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 72184823,
            "range": "± 294155",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 121387348,
            "range": "± 1149269",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 124515076,
            "range": "± 1789874",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 144659707,
            "range": "± 138782",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 144574107,
            "range": "± 218484",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 256692129,
            "range": "± 688404",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 257422270,
            "range": "± 633310",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 288622294,
            "range": "± 277527",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 288810527,
            "range": "± 267594",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 521064203,
            "range": "± 858924",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 512180986,
            "range": "± 1245190",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 56471248,
            "range": "± 180615",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 117517936,
            "range": "± 221388",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 249383477,
            "range": "± 263864",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 465978110,
            "range": "± 1046444",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 1175262773,
            "range": "± 1988749",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 2468784203,
            "range": "± 3531943",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 5134360927,
            "range": "± 3281139",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 10594338090,
            "range": "± 28969988",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 1254676656,
            "range": "± 3177280",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 2612518613,
            "range": "± 4459265",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 5422731361,
            "range": "± 3469023",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 11207476415,
            "range": "± 15542744",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 1850,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 118358,
            "range": "± 80",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 1351,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 412,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 1916,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 104062,
            "range": "± 70",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 4602,
            "range": "± 1644",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 241647,
            "range": "± 877",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 1872,
            "range": "± 2",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ajgarassino@gmail.com",
            "name": "ajgara",
            "username": "ajgara"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "a653c5adab9f67f2fe8fd1b74478cdd927359945",
          "message": "Update docs stone compatibility (#635)\n\n* add serialization to stark proof and two tests\n\n* test pub\n\n* remove pub use\n\n* add grinding nonce to serialized proof\n\n* pow in little endian\n\n* nonce in be bytes\n\n* clippy, fmt\n\n* add code comment\n\n* fix position of nonce in proof\n\n* poc merge auth paths\n\n* fix state\n\n* add missing sym trace paths\n\n* fmt\n\n* clippy\n\n* add test with 3 queries\n\n* minor refactor\n\n* remove repeated query openings\n\n* fix repeated composition poly openings\n\n* minor change\n\n* use strong fiat shamir for fibonacci test cases\n\n* make public parameters input to serializer\n\n* make serialize method public\n\n* minor refactor\n\n* fmt\n\n* clippy\n\n* clippy\n\n* make serialize_proof pub\n\n* add docs\n\n* add docs\n\n* Add small comment\n\n* Add small comment\n\n* Add small comment\n\n* Add small comment\n\n* Update provers/stark/src/proof/stark.rs\n\n* Fmt\n\n* Add small comment\n\n* remove unused function\n\n* minor change in docs\n\n* minor change\n\n* minor change\n\n* add code comments\n\n* fix code comment\n\n* change struct docs\n\n* add code comment\n\n* remove unnecessary allow(unused)\n\n---------\n\nCo-authored-by: Sergio Chouhy <sergio.chouhy@gmail.com>\nCo-authored-by: Sergio Chouhy <41742639+schouhy@users.noreply.github.com>\nCo-authored-by: Agustin <agustin@pop-os.localdomain>\nCo-authored-by: Mauro Toscano <12560266+MauroToscano@users.noreply.github.com>\nCo-authored-by: MauroFab <maurotoscano2@gmail.com>",
          "timestamp": "2023-10-27T19:16:13Z",
          "tree_id": "b05d22a9b8ea8e70e0aea0721125e52c4e3ed973",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/a653c5adab9f67f2fe8fd1b74478cdd927359945"
        },
        "date": 1698435781252,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 101224863,
            "range": "± 7761522",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 163141391,
            "range": "± 5075345",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 318407010,
            "range": "± 3217838",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 622089437,
            "range": "± 19455391",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 35001144,
            "range": "± 209720",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 69281332,
            "range": "± 513717",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 137108808,
            "range": "± 1451799",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 284014239,
            "range": "± 4159254",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 30090482,
            "range": "± 239795",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 57129752,
            "range": "± 742777",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 114447725,
            "range": "± 3379113",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 235865659,
            "range": "± 9106650",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 118682799,
            "range": "± 1047864",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 238230055,
            "range": "± 815018",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 474109208,
            "range": "± 14000559",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 988255104,
            "range": "± 27101531",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 408002156,
            "range": "± 8484362",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 820427541,
            "range": "± 12731684",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1647909562,
            "range": "± 32257360",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3173461875,
            "range": "± 95724205",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "ajgarassino@gmail.com",
            "name": "ajgara",
            "username": "ajgara"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "a653c5adab9f67f2fe8fd1b74478cdd927359945",
          "message": "Update docs stone compatibility (#635)\n\n* add serialization to stark proof and two tests\n\n* test pub\n\n* remove pub use\n\n* add grinding nonce to serialized proof\n\n* pow in little endian\n\n* nonce in be bytes\n\n* clippy, fmt\n\n* add code comment\n\n* fix position of nonce in proof\n\n* poc merge auth paths\n\n* fix state\n\n* add missing sym trace paths\n\n* fmt\n\n* clippy\n\n* add test with 3 queries\n\n* minor refactor\n\n* remove repeated query openings\n\n* fix repeated composition poly openings\n\n* minor change\n\n* use strong fiat shamir for fibonacci test cases\n\n* make public parameters input to serializer\n\n* make serialize method public\n\n* minor refactor\n\n* fmt\n\n* clippy\n\n* clippy\n\n* make serialize_proof pub\n\n* add docs\n\n* add docs\n\n* Add small comment\n\n* Add small comment\n\n* Add small comment\n\n* Add small comment\n\n* Update provers/stark/src/proof/stark.rs\n\n* Fmt\n\n* Add small comment\n\n* remove unused function\n\n* minor change in docs\n\n* minor change\n\n* minor change\n\n* add code comments\n\n* fix code comment\n\n* change struct docs\n\n* add code comment\n\n* remove unnecessary allow(unused)\n\n---------\n\nCo-authored-by: Sergio Chouhy <sergio.chouhy@gmail.com>\nCo-authored-by: Sergio Chouhy <41742639+schouhy@users.noreply.github.com>\nCo-authored-by: Agustin <agustin@pop-os.localdomain>\nCo-authored-by: Mauro Toscano <12560266+MauroToscano@users.noreply.github.com>\nCo-authored-by: MauroFab <maurotoscano2@gmail.com>",
          "timestamp": "2023-10-27T19:16:13Z",
          "tree_id": "b05d22a9b8ea8e70e0aea0721125e52c4e3ed973",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/a653c5adab9f67f2fe8fd1b74478cdd927359945"
        },
        "date": 1698437489738,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 1071357719,
            "range": "± 17849520",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 3530743018,
            "range": "± 27957940",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 2266451044,
            "range": "± 23364551",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 7801361883,
            "range": "± 65682864",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 4680185497,
            "range": "± 29960464",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 16826238299,
            "range": "± 120018738",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 9764925412,
            "range": "± 108530097",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 36069912751,
            "range": "± 268988588",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 37932858,
            "range": "± 852645",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 38563522,
            "range": "± 1147593",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 63456247,
            "range": "± 2795158",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 63746718,
            "range": "± 3711198",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 75662348,
            "range": "± 1760069",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 76843341,
            "range": "± 905030",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 138632066,
            "range": "± 2804152",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 136301468,
            "range": "± 4424302",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 156474032,
            "range": "± 3672862",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 153780415,
            "range": "± 2064292",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 285080273,
            "range": "± 5262402",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 283032467,
            "range": "± 2751663",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 306479827,
            "range": "± 5133458",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 309673370,
            "range": "± 10712978",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 571871355,
            "range": "± 9450113",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 593962726,
            "range": "± 14742326",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 64742663,
            "range": "± 1923586",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 133746152,
            "range": "± 2569081",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 270561034,
            "range": "± 6092874",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 548286510,
            "range": "± 13709359",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 1324773015,
            "range": "± 28418309",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 2755914120,
            "range": "± 51657126",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 5733702039,
            "range": "± 67605827",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 11919557883,
            "range": "± 111937346",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 1416462912,
            "range": "± 22882342",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 2948003842,
            "range": "± 66195295",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 6067683420,
            "range": "± 75504940",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 12530913344,
            "range": "± 272947071",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 919,
            "range": "± 47",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 29565,
            "range": "± 1502",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 865,
            "range": "± 32",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 319,
            "range": "± 12",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 1211,
            "range": "± 54",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 28091,
            "range": "± 949",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 3042,
            "range": "± 644",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 75348,
            "range": "± 6490",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 954,
            "range": "± 63",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}