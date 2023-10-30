window.BENCHMARK_DATA = {
  "lastUpdate": 1698697304554,
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
      },
      {
        "commit": {
          "author": {
            "email": "12560266+MauroToscano@users.noreply.github.com",
            "name": "Mauro Toscano",
            "username": "MauroToscano"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "6f7d87977fb51208bd9fd24dce142a202b2a04cc",
          "message": "Add submodule with stone fork to cross verify (#637)\n\n* Add submodule with stone fork\n\n* Improved docs\n\n* Better README\n\n* Update provers/stark/README.md\n\n---------\n\nCo-authored-by: Mariano A. Nicolini <mariano.nicolini.91@gmail.com>",
          "timestamp": "2023-10-27T20:32:52Z",
          "tree_id": "f6f4be0b0a35d3331712dcfbd91b0476643c7044",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/6f7d87977fb51208bd9fd24dce142a202b2a04cc"
        },
        "date": 1698440175127,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 98816604,
            "range": "± 6844361",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 164868438,
            "range": "± 5677332",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 314657968,
            "range": "± 1859934",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 633158833,
            "range": "± 16791753",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 34941006,
            "range": "± 389814",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 69323843,
            "range": "± 586886",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 137102255,
            "range": "± 1397626",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 283211406,
            "range": "± 2465728",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 27009797,
            "range": "± 392570",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 55583256,
            "range": "± 2183893",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 111930230,
            "range": "± 4617658",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 240880138,
            "range": "± 8893721",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 115641374,
            "range": "± 989450",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 241779368,
            "range": "± 1596924",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 498880927,
            "range": "± 5419651",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 982236271,
            "range": "± 19079083",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 409072583,
            "range": "± 5469653",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 793587437,
            "range": "± 6297722",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1582933792,
            "range": "± 7800968",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3131230250,
            "range": "± 15822522",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "12560266+MauroToscano@users.noreply.github.com",
            "name": "Mauro Toscano",
            "username": "MauroToscano"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "6f7d87977fb51208bd9fd24dce142a202b2a04cc",
          "message": "Add submodule with stone fork to cross verify (#637)\n\n* Add submodule with stone fork\n\n* Improved docs\n\n* Better README\n\n* Update provers/stark/README.md\n\n---------\n\nCo-authored-by: Mariano A. Nicolini <mariano.nicolini.91@gmail.com>",
          "timestamp": "2023-10-27T20:32:52Z",
          "tree_id": "f6f4be0b0a35d3331712dcfbd91b0476643c7044",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/6f7d87977fb51208bd9fd24dce142a202b2a04cc"
        },
        "date": 1698441785007,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 964942502,
            "range": "± 30975354",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 3543268121,
            "range": "± 29921000",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 2100215801,
            "range": "± 47799320",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 7765547372,
            "range": "± 34605540",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 4148932540,
            "range": "± 55321939",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 16878465151,
            "range": "± 134911950",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 9159792546,
            "range": "± 203281017",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 36043458171,
            "range": "± 255723937",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 32539604,
            "range": "± 1302223",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 33760011,
            "range": "± 1747600",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 53045185,
            "range": "± 4457449",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 55142242,
            "range": "± 4462967",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 73061218,
            "range": "± 2399424",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 71147154,
            "range": "± 1385547",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 120342018,
            "range": "± 4452763",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 124404416,
            "range": "± 3301152",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 139972220,
            "range": "± 3516855",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 144044389,
            "range": "± 4690521",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 264364393,
            "range": "± 4597179",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 264570622,
            "range": "± 5328112",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 286816553,
            "range": "± 4657932",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 277529521,
            "range": "± 7440906",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 525566709,
            "range": "± 7462598",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 555926746,
            "range": "± 30564190",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 60620205,
            "range": "± 2908332",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 131918831,
            "range": "± 2991060",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 264149696,
            "range": "± 3744932",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 532131081,
            "range": "± 10867445",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 1086129859,
            "range": "± 19694594",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 2395917312,
            "range": "± 59958666",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 5056175194,
            "range": "± 65746882",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 10662376626,
            "range": "± 192353168",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 1275833613,
            "range": "± 30552611",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 2627289964,
            "range": "± 54662395",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 5534335352,
            "range": "± 73847601",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 11447716599,
            "range": "± 212006305",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 1744,
            "range": "± 107",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 110310,
            "range": "± 5753",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 1356,
            "range": "± 74",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 460,
            "range": "± 25",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 1828,
            "range": "± 107",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 105295,
            "range": "± 5020",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 4582,
            "range": "± 1936",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 235457,
            "range": "± 17625",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 1690,
            "range": "± 144",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "12560266+MauroToscano@users.noreply.github.com",
            "name": "Mauro Toscano",
            "username": "MauroToscano"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "5c81dc4efc54e43a2dd419bd63aae0127c496238",
          "message": "Update (#640)",
          "timestamp": "2023-10-27T21:59:54Z",
          "tree_id": "7e1b87337909772ab39c2c1d6b6c4aa020f0172f",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/5c81dc4efc54e43a2dd419bd63aae0127c496238"
        },
        "date": 1698445277879,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 102617596,
            "range": "± 6841982",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 167654114,
            "range": "± 12119501",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 315601031,
            "range": "± 3539800",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 626929374,
            "range": "± 28530610",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 35064912,
            "range": "± 1234800",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 68893854,
            "range": "± 1203941",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 136241162,
            "range": "± 2135758",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 284272052,
            "range": "± 3201402",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 27105260,
            "range": "± 960563",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 54035424,
            "range": "± 2674360",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 107250793,
            "range": "± 3701295",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 240561569,
            "range": "± 7501028",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 118751288,
            "range": "± 1051968",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 239343187,
            "range": "± 1776952",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 489840218,
            "range": "± 2550104",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 987286792,
            "range": "± 15844433",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 401067958,
            "range": "± 6347121",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 797410062,
            "range": "± 5896296",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1561524708,
            "range": "± 12527817",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3136066834,
            "range": "± 28922798",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "12560266+MauroToscano@users.noreply.github.com",
            "name": "Mauro Toscano",
            "username": "MauroToscano"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "5c81dc4efc54e43a2dd419bd63aae0127c496238",
          "message": "Update (#640)",
          "timestamp": "2023-10-27T21:59:54Z",
          "tree_id": "7e1b87337909772ab39c2c1d6b6c4aa020f0172f",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/5c81dc4efc54e43a2dd419bd63aae0127c496238"
        },
        "date": 1698446868110,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 1034639329,
            "range": "± 23041367",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 3530157543,
            "range": "± 87293807",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 2158752520,
            "range": "± 39194546",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 7750902216,
            "range": "± 169940557",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 4489536483,
            "range": "± 120902032",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 16604870785,
            "range": "± 233510493",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 8485924303,
            "range": "± 207887198",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 35281632427,
            "range": "± 293096528",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 32971729,
            "range": "± 1243387",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 32923521,
            "range": "± 993051",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 58578946,
            "range": "± 2191804",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 60947374,
            "range": "± 3027207",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 66228208,
            "range": "± 1752500",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 68606099,
            "range": "± 2684357",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 129486027,
            "range": "± 3064359",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 126689619,
            "range": "± 1547586",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 140136746,
            "range": "± 4982923",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 134040666,
            "range": "± 2576037",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 264979976,
            "range": "± 5951672",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 262673139,
            "range": "± 6281207",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 267902836,
            "range": "± 6060720",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 285018755,
            "range": "± 14768732",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 534054125,
            "range": "± 18048480",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 532075282,
            "range": "± 16938848",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 59541814,
            "range": "± 2867795",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 133820735,
            "range": "± 1717401",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 266007348,
            "range": "± 4404996",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 526718884,
            "range": "± 9790520",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 1113091183,
            "range": "± 31359810",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 2377100801,
            "range": "± 89205209",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 5140927875,
            "range": "± 187077475",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 10731521551,
            "range": "± 258409148",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 1263117960,
            "range": "± 17404165",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 2707404984,
            "range": "± 79465210",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 5766636054,
            "range": "± 252785703",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 11195327820,
            "range": "± 229507261",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 402,
            "range": "± 28",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 6517,
            "range": "± 359",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 473,
            "range": "± 27",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 229,
            "range": "± 14",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 672,
            "range": "± 34",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 6499,
            "range": "± 556",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 1743,
            "range": "± 93",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 20472,
            "range": "± 1557",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 402,
            "range": "± 25",
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
          "distinct": true,
          "id": "f95a691a7fbef700e208ef34bdc1a608005182b0",
          "message": "Update README.md (#636)",
          "timestamp": "2023-10-30T13:14:10Z",
          "tree_id": "56eb33d362800cc756e36d31c0dc31235bfeceaf",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/f95a691a7fbef700e208ef34bdc1a608005182b0"
        },
        "date": 1698672967164,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 127899305,
            "range": "± 8887266",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 206071062,
            "range": "± 5285893",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 438508583,
            "range": "± 47825522",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 790229458,
            "range": "± 24095879",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 43871500,
            "range": "± 2563391",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 81828420,
            "range": "± 2225840",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 163922317,
            "range": "± 5866937",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 317793187,
            "range": "± 8977247",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 39938214,
            "range": "± 2542207",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 76539574,
            "range": "± 13918592",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 165258395,
            "range": "± 14637264",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 285495791,
            "range": "± 8793934",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 148282966,
            "range": "± 5014320",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 292455656,
            "range": "± 16151295",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 580862624,
            "range": "± 13305157",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1129877145,
            "range": "± 46793142",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 475253802,
            "range": "± 21270042",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 931538458,
            "range": "± 56121641",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1819002146,
            "range": "± 33349470",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3309159729,
            "range": "± 160731974",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "96737978+feltroidprime@users.noreply.github.com",
            "name": "feltroid Prime",
            "username": "feltroidprime"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "56d6054599a84c712514dc089418ca7e74d94f39",
          "message": "Starknet Poseidon Hash & Starknet Poseidon Merkle Tree backends (#641)\n\n* starknet poseidon hash\n\n* starknet poseidon merkle tree backends",
          "timestamp": "2023-10-30T13:31:27Z",
          "tree_id": "7cc117463415d7f13481fbd9a314a92d8f803e8b",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/56d6054599a84c712514dc089418ca7e74d94f39"
        },
        "date": 1698673914936,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 101402659,
            "range": "± 5144235",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 169900678,
            "range": "± 12806486",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 320120406,
            "range": "± 5412947",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 639682458,
            "range": "± 30180640",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 35073212,
            "range": "± 464652",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 69491070,
            "range": "± 623009",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 136800003,
            "range": "± 1929239",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 282802625,
            "range": "± 3291215",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 26173549,
            "range": "± 907564",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 54016455,
            "range": "± 2705398",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 113094284,
            "range": "± 2832525",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 238936402,
            "range": "± 10762282",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 118607492,
            "range": "± 891998",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 242587284,
            "range": "± 1230735",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 496503239,
            "range": "± 4477427",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 986822833,
            "range": "± 9419715",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 408919739,
            "range": "± 3483432",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 796334646,
            "range": "± 5813256",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1583375645,
            "range": "± 11238356",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3155670458,
            "range": "± 23732021",
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
          "distinct": true,
          "id": "f95a691a7fbef700e208ef34bdc1a608005182b0",
          "message": "Update README.md (#636)",
          "timestamp": "2023-10-30T13:14:10Z",
          "tree_id": "56eb33d362800cc756e36d31c0dc31235bfeceaf",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/f95a691a7fbef700e208ef34bdc1a608005182b0"
        },
        "date": 1698674246525,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 850080652,
            "range": "± 2741485",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 3048242684,
            "range": "± 12781202",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 1778937000,
            "range": "± 5939695",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 6694717618,
            "range": "± 34362182",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 3711687080,
            "range": "± 867554",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 14475777200,
            "range": "± 43739396",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 7734705637,
            "range": "± 2169140",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 31444629978,
            "range": "± 32835769",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 31070853,
            "range": "± 115474",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 31064045,
            "range": "± 51512",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 54991825,
            "range": "± 381224",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 54904038,
            "range": "± 798692",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 62348703,
            "range": "± 227682",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 62526081,
            "range": "± 77177",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 122330413,
            "range": "± 397824",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 122324898,
            "range": "± 764861",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 125655701,
            "range": "± 167965",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 125445737,
            "range": "± 535394",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 252780632,
            "range": "± 664395",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 252214280,
            "range": "± 378298",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 250368622,
            "range": "± 260332",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 250772831,
            "range": "± 246594",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 504389469,
            "range": "± 1482382",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 504812472,
            "range": "± 1254024",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 61746303,
            "range": "± 418616",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 125575007,
            "range": "± 952167",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 263274347,
            "range": "± 320515",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 535273547,
            "range": "± 3221505",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 1029842530,
            "range": "± 2209997",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 2150448100,
            "range": "± 1666694",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 4461920785,
            "range": "± 2940937",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 9244731436,
            "range": "± 6373452",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 1096972585,
            "range": "± 2025768",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 2282350912,
            "range": "± 989066",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 4733479065,
            "range": "± 3108679",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 9768070047,
            "range": "± 6096786",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 1541,
            "range": "± 15",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 98476,
            "range": "± 350",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 1151,
            "range": "± 25",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 394,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 1573,
            "range": "± 29",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 85141,
            "range": "± 192",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 3806,
            "range": "± 1561",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 197575,
            "range": "± 3283",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 1561,
            "range": "± 5",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "96737978+feltroidprime@users.noreply.github.com",
            "name": "feltroid Prime",
            "username": "feltroidprime"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "56d6054599a84c712514dc089418ca7e74d94f39",
          "message": "Starknet Poseidon Hash & Starknet Poseidon Merkle Tree backends (#641)\n\n* starknet poseidon hash\n\n* starknet poseidon merkle tree backends",
          "timestamp": "2023-10-30T13:31:27Z",
          "tree_id": "7cc117463415d7f13481fbd9a314a92d8f803e8b",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/56d6054599a84c712514dc089418ca7e74d94f39"
        },
        "date": 1698675456477,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 998063010,
            "range": "± 2998264",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 3133688626,
            "range": "± 9473732",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 2097924672,
            "range": "± 4782079",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 6952225499,
            "range": "± 32319305",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 4391014884,
            "range": "± 7197057",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 14961473822,
            "range": "± 22322019",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 9119020978,
            "range": "± 16199061",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 32107745888,
            "range": "± 83524099",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 36475200,
            "range": "± 164517",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 36291605,
            "range": "± 245467",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 58763008,
            "range": "± 949957",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 58420740,
            "range": "± 1488015",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 72466468,
            "range": "± 518198",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 72661793,
            "range": "± 323064",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 127918243,
            "range": "± 1179939",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 128366579,
            "range": "± 1093101",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 145533385,
            "range": "± 346702",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 145694227,
            "range": "± 471196",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 265603178,
            "range": "± 1171847",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 266173429,
            "range": "± 848424",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 290543620,
            "range": "± 845700",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 291893575,
            "range": "± 1160538",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 527151700,
            "range": "± 2083149",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 529829778,
            "range": "± 2587953",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 59580143,
            "range": "± 326639",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 120264382,
            "range": "± 590523",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 251430030,
            "range": "± 1219803",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 512257308,
            "range": "± 1166425",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 1186648379,
            "range": "± 3704137",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 2475540132,
            "range": "± 3591700",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 5138941063,
            "range": "± 6524109",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 10683922953,
            "range": "± 5941727",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 1261074320,
            "range": "± 2107129",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 2622247346,
            "range": "± 3389129",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 5427889003,
            "range": "± 12651491",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 11269267109,
            "range": "± 17871489",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 451,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 7248,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 459,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 235,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 707,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 6822,
            "range": "± 5",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 2212,
            "range": "± 250",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 21434,
            "range": "± 115",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 459,
            "range": "± 1",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "62400508+juan518munoz@users.noreply.github.com",
            "name": "juan518munoz",
            "username": "juan518munoz"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "14ae37148ce559d886378f3fd4e6b38fe43c84da",
          "message": "Merkle cli (#618)\n\n* initial commit\n\n* basic generate tree func\n\n* fun declarations\n\n* comments, steps of each func\n\n* err handling\n\n* format & argument use\n\n* serde incl\n\n* remove wrong deserialize\n\n* remove useless imports\n\n* updated toml\n\n* working verify command\n\n* fmt & better output\n\n* serialize merkle tree to json\n\n* fmt\n\n* fix cargo toml\n\n* update README\n\n* add clap4 to references\n\n* add rest of references to README\n\n* add release flag to readme\n\n* colored verifycation output\n\n* move merkle-cli to examples dir\n\n* update main README\n\n* update merkle tree cli README\n\n* renamed command, better README\n\n* wrap gen proof inside result\n\n* fix get_proof_by_pos\n\n* remove newline when reading leaf\n\n---------\n\nCo-authored-by: Jmunoz <juan.munoz@lambdaclass.com>\nCo-authored-by: Mariano A. Nicolini <mariano.nicolini.91@gmail.com>",
          "timestamp": "2023-10-30T14:49:44Z",
          "tree_id": "f0f0eae950d359161309a1d30d66f9c9ba4aae3e",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/14ae37148ce559d886378f3fd4e6b38fe43c84da"
        },
        "date": 1698679239502,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 113972012,
            "range": "± 7441110",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 220116291,
            "range": "± 16743007",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 444563115,
            "range": "± 19205799",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 800108562,
            "range": "± 32625548",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 44561697,
            "range": "± 2057070",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 93001260,
            "range": "± 6805601",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 166326277,
            "range": "± 5862011",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 328701927,
            "range": "± 10923241",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 51662850,
            "range": "± 6069067",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 91781510,
            "range": "± 16789721",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 141900929,
            "range": "± 9690690",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 288868718,
            "range": "± 13455045",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 159175537,
            "range": "± 9076329",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 314178271,
            "range": "± 14506671",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 658982895,
            "range": "± 27449451",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1359773875,
            "range": "± 77916498",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 498057021,
            "range": "± 34135210",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 945411146,
            "range": "± 25584468",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1714795229,
            "range": "± 41909401",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3961287166,
            "range": "± 160478257",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "62400508+juan518munoz@users.noreply.github.com",
            "name": "juan518munoz",
            "username": "juan518munoz"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "14ae37148ce559d886378f3fd4e6b38fe43c84da",
          "message": "Merkle cli (#618)\n\n* initial commit\n\n* basic generate tree func\n\n* fun declarations\n\n* comments, steps of each func\n\n* err handling\n\n* format & argument use\n\n* serde incl\n\n* remove wrong deserialize\n\n* remove useless imports\n\n* updated toml\n\n* working verify command\n\n* fmt & better output\n\n* serialize merkle tree to json\n\n* fmt\n\n* fix cargo toml\n\n* update README\n\n* add clap4 to references\n\n* add rest of references to README\n\n* add release flag to readme\n\n* colored verifycation output\n\n* move merkle-cli to examples dir\n\n* update main README\n\n* update merkle tree cli README\n\n* renamed command, better README\n\n* wrap gen proof inside result\n\n* fix get_proof_by_pos\n\n* remove newline when reading leaf\n\n---------\n\nCo-authored-by: Jmunoz <juan.munoz@lambdaclass.com>\nCo-authored-by: Mariano A. Nicolini <mariano.nicolini.91@gmail.com>",
          "timestamp": "2023-10-30T14:49:44Z",
          "tree_id": "f0f0eae950d359161309a1d30d66f9c9ba4aae3e",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/14ae37148ce559d886378f3fd4e6b38fe43c84da"
        },
        "date": 1698680554301,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 1007549237,
            "range": "± 2809154",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 3192660117,
            "range": "± 22554170",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 2105601327,
            "range": "± 1391922",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 6987573380,
            "range": "± 29159598",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 4394446952,
            "range": "± 3166454",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 15191052462,
            "range": "± 37386108",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 9162940841,
            "range": "± 5444169",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 32116711382,
            "range": "± 99395594",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 36625871,
            "range": "± 116575",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 36683536,
            "range": "± 66751",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 62290857,
            "range": "± 607908",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 62640887,
            "range": "± 979707",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 73319611,
            "range": "± 202138",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 73608989,
            "range": "± 140866",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 133808132,
            "range": "± 831387",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 133952711,
            "range": "± 1502771",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 147220015,
            "range": "± 211376",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 146288399,
            "range": "± 245667",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 269923550,
            "range": "± 3900431",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 271254258,
            "range": "± 1138804",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 293348668,
            "range": "± 402154",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 293104804,
            "range": "± 404350",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 537191930,
            "range": "± 1252941",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 534653150,
            "range": "± 1507911",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 59416806,
            "range": "± 253268",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 123905835,
            "range": "± 1033359",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 254480438,
            "range": "± 1381436",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 521561102,
            "range": "± 2339392",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 1191654638,
            "range": "± 1976236",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 2490690311,
            "range": "± 2229711",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 5174264119,
            "range": "± 2599343",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 10706489905,
            "range": "± 14321329",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 1262833478,
            "range": "± 2636276",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 2637736712,
            "range": "± 3189279",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 5465936715,
            "range": "± 7919068",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 11301485281,
            "range": "± 14484511",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 97,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 409,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 137,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 45,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 185,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 554,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 1030,
            "range": "± 9",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 1995,
            "range": "± 34",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 101,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "12560266+MauroToscano@users.noreply.github.com",
            "name": "Mauro Toscano",
            "username": "MauroToscano"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "e625599b5dd80467b68be25f5867c97a43d34f0e",
          "message": "Update CI (#643)",
          "timestamp": "2023-10-30T16:31:57Z",
          "tree_id": "ee52d4f14a6146e29680cbb51a5b4d9f58e08794",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/e625599b5dd80467b68be25f5867c97a43d34f0e"
        },
        "date": 1698684740271,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 102714171,
            "range": "± 9908532",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 231931426,
            "range": "± 18216904",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 411940187,
            "range": "± 20747194",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 868952458,
            "range": "± 71125707",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 45930463,
            "range": "± 4204161",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 99499400,
            "range": "± 4321593",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 192518382,
            "range": "± 9265277",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 349493917,
            "range": "± 20406025",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 53827362,
            "range": "± 8241573",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 88204639,
            "range": "± 9967162",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 143906041,
            "range": "± 6540868",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 266251093,
            "range": "± 29462647",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 145486549,
            "range": "± 5054385",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 303839718,
            "range": "± 31638343",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 567014604,
            "range": "± 22720971",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1178509375,
            "range": "± 25656477",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 451317479,
            "range": "± 20764062",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 892644063,
            "range": "± 58214146",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1931568312,
            "range": "± 98486309",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3875872208,
            "range": "± 70748039",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "12560266+MauroToscano@users.noreply.github.com",
            "name": "Mauro Toscano",
            "username": "MauroToscano"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "e625599b5dd80467b68be25f5867c97a43d34f0e",
          "message": "Update CI (#643)",
          "timestamp": "2023-10-30T16:31:57Z",
          "tree_id": "ee52d4f14a6146e29680cbb51a5b4d9f58e08794",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/e625599b5dd80467b68be25f5867c97a43d34f0e"
        },
        "date": 1698685819027,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 916880083,
            "range": "± 1005273",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 1862379716,
            "range": "± 17861633",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 1919544268,
            "range": "± 6926167",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 4457414595,
            "range": "± 30227708",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 4008406764,
            "range": "± 956673",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 9792992695,
            "range": "± 25263953",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 8359341775,
            "range": "± 2891586",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 21583324192,
            "range": "± 97736128",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 34520932,
            "range": "± 136163",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 34404266,
            "range": "± 115856",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 56300138,
            "range": "± 602810",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 56466672,
            "range": "± 476628",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 68584508,
            "range": "± 102677",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 68516040,
            "range": "± 214462",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 113139880,
            "range": "± 1126666",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 113492962,
            "range": "± 1093243",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 136433720,
            "range": "± 411157",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 136293687,
            "range": "± 925536",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 230166520,
            "range": "± 5082836",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 231641785,
            "range": "± 985930",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 270505067,
            "range": "± 1294734",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 269350057,
            "range": "± 1007778",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 456981185,
            "range": "± 1764745",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 456839995,
            "range": "± 1962709",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 45279874,
            "range": "± 328024",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 95149185,
            "range": "± 337543",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 192356373,
            "range": "± 1785801",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 383062103,
            "range": "± 1871896",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 1062766902,
            "range": "± 1625733",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 2217070301,
            "range": "± 4180673",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 4602540032,
            "range": "± 2423139",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 9531303767,
            "range": "± 6597195",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 1120088565,
            "range": "± 3519097",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 2342814142,
            "range": "± 4131481",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 4860071201,
            "range": "± 13900670",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 10049070135,
            "range": "± 5456652",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 501,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 16095,
            "range": "± 8",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 403,
            "range": "± 33",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 231,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 646,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 8528,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 1696,
            "range": "± 506",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 36376,
            "range": "± 1355",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 485,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "12560266+MauroToscano@users.noreply.github.com",
            "name": "Mauro Toscano",
            "username": "MauroToscano"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "8b95b8177155f05bc6e3b65b5caaa72034400d69",
          "message": "Fix features (#644)",
          "timestamp": "2023-10-30T18:46:11Z",
          "tree_id": "3c0ea58efd1529b2bad52d488c316944a945c4a1",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/8b95b8177155f05bc6e3b65b5caaa72034400d69"
        },
        "date": 1698692875792,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 101874822,
            "range": "± 14832745",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 168818148,
            "range": "± 20215463",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 317653958,
            "range": "± 4684023",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 638426458,
            "range": "± 13060012",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 35038050,
            "range": "± 374438",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 69173080,
            "range": "± 1059950",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 136351041,
            "range": "± 1193958",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 284881020,
            "range": "± 3499661",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 28377330,
            "range": "± 573587",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 55364250,
            "range": "± 2724355",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 114662351,
            "range": "± 4068476",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 234866111,
            "range": "± 7747864",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 115504817,
            "range": "± 728857",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 242608319,
            "range": "± 1442598",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 495049062,
            "range": "± 6052369",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 985878812,
            "range": "± 17994930",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 411208812,
            "range": "± 3343937",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 793877541,
            "range": "± 7200929",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1589849771,
            "range": "± 15421705",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3150799062,
            "range": "± 22098466",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "12560266+MauroToscano@users.noreply.github.com",
            "name": "Mauro Toscano",
            "username": "MauroToscano"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "8b95b8177155f05bc6e3b65b5caaa72034400d69",
          "message": "Fix features (#644)",
          "timestamp": "2023-10-30T18:46:11Z",
          "tree_id": "3c0ea58efd1529b2bad52d488c316944a945c4a1",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/8b95b8177155f05bc6e3b65b5caaa72034400d69"
        },
        "date": 1698694504716,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 990815232,
            "range": "± 16040442",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 3502942621,
            "range": "± 23984145",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 2091347806,
            "range": "± 25771701",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 7797451478,
            "range": "± 34864817",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 4326049875,
            "range": "± 49607669",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 16817555689,
            "range": "± 51749019",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 9118574829,
            "range": "± 136692004",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 36174896131,
            "range": "± 86153903",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 33700457,
            "range": "± 1022914",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 33846508,
            "range": "± 1608645",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 53458149,
            "range": "± 2956306",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 53660832,
            "range": "± 2856630",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 71013822,
            "range": "± 1652697",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 71789241,
            "range": "± 2105586",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 128160898,
            "range": "± 3773035",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 131719213,
            "range": "± 4713963",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 142455814,
            "range": "± 1919898",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 142668981,
            "range": "± 2963755",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 272441777,
            "range": "± 4589052",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 270873527,
            "range": "± 5375887",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 290558407,
            "range": "± 8007974",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 285731951,
            "range": "± 3565857",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 549149140,
            "range": "± 9557430",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 547985775,
            "range": "± 7531970",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 59541988,
            "range": "± 1568167",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 133187905,
            "range": "± 2353945",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 264158030,
            "range": "± 2073703",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 529689670,
            "range": "± 8405971",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 1152715143,
            "range": "± 11358764",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 2452671136,
            "range": "± 52237073",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 5147031135,
            "range": "± 64103332",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 10663148641,
            "range": "± 131921213",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 1273552110,
            "range": "± 22868640",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 2684122642,
            "range": "± 47233208",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 5526852310,
            "range": "± 48914604",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 11405910691,
            "range": "± 97213710",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 3482,
            "range": "± 158",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 456772,
            "range": "± 28310",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 2320,
            "range": "± 157",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 739,
            "range": "± 43",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 3359,
            "range": "± 150",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 408584,
            "range": "± 22805",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 7833,
            "range": "± 3934",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 818358,
            "range": "± 42628",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 3669,
            "range": "± 148",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "12560266+MauroToscano@users.noreply.github.com",
            "name": "Mauro Toscano",
            "username": "MauroToscano"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "dff5f2d6b3cc5491ecef29ee7125f90af2e3cb56",
          "message": "Parallel proof of work + Update Rayon (#645)\n\n* Parallel proof of work\n\n* Fmt\n\n* Fix clippy",
          "timestamp": "2023-10-30T20:04:07Z",
          "tree_id": "a21a909bf0f28ee77041105053e1c4ced021a39b",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/dff5f2d6b3cc5491ecef29ee7125f90af2e3cb56"
        },
        "date": 1698697299944,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 98905283,
            "range": "± 4434065",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 164637435,
            "range": "± 5761321",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 316518416,
            "range": "± 3684361",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 622300333,
            "range": "± 21980380",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 35272583,
            "range": "± 419196",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 69298540,
            "range": "± 679941",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 137428062,
            "range": "± 1397676",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 282874635,
            "range": "± 4434621",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 30180486,
            "range": "± 218800",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 58044180,
            "range": "± 2166103",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 113151724,
            "range": "± 5608120",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 236594361,
            "range": "± 8527196",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 118536754,
            "range": "± 597366",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 240296374,
            "range": "± 1410378",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 496715375,
            "range": "± 3730050",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 993675979,
            "range": "± 13777255",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 407467823,
            "range": "± 2915158",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 796891104,
            "range": "± 5677305",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1575887729,
            "range": "± 14350153",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3146858875,
            "range": "± 20539309",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}