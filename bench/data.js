window.BENCHMARK_DATA = {
  "lastUpdate": 1698430357132,
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
      }
    ]
  }
}