window.BENCHMARK_DATA = {
  "lastUpdate": 1695311947498,
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
          "id": "21cc017bba911e0ada0822b291a180c393616c3b",
          "message": "Stark: Remove degree adjustment from composition poly (#563)\n\n* remove degree adjustment\n\n* remove unnecessary challenges\n\n* rename coefficients variables\n\n* remove degree adjustment from docs\n\n* remove whitespaces\n\n---------\n\nCo-authored-by: Mauro Toscano <12560266+MauroToscano@users.noreply.github.com>",
          "timestamp": "2023-09-21T15:43:08Z",
          "tree_id": "6fe57f30fab3e787d8b7519b2355116c96a3cb51",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/21cc017bba911e0ada0822b291a180c393616c3b"
        },
        "date": 1695311943511,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 80906685,
            "range": "± 3973747",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 163827314,
            "range": "± 3074837",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 307505052,
            "range": "± 2128221",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 629258729,
            "range": "± 16281845",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 34259067,
            "range": "± 837515",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 68767691,
            "range": "± 781590",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 136166075,
            "range": "± 1437006",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 284195927,
            "range": "± 3127477",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 26449665,
            "range": "± 618950",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 51962362,
            "range": "± 3358772",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 108742441,
            "range": "± 6765278",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 238466666,
            "range": "± 8798259",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 118100678,
            "range": "± 876090",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 236941069,
            "range": "± 1461953",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 475929062,
            "range": "± 13338609",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 960118625,
            "range": "± 33084625",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 399135895,
            "range": "± 4741370",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 799458271,
            "range": "± 2235816",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1579105979,
            "range": "± 7845662",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3089930583,
            "range": "± 22795686",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}