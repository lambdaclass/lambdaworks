window.BENCHMARK_DATA = {
  "lastUpdate": 1687554119624,
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
          "id": "a1c06303e0af6185c02a35da8b0b29fbc70a53c3",
          "message": "Document Range-check builtin (#439)\n\n* Add memory holes fill\n\n* Add ordered set in docs",
          "timestamp": "2023-06-23T20:50:30Z",
          "tree_id": "754776c7fd9cf5a7db183c093426fe1c1b3aedcd",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/a1c06303e0af6185c02a35da8b0b29fbc70a53c3"
        },
        "date": 1687553907094,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 132547288,
            "range": "± 702987",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 254887802,
            "range": "± 1863749",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 491663489,
            "range": "± 7879634",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 997069313,
            "range": "± 16577175",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 34293832,
            "range": "± 295305",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 68258743,
            "range": "± 806784",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 134194129,
            "range": "± 1055676",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 279063323,
            "range": "± 4462726",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 31018029,
            "range": "± 174344",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 59609462,
            "range": "± 1115808",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 120044887,
            "range": "± 7031634",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 249223118,
            "range": "± 26534869",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 597093728,
            "range": "± 16376970",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 1347175958,
            "range": "± 94778980",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 2888296958,
            "range": "± 186514759",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 5127754479,
            "range": "± 246346617",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 610509979,
            "range": "± 3975477",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 1260540354,
            "range": "± 5898490",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 2632010562,
            "range": "± 17683102",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 5457784854,
            "range": "± 51653921",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "102986292+Juan-M-V@users.noreply.github.com",
            "name": "Juan-M-V",
            "username": "Juan-M-V"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "754cea079d2b6919e805e6eceb4d4bd7c352c510",
          "message": "Add fields fuzzer (#444)\n\n* Start field fuzzer\n\n* fix names to run fuzzer\n\n* fix files to run fuzzer\n\n* make basic operations in fuzzer\n\n* add ibig to compare\n\n* add result checks\n\n* Create main ops from hex string fuzzer\n\n* Add pow fuzzing\n\n* Add instructions\n\n* Simplify if\n\n* Add axiom soundness testing\n\n* add fixed fuzzer\n\n* add demonstrations to regular fuzzer\n\n* Update Cargo.toml\n\n* delete extra dependency and print\n\n---------\n\nCo-authored-by: Juanma <juanma@Juanmas-MacBook-Air.local>\nCo-authored-by: dafifynn <slimbieber@gmail.com>",
          "timestamp": "2023-06-23T20:54:29Z",
          "tree_id": "7cbc63d37515c49d45b4f7a856a814b6a5e9a3c1",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/754cea079d2b6919e805e6eceb4d4bd7c352c510"
        },
        "date": 1687554117011,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 123928672,
            "range": "± 2328723",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 242301916,
            "range": "± 1454856",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 478873104,
            "range": "± 7682988",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 957688374,
            "range": "± 9875160",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 33969047,
            "range": "± 256556",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 67524408,
            "range": "± 585876",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 133149925,
            "range": "± 1144018",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 272115906,
            "range": "± 1907256",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 27380854,
            "range": "± 742687",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 54367096,
            "range": "± 1817883",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 109291447,
            "range": "± 2747256",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 209734979,
            "range": "± 4545105",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 558909291,
            "range": "± 1007873",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 1168837146,
            "range": "± 2252758",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 2428306062,
            "range": "± 3092743",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 5055014729,
            "range": "± 1528194",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 594292666,
            "range": "± 1357889",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 1243517395,
            "range": "± 2945541",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 2588009354,
            "range": "± 5929223",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 5351468166,
            "range": "± 16047550",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}