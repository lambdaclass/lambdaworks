window.BENCHMARK_DATA = {
  "lastUpdate": 1687793611863,
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
      },
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
        "date": 1687555008243,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 1062975029,
            "range": "± 1836930",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 1916963245,
            "range": "± 14509519",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 2226038549,
            "range": "± 597637",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 4648610389,
            "range": "± 41095292",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 4650237178,
            "range": "± 1940260",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 10700960338,
            "range": "± 120408158",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 9704494423,
            "range": "± 2377473",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 23195971068,
            "range": "± 32023708",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 43213676,
            "range": "± 80318",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 43149788,
            "range": "± 83404",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 61154403,
            "range": "± 327900",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 61749396,
            "range": "± 476547",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 86177492,
            "range": "± 164249",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 86347910,
            "range": "± 176697",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 131358589,
            "range": "± 386762",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 131759062,
            "range": "± 326474",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 172540801,
            "range": "± 162125",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 172337522,
            "range": "± 159817",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 267638571,
            "range": "± 278308",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 267733784,
            "range": "± 390639",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 342929339,
            "range": "± 439795",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 342980037,
            "range": "± 437279",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 533690066,
            "range": "± 1491741",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 532725676,
            "range": "± 959052",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 45018016,
            "range": "± 291530",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 95938730,
            "range": "± 369667",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 195871134,
            "range": "± 773699",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 391760079,
            "range": "± 1585634",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 1221803066,
            "range": "± 1474194",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 2548038908,
            "range": "± 2998074",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 5288586727,
            "range": "± 3015463",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 10946836309,
            "range": "± 7508037",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 1278895705,
            "range": "± 2616798",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 2677220306,
            "range": "± 2478482",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 5558335488,
            "range": "± 2379741",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 11506726160,
            "range": "± 6333461",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 34,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 149,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 78,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 35,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 113,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 215,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 363,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 942,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 34,
            "range": "± 0",
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
        "date": 1687555349997,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 957282666,
            "range": "± 628235",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 3049833764,
            "range": "± 45432111",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 2010094549,
            "range": "± 2052833",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 6784016783,
            "range": "± 65601212",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 4198681609,
            "range": "± 5077649",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 14326016743,
            "range": "± 235402695",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 8758127154,
            "range": "± 6455648",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 31061592818,
            "range": "± 413873884",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 37232136,
            "range": "± 361102",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 37321195,
            "range": "± 220687",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 55396677,
            "range": "± 4061804",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 55544160,
            "range": "± 2854056",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 73567402,
            "range": "± 334209",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 74031489,
            "range": "± 621955",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 127937091,
            "range": "± 2074600",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 127214138,
            "range": "± 1307134",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 149088164,
            "range": "± 298960",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 150134316,
            "range": "± 1081696",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 255296571,
            "range": "± 5423512",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 251060437,
            "range": "± 3610139",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 300487376,
            "range": "± 66743",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 300586322,
            "range": "± 224904",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 503459758,
            "range": "± 4259410",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 505228081,
            "range": "± 2254381",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 47382315,
            "range": "± 2460295",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 105869049,
            "range": "± 2147986",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 219498928,
            "range": "± 5641158",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 453989786,
            "range": "± 5348219",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 1118688408,
            "range": "± 3967151",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 2350314398,
            "range": "± 3979498",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 4883210913,
            "range": "± 6125364",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 10152998471,
            "range": "± 15199623",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 1212011273,
            "range": "± 3240658",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 2512505219,
            "range": "± 5099647",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 5200360674,
            "range": "± 4409934",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 10778000344,
            "range": "± 7733142",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 39,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 96,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 94,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 37,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 134,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 127,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 446,
            "range": "± 19",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 448,
            "range": "± 40",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 42,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "94461321+daphneherlambda@users.noreply.github.com",
            "name": "daphneherlambda",
            "username": "daphneherlambda"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "96ed9b375c09d4d22f70aa0fe7fba98c1cbe9821",
          "message": "Add raw fields fuzzer (#450)\n\n* Start field fuzzer\n\n* fix names to run fuzzer\n\n* fix files to run fuzzer\n\n* make basic operations in fuzzer\n\n* add ibig to compare\n\n* add result checks\n\n* Create main ops from hex string fuzzer\n\n* Add pow fuzzing\n\n* Add instructions\n\n* Simplify if\n\n* Add axiom soundness testing\n\n* add fixed fuzzer\n\n* add demonstrations to regular fuzzer\n\n* Update Cargo.toml\n\n* delete extra dependency and print\n\n* add field from raw fuzzer\n\n* change comparizon\n\n* change comparizon\n\n---------\n\nCo-authored-by: Juanma <juanma@Juanmas-MacBook-Air.local>\nCo-authored-by: dafifynn <slimbieber@gmail.com>",
          "timestamp": "2023-06-26T15:04:48Z",
          "tree_id": "da676b7869da67d05182502cca45da913ac44cac",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/96ed9b375c09d4d22f70aa0fe7fba98c1cbe9821"
        },
        "date": 1687792333953,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 132353401,
            "range": "± 1151611",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 252102458,
            "range": "± 6245837",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 484415541,
            "range": "± 8343405",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 970672396,
            "range": "± 13486894",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 35084482,
            "range": "± 389231",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 68837468,
            "range": "± 710583",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 136476841,
            "range": "± 1477440",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 279375031,
            "range": "± 3552936",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 29061957,
            "range": "± 852067",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 51700908,
            "range": "± 2465872",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 116499440,
            "range": "± 3684241",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 263948291,
            "range": "± 16943224",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 558589313,
            "range": "± 402429",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 1167423729,
            "range": "± 958872",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 2433001083,
            "range": "± 937483",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 5052140937,
            "range": "± 3150358",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 598112271,
            "range": "± 880435",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 1239722375,
            "range": "± 744124",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 2605474646,
            "range": "± 67985278",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 5366375187,
            "range": "± 146235520",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "94461321+daphneherlambda@users.noreply.github.com",
            "name": "daphneherlambda",
            "username": "daphneherlambda"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "96ed9b375c09d4d22f70aa0fe7fba98c1cbe9821",
          "message": "Add raw fields fuzzer (#450)\n\n* Start field fuzzer\n\n* fix names to run fuzzer\n\n* fix files to run fuzzer\n\n* make basic operations in fuzzer\n\n* add ibig to compare\n\n* add result checks\n\n* Create main ops from hex string fuzzer\n\n* Add pow fuzzing\n\n* Add instructions\n\n* Simplify if\n\n* Add axiom soundness testing\n\n* add fixed fuzzer\n\n* add demonstrations to regular fuzzer\n\n* Update Cargo.toml\n\n* delete extra dependency and print\n\n* add field from raw fuzzer\n\n* change comparizon\n\n* change comparizon\n\n---------\n\nCo-authored-by: Juanma <juanma@Juanmas-MacBook-Air.local>\nCo-authored-by: dafifynn <slimbieber@gmail.com>",
          "timestamp": "2023-06-26T15:04:48Z",
          "tree_id": "da676b7869da67d05182502cca45da913ac44cac",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/96ed9b375c09d4d22f70aa0fe7fba98c1cbe9821"
        },
        "date": 1687793610940,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 949193755,
            "range": "± 3310558",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 3133342153,
            "range": "± 22807971",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 1990555960,
            "range": "± 2091369",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 7098180430,
            "range": "± 18948260",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 4160208382,
            "range": "± 6659779",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 15478858486,
            "range": "± 21089740",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 8707490413,
            "range": "± 4891431",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 33187154363,
            "range": "± 56796763",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 35758529,
            "range": "± 310490",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 35727103,
            "range": "± 176368",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 49054635,
            "range": "± 732459",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 49093140,
            "range": "± 735458",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 73517205,
            "range": "± 100229",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 73614106,
            "range": "± 173160",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 120772910,
            "range": "± 2952219",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 120558233,
            "range": "± 909524",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 149359364,
            "range": "± 272388",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 149109838,
            "range": "± 261364",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 264669188,
            "range": "± 1544467",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 264072826,
            "range": "± 2089413",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 299648205,
            "range": "± 297914",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 299763604,
            "range": "± 130868",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 546351660,
            "range": "± 2246470",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 547205889,
            "range": "± 2682398",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 48195374,
            "range": "± 1047857",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 113860971,
            "range": "± 745928",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 252593868,
            "range": "± 1406282",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 529854143,
            "range": "± 2226702",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 1110014048,
            "range": "± 2458789",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 2344137816,
            "range": "± 2214860",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 4893244148,
            "range": "± 4632161",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 10169843056,
            "range": "± 7987939",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 1195088312,
            "range": "± 2720317",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 2512450238,
            "range": "± 2843600",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 5238926828,
            "range": "± 2071670",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 10850987726,
            "range": "± 8938320",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 82,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 344,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 116,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 41,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 158,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 442,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 753,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 1549,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 85,
            "range": "± 31",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}