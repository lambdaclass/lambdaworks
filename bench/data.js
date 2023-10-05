window.BENCHMARK_DATA = {
  "lastUpdate": 1696539560658,
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
          "id": "21cc017bba911e0ada0822b291a180c393616c3b",
          "message": "Stark: Remove degree adjustment from composition poly (#563)\n\n* remove degree adjustment\n\n* remove unnecessary challenges\n\n* rename coefficients variables\n\n* remove degree adjustment from docs\n\n* remove whitespaces\n\n---------\n\nCo-authored-by: Mauro Toscano <12560266+MauroToscano@users.noreply.github.com>",
          "timestamp": "2023-09-21T15:43:08Z",
          "tree_id": "6fe57f30fab3e787d8b7519b2355116c96a3cb51",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/21cc017bba911e0ada0822b291a180c393616c3b"
        },
        "date": 1695313154465,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 846143153,
            "range": "± 4278230",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 2664558008,
            "range": "± 61528780",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 1775984479,
            "range": "± 1474799",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 6035932501,
            "range": "± 30143927",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 3706682968,
            "range": "± 1873073",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 13163489953,
            "range": "± 35002482",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 7720267977,
            "range": "± 4902680",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 27481631561,
            "range": "± 44307257",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 30103345,
            "range": "± 794654",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 29329320,
            "range": "± 312680",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 41492564,
            "range": "± 1631766",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 43107708,
            "range": "± 1183730",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 60875563,
            "range": "± 407160",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 61377316,
            "range": "± 227855",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 108662113,
            "range": "± 1013132",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 107370029,
            "range": "± 1084430",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 124482309,
            "range": "± 465141",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 123408297,
            "range": "± 128571",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 223863609,
            "range": "± 846011",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 223836008,
            "range": "± 1641128",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 247173208,
            "range": "± 276405",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 247104947,
            "range": "± 336524",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 453489397,
            "range": "± 977229",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 455002859,
            "range": "± 1316447",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 50304523,
            "range": "± 263170",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 104601380,
            "range": "± 489063",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 216840460,
            "range": "± 1296730",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 444210516,
            "range": "± 1229194",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 994875309,
            "range": "± 2496603",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 2095032548,
            "range": "± 2213022",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 4361064036,
            "range": "± 3520451",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 9051281763,
            "range": "± 15083365",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 1060782302,
            "range": "± 2244845",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 2219063603,
            "range": "± 4137778",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 4613816404,
            "range": "± 6034531",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 9594912094,
            "range": "± 33638574",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 1539,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 98672,
            "range": "± 77",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 1224,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 433,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 1524,
            "range": "± 34",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 83628,
            "range": "± 116",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 3920,
            "range": "± 1441",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 200145,
            "range": "± 1053",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 1571,
            "range": "± 8",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "71292159+mdvillagra@users.noreply.github.com",
            "name": "Marcos Villagra",
            "username": "mdvillagra"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "976c2375bad839a04d5bc8e629191d0b2d3c44fb",
          "message": "added `pub` keyword to poseidon hash function (#569)\n\nCo-authored-by: Sergio Chouhy <41742639+schouhy@users.noreply.github.com>",
          "timestamp": "2023-09-22T13:44:40Z",
          "tree_id": "87a7030bff1b8c818adf48b0f693b3edd80d8378",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/976c2375bad839a04d5bc8e629191d0b2d3c44fb"
        },
        "date": 1695391267240,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 102539174,
            "range": "± 11053277",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 176882715,
            "range": "± 15945851",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 334263448,
            "range": "± 16848572",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 646448792,
            "range": "± 24307815",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 33766591,
            "range": "± 155416",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 75302892,
            "range": "± 6372782",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 132896377,
            "range": "± 1187104",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 283671458,
            "range": "± 2043153",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 28317216,
            "range": "± 1684814",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 53664080,
            "range": "± 2514185",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 106373677,
            "range": "± 5415439",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 227402860,
            "range": "± 8099635",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 115480887,
            "range": "± 2074079",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 241188340,
            "range": "± 1989213",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 478629500,
            "range": "± 10585724",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 963858729,
            "range": "± 39376452",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 402655051,
            "range": "± 5174533",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 806759979,
            "range": "± 12623315",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1574023833,
            "range": "± 28840401",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3319014896,
            "range": "± 286344663",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "71292159+mdvillagra@users.noreply.github.com",
            "name": "Marcos Villagra",
            "username": "mdvillagra"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "976c2375bad839a04d5bc8e629191d0b2d3c44fb",
          "message": "added `pub` keyword to poseidon hash function (#569)\n\nCo-authored-by: Sergio Chouhy <41742639+schouhy@users.noreply.github.com>",
          "timestamp": "2023-09-22T13:44:40Z",
          "tree_id": "87a7030bff1b8c818adf48b0f693b3edd80d8378",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/976c2375bad839a04d5bc8e629191d0b2d3c44fb"
        },
        "date": 1695392676308,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 1011081156,
            "range": "± 3695536",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 3110010866,
            "range": "± 5443073",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 2113824093,
            "range": "± 5236702",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 6823005902,
            "range": "± 28385985",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 4458617885,
            "range": "± 7259287",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 14784961044,
            "range": "± 45512041",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 9266342074,
            "range": "± 43792615",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 31641013783,
            "range": "± 66235225",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 36572928,
            "range": "± 156152",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 36452906,
            "range": "± 244092",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 59335428,
            "range": "± 1214342",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 60338857,
            "range": "± 832931",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 72974480,
            "range": "± 287560",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 73584498,
            "range": "± 397481",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 125949860,
            "range": "± 1475522",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 127921969,
            "range": "± 1719342",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 146585385,
            "range": "± 358240",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 146330557,
            "range": "± 576116",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 266575110,
            "range": "± 17160967",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 266360712,
            "range": "± 563578",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 292513740,
            "range": "± 552281",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 292250499,
            "range": "± 535826",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 533210883,
            "range": "± 1671072",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 533540761,
            "range": "± 1551346",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 57247241,
            "range": "± 112799",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 117319038,
            "range": "± 218997",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 243273944,
            "range": "± 400004",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 507319787,
            "range": "± 1141482",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 1206383399,
            "range": "± 2478950",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 2510188052,
            "range": "± 4106240",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 5220810178,
            "range": "± 11592323",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 10803818293,
            "range": "± 27609103",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 1266354069,
            "range": "± 5766274",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 2658736358,
            "range": "± 9562558",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 5493566709,
            "range": "± 29435490",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 11389085467,
            "range": "± 36683969",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 914,
            "range": "± 9",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 29437,
            "range": "± 95",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 834,
            "range": "± 8",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 384,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 1134,
            "range": "± 8",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 26156,
            "range": "± 109",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 2964,
            "range": "± 546",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 71937,
            "range": "± 1000",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 927,
            "range": "± 11",
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
          "id": "fe9c24f4c5e9db8ed1818877255aa937e5674a01",
          "message": "Add makefile (#571)",
          "timestamp": "2023-09-22T16:41:42Z",
          "tree_id": "6097cd455311087c4c03ca001ac459c5efbc7254",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/fe9c24f4c5e9db8ed1818877255aa937e5674a01"
        },
        "date": 1695401925827,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 88564915,
            "range": "± 6265385",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 164429268,
            "range": "± 5058062",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 300401843,
            "range": "± 3318662",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 646287021,
            "range": "± 48195265",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 42796184,
            "range": "± 2134357",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 78229297,
            "range": "± 4917999",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 147928341,
            "range": "± 4032024",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 308701239,
            "range": "± 8752517",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 51822733,
            "range": "± 4672102",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 67314498,
            "range": "± 4234561",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 132205628,
            "range": "± 9120218",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 231810826,
            "range": "± 10433062",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 166688421,
            "range": "± 5091955",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 345046531,
            "range": "± 25742143",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 674331500,
            "range": "± 37957268",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1262564020,
            "range": "± 238486491",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 406526646,
            "range": "± 1631787",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 820768396,
            "range": "± 21292382",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1792259250,
            "range": "± 100293168",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3736440083,
            "range": "± 97514073",
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
          "id": "ba2645c15a85b4f60d513150a19f885d3142b171",
          "message": "Change generator in the Stark252PrimeField to one of maximal order $2^{192}$ (#572)\n\n* use a generator of the 2-Sylow subgroup in the Stark252PrimeField\n\n* use square",
          "timestamp": "2023-09-22T16:42:12Z",
          "tree_id": "004a8fe88e82a2b770c90e8cb60536dfdb3be6cc",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/ba2645c15a85b4f60d513150a19f885d3142b171"
        },
        "date": 1695402014610,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 108925649,
            "range": "± 5302474",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 177831624,
            "range": "± 8484376",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 351638197,
            "range": "± 6927356",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 661351333,
            "range": "± 16377371",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 44188072,
            "range": "± 3322142",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 88349800,
            "range": "± 4608551",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 170586136,
            "range": "± 6097545",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 354126406,
            "range": "± 30624201",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 33092202,
            "range": "± 2152692",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 61306393,
            "range": "± 924217",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 117642335,
            "range": "± 2085019",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 231425021,
            "range": "± 7507328",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 228854166,
            "range": "± 85779219",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 446731166,
            "range": "± 173206978",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 810111500,
            "range": "± 410187647",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 996154646,
            "range": "± 703171565",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 408820458,
            "range": "± 3590456",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 800869896,
            "range": "± 4413653",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1579673145,
            "range": "± 5892977",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3142198646,
            "range": "± 16865163",
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
          "id": "fe9c24f4c5e9db8ed1818877255aa937e5674a01",
          "message": "Add makefile (#571)",
          "timestamp": "2023-09-22T16:41:42Z",
          "tree_id": "6097cd455311087c4c03ca001ac459c5efbc7254",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/fe9c24f4c5e9db8ed1818877255aa937e5674a01"
        },
        "date": 1695403081772,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 845620175,
            "range": "± 516594",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 2744734856,
            "range": "± 55190857",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 1774290545,
            "range": "± 837501",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 6047853694,
            "range": "± 17591734",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 3706379272,
            "range": "± 2605969",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 13138951980,
            "range": "± 55954320",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 7721739239,
            "range": "± 3468125",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 28092719304,
            "range": "± 79888998",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 30816443,
            "range": "± 129105",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 31215590,
            "range": "± 274507",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 52734443,
            "range": "± 1752387",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 51387154,
            "range": "± 1527059",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 61949117,
            "range": "± 96982",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 62176914,
            "range": "± 61287",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 113809711,
            "range": "± 1752258",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 116752428,
            "range": "± 1076722",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 124977731,
            "range": "± 240015",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 124703337,
            "range": "± 169878",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 236476929,
            "range": "± 957291",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 235381232,
            "range": "± 755884",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 248571303,
            "range": "± 273522",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 248634721,
            "range": "± 261463",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 474962632,
            "range": "± 596105",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 475093323,
            "range": "± 552711",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 52333753,
            "range": "± 91284",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 106321154,
            "range": "± 580469",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 220610628,
            "range": "± 656375",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 445047819,
            "range": "± 604241",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 1009250236,
            "range": "± 3088445",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 2108901924,
            "range": "± 3751586",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 4370163882,
            "range": "± 2968167",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 9076385830,
            "range": "± 7887031",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 1069708169,
            "range": "± 2058538",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 2241556643,
            "range": "± 2399466",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 4640261607,
            "range": "± 3993011",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 9602438127,
            "range": "± 15740924",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 20,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 34,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 78,
            "range": "± 7",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 31,
            "range": "± 7",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 107,
            "range": "± 5",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 57,
            "range": "± 13",
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
          "id": "ba2645c15a85b4f60d513150a19f885d3142b171",
          "message": "Change generator in the Stark252PrimeField to one of maximal order $2^{192}$ (#572)\n\n* use a generator of the 2-Sylow subgroup in the Stark252PrimeField\n\n* use square",
          "timestamp": "2023-09-22T16:42:12Z",
          "tree_id": "004a8fe88e82a2b770c90e8cb60536dfdb3be6cc",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/ba2645c15a85b4f60d513150a19f885d3142b171"
        },
        "date": 1695403549022,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 1086930260,
            "range": "± 17828901",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 3593954127,
            "range": "± 51304983",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 2191250533,
            "range": "± 26076101",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 7856112925,
            "range": "± 109528600",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 4601549133,
            "range": "± 71643386",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 16908613703,
            "range": "± 117011791",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 9334409278,
            "range": "± 142567173",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 35987448125,
            "range": "± 526031066",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 38992247,
            "range": "± 1068513",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 38734017,
            "range": "± 851242",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 70734450,
            "range": "± 1337310",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 71257479,
            "range": "± 1186607",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 76662695,
            "range": "± 522751",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 76700202,
            "range": "± 756882",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 143929940,
            "range": "± 2775787",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 143794983,
            "range": "± 2511663",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 155605509,
            "range": "± 2243599",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 156567134,
            "range": "± 3049553",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 287784631,
            "range": "± 6235719",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 287737991,
            "range": "± 6598391",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 310453979,
            "range": "± 6812602",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 310000034,
            "range": "± 8259449",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 567221753,
            "range": "± 8010573",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 569297375,
            "range": "± 9098875",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 66789192,
            "range": "± 1546139",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 130518894,
            "range": "± 2259390",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 269619555,
            "range": "± 6061610",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 542675991,
            "range": "± 11270242",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 1238450007,
            "range": "± 26551949",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 2587043985,
            "range": "± 39175812",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 5343581869,
            "range": "± 86348114",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 11167909954,
            "range": "± 122812613",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 1321545676,
            "range": "± 16789391",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 2758574073,
            "range": "± 36610026",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 5765930276,
            "range": "± 67953989",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 11580974583,
            "range": "± 102796371",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 872,
            "range": "± 86",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 28324,
            "range": "± 1264",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 829,
            "range": "± 43",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 383,
            "range": "± 17",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 1134,
            "range": "± 67",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 26612,
            "range": "± 1993",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 3015,
            "range": "± 413",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 73143,
            "range": "± 3276",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 912,
            "range": "± 35",
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
          "id": "da688cd9f845c1c32216e179dd9389f339117a02",
          "message": "Stark: make transcript compatible with Stone Prover (#570)\n\n* add StarkTranscript trait and implementation\n\n* make append field element compatible with stone prover\n\n* add test\n\n* add tests\n\n* uncomment test\n\n* remove code added by mistake to exercises\n\n* make counter of type u32",
          "timestamp": "2023-09-22T19:56:35Z",
          "tree_id": "cac007153bc2c2a91c9c30c72008225a7082860f",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/da688cd9f845c1c32216e179dd9389f339117a02"
        },
        "date": 1695413737979,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 88452593,
            "range": "± 10114947",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 162210312,
            "range": "± 3274982",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 310486240,
            "range": "± 3648050",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 626147916,
            "range": "± 29029684",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 34091685,
            "range": "± 207389",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 67439151,
            "range": "± 450592",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 133624131,
            "range": "± 857875",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 279868573,
            "range": "± 3373396",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 29448542,
            "range": "± 251124",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 56449454,
            "range": "± 1386042",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 106012347,
            "range": "± 3969365",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 238343486,
            "range": "± 10575957",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 118072937,
            "range": "± 838288",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 238579166,
            "range": "± 1012500",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 496510312,
            "range": "± 5650720",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 990154625,
            "range": "± 12970507",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 406048041,
            "range": "± 3065748",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 795755833,
            "range": "± 4646334",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1569354166,
            "range": "± 22091684",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3136845541,
            "range": "± 28612161",
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
          "id": "da688cd9f845c1c32216e179dd9389f339117a02",
          "message": "Stark: make transcript compatible with Stone Prover (#570)\n\n* add StarkTranscript trait and implementation\n\n* make append field element compatible with stone prover\n\n* add test\n\n* add tests\n\n* uncomment test\n\n* remove code added by mistake to exercises\n\n* make counter of type u32",
          "timestamp": "2023-09-22T19:56:35Z",
          "tree_id": "cac007153bc2c2a91c9c30c72008225a7082860f",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/da688cd9f845c1c32216e179dd9389f339117a02"
        },
        "date": 1695414860782,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 944337731,
            "range": "± 612738",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 1962822427,
            "range": "± 17082111",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 1975801824,
            "range": "± 1759890",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 4392851305,
            "range": "± 14543517",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 4124944145,
            "range": "± 1897986",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 9949149914,
            "range": "± 14191238",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 8598542599,
            "range": "± 5759404",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 21443665945,
            "range": "± 25079847",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 33210417,
            "range": "± 82493",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 33124355,
            "range": "± 59686",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 49773769,
            "range": "± 664698",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 51323025,
            "range": "± 565171",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 66065993,
            "range": "± 80707",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 66278829,
            "range": "± 96012",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 110851338,
            "range": "± 694966",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 110688878,
            "range": "± 559560",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 132542429,
            "range": "± 127027",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 132258744,
            "range": "± 375465",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 226668950,
            "range": "± 383265",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 226546941,
            "range": "± 431081",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 263042954,
            "range": "± 340890",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 262968931,
            "range": "± 354722",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 449942834,
            "range": "± 599845",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 450209246,
            "range": "± 507765",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 45424061,
            "range": "± 159364",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 94067998,
            "range": "± 241039",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 189335964,
            "range": "± 458475",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 375066846,
            "range": "± 1141660",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 1087717497,
            "range": "± 1816344",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 2269820040,
            "range": "± 2042287",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 4703945120,
            "range": "± 2687739",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 9795171016,
            "range": "± 13890883",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 1154499080,
            "range": "± 1935092",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 2403512451,
            "range": "± 1079422",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 4983100615,
            "range": "± 1841370",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 10315540429,
            "range": "± 3726411",
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
            "value": 153,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 76,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 36,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 108,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 214,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 340,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 917,
            "range": "± 2",
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
            "email": "41742639+schouhy@users.noreply.github.com",
            "name": "Sergio Chouhy",
            "username": "schouhy"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "0a299a76a0a6f61e6c59aacb499c70c42b81c7af",
          "message": "Stark: Small refactor of StarkTranscript (#574)\n\n* small refactor of starkprovertranscript\n\n* move keccak_hash to StarkProverTranscript impl",
          "timestamp": "2023-09-25T14:17:27Z",
          "tree_id": "1c33239360f337ca6d9d76fa4ba0c30f851647e0",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/0a299a76a0a6f61e6c59aacb499c70c42b81c7af"
        },
        "date": 1695652654942,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 103776131,
            "range": "± 9542133",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 210099757,
            "range": "± 11940642",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 410962854,
            "range": "± 21745834",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 681870541,
            "range": "± 32539529",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 46661280,
            "range": "± 4294096",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 81800818,
            "range": "± 3717891",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 138840687,
            "range": "± 2459400",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 315714635,
            "range": "± 15778293",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 30316042,
            "range": "± 875767",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 56217499,
            "range": "± 830861",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 110534563,
            "range": "± 2947654",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 229212923,
            "range": "± 8950250",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 118487356,
            "range": "± 989811",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 242343923,
            "range": "± 1313012",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 499228354,
            "range": "± 13315955",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 984501521,
            "range": "± 9918763",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 411000854,
            "range": "± 3384334",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 799364063,
            "range": "± 6272688",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1592850041,
            "range": "± 8950540",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3150361000,
            "range": "± 12614311",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "71292159+mdvillagra@users.noreply.github.com",
            "name": "Marcos Villagra",
            "username": "mdvillagra"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8e0438adb11cdbcb404116a63f51970369102c62",
          "message": "changed order of evaluation to avoid underflow (#575)",
          "timestamp": "2023-09-25T14:31:17Z",
          "tree_id": "cbf33abcc270acc3583765b5ff7fff89e3a7fb5a",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/8e0438adb11cdbcb404116a63f51970369102c62"
        },
        "date": 1695653307550,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 90813679,
            "range": "± 9833671",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 160172198,
            "range": "± 3891459",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 305762240,
            "range": "± 2844605",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 628935250,
            "range": "± 23010851",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 34690949,
            "range": "± 398473",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 69516833,
            "range": "± 497958",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 136802872,
            "range": "± 2128464",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 287174291,
            "range": "± 3115487",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 27596957,
            "range": "± 764467",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 56309147,
            "range": "± 1802114",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 109021808,
            "range": "± 3031316",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 228570958,
            "range": "± 6611478",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 115193312,
            "range": "± 692354",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 236045618,
            "range": "± 1372906",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 488282146,
            "range": "± 7843484",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 987419875,
            "range": "± 21200315",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 405468500,
            "range": "± 2362246",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 798426479,
            "range": "± 4516917",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1576969854,
            "range": "± 6095957",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3132808062,
            "range": "± 28096749",
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
          "distinct": false,
          "id": "0a299a76a0a6f61e6c59aacb499c70c42b81c7af",
          "message": "Stark: Small refactor of StarkTranscript (#574)\n\n* small refactor of starkprovertranscript\n\n* move keccak_hash to StarkProverTranscript impl",
          "timestamp": "2023-09-25T14:17:27Z",
          "tree_id": "1c33239360f337ca6d9d76fa4ba0c30f851647e0",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/0a299a76a0a6f61e6c59aacb499c70c42b81c7af"
        },
        "date": 1695653863759,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 845757378,
            "range": "± 587148",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 2683717719,
            "range": "± 12395191",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 1772440677,
            "range": "± 1151510",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 5994651159,
            "range": "± 25111035",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 3702015733,
            "range": "± 1241691",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 12978429022,
            "range": "± 38538279",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 7718028433,
            "range": "± 4459508",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 28752720907,
            "range": "± 1863184915",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 30106679,
            "range": "± 156314",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 29912197,
            "range": "± 204621",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 50167014,
            "range": "± 1376135",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 47931984,
            "range": "± 1950548",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 61832523,
            "range": "± 272676",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 61528923,
            "range": "± 167673",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 110342894,
            "range": "± 583981",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 111079008,
            "range": "± 1165777",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 124898136,
            "range": "± 963017",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 124402609,
            "range": "± 346385",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 230106702,
            "range": "± 808826",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 229427556,
            "range": "± 3082446",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 247685090,
            "range": "± 578173",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 247951557,
            "range": "± 409712",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 454135403,
            "range": "± 1101114",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 454040204,
            "range": "± 1159040",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 50937927,
            "range": "± 276229",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 104123522,
            "range": "± 167455",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 208909237,
            "range": "± 722244",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 433691133,
            "range": "± 1586199",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 999497259,
            "range": "± 6700590",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 2101013534,
            "range": "± 10016507",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 4388080552,
            "range": "± 6867411",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 9105089219,
            "range": "± 13554349",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 1067457663,
            "range": "± 3802393",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 2233646197,
            "range": "± 3313022",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 4639127913,
            "range": "± 5840829",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 9605853765,
            "range": "± 5899075",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 176,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 1437,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 219,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 101,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 340,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 1557,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 986,
            "range": "± 13",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 4644,
            "range": "± 10",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 179,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "71292159+mdvillagra@users.noreply.github.com",
            "name": "Marcos Villagra",
            "username": "mdvillagra"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8e0438adb11cdbcb404116a63f51970369102c62",
          "message": "changed order of evaluation to avoid underflow (#575)",
          "timestamp": "2023-09-25T14:31:17Z",
          "tree_id": "cbf33abcc270acc3583765b5ff7fff89e3a7fb5a",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/8e0438adb11cdbcb404116a63f51970369102c62"
        },
        "date": 1695654825773,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 1004396656,
            "range": "± 14966183",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 3517915208,
            "range": "± 36282329",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 2150902840,
            "range": "± 24739038",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 7760480930,
            "range": "± 61626609",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 4536899012,
            "range": "± 63289666",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 16908975115,
            "range": "± 94550350",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 9365430950,
            "range": "± 180860826",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 35894935843,
            "range": "± 145775530",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 35691252,
            "range": "± 933480",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 36184006,
            "range": "± 1247461",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 61666899,
            "range": "± 2409200",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 62447796,
            "range": "± 3992533",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 73016780,
            "range": "± 1447706",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 73740560,
            "range": "± 1708168",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 131865841,
            "range": "± 1955892",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 132866607,
            "range": "± 2619571",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 145409573,
            "range": "± 3937374",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 145351631,
            "range": "± 3094565",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 279621709,
            "range": "± 9204729",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 278832443,
            "range": "± 4254009",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 296473222,
            "range": "± 6996070",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 299887423,
            "range": "± 5965423",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 563948773,
            "range": "± 10916203",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 553708832,
            "range": "± 7909139",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 61478421,
            "range": "± 2101754",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 128674737,
            "range": "± 1602873",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 264134452,
            "range": "± 3202602",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 526522205,
            "range": "± 8164323",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 1240697492,
            "range": "± 25366027",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 2560057785,
            "range": "± 37880096",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 5277019130,
            "range": "± 89902985",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 10830419467,
            "range": "± 95011316",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 1267293000,
            "range": "± 16721304",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 2661061961,
            "range": "± 25112762",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 5586664318,
            "range": "± 53821597",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 11494049027,
            "range": "± 92285090",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 47,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 112,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 99,
            "range": "± 5",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 38,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 143,
            "range": "± 9",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 150,
            "range": "± 8",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 871,
            "range": "± 63",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 874,
            "range": "± 60",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 52,
            "range": "± 39",
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
          "id": "9825074e9269326bea93b788540b3bac7cde4056",
          "message": "Multiple small fixes for the prover CLI (#579)\n\n* Add fixes\n\n* Move requirements\n\n* Update ci",
          "timestamp": "2023-09-27T14:53:39Z",
          "tree_id": "64371c82774782455ab832a6320103a7ecfa1a24",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/9825074e9269326bea93b788540b3bac7cde4056"
        },
        "date": 1695827778422,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 87006244,
            "range": "± 7131810",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 160642936,
            "range": "± 6868632",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 303697448,
            "range": "± 4553277",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 662370666,
            "range": "± 34753211",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 42456189,
            "range": "± 1152361",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 72371672,
            "range": "± 2386109",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 153709776,
            "range": "± 4824593",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 305029531,
            "range": "± 4988827",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 33577447,
            "range": "± 3364080",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 63446136,
            "range": "± 2245305",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 107886064,
            "range": "± 1314096",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 222888222,
            "range": "± 10039915",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 116806077,
            "range": "± 1252919",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 242871368,
            "range": "± 1246661",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 499098229,
            "range": "± 8677462",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 985588083,
            "range": "± 12879347",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 408827083,
            "range": "± 3432456",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 790591124,
            "range": "± 5149137",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1573447124,
            "range": "± 17362420",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3149364271,
            "range": "± 12402023",
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
          "id": "9825074e9269326bea93b788540b3bac7cde4056",
          "message": "Multiple small fixes for the prover CLI (#579)\n\n* Add fixes\n\n* Move requirements\n\n* Update ci",
          "timestamp": "2023-09-27T14:53:39Z",
          "tree_id": "64371c82774782455ab832a6320103a7ecfa1a24",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/9825074e9269326bea93b788540b3bac7cde4056"
        },
        "date": 1695829088630,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 859227821,
            "range": "± 410896",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 3185504408,
            "range": "± 6594272",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 1797483634,
            "range": "± 806819",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 6982817015,
            "range": "± 10657077",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 3757353028,
            "range": "± 832512",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 14783046889,
            "range": "± 117642413",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 7828571963,
            "range": "± 2495794",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 31952585229,
            "range": "± 39043335",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 31737138,
            "range": "± 49679",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 31697024,
            "range": "± 54668",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 59980613,
            "range": "± 388996",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 59675686,
            "range": "± 249659",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 63330969,
            "range": "± 77816",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 63508025,
            "range": "± 72583",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 126544227,
            "range": "± 534298",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 127188387,
            "range": "± 428476",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 127266192,
            "range": "± 211902",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 127064959,
            "range": "± 62606",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 259825044,
            "range": "± 278144",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 259637328,
            "range": "± 532898",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 252819423,
            "range": "± 236179",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 252677519,
            "range": "± 113121",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 528712762,
            "range": "± 680971",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 530513619,
            "range": "± 1784450",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 61809749,
            "range": "± 343189",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 129972702,
            "range": "± 299862",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 274865142,
            "range": "± 370573",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 560729625,
            "range": "± 927469",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 1046702705,
            "range": "± 1657747",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 2186566298,
            "range": "± 2281132",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 4538156610,
            "range": "± 5673322",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 9425146094,
            "range": "± 22256292",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 1145420648,
            "range": "± 1707192",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 2380842395,
            "range": "± 4591939",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 4881151875,
            "range": "± 9504806",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 10142493408,
            "range": "± 35295422",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 1542,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 98767,
            "range": "± 148",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 1222,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 423,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 1521,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 83688,
            "range": "± 91",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 4167,
            "range": "± 1953",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 200774,
            "range": "± 941",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 1573,
            "range": "± 10",
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
          "distinct": true,
          "id": "c76dbf6062ad8c8388b6757e9eef11776df5763e",
          "message": "Fix verifier and CairoAIR config (#578)\n\n* Fix verifier and CairoAIR config\n\n* Add test to make sure authentication paths are being checked\n\n* Cargo format\n\n* Fix clippy for feature metal\n\n---------\n\nCo-authored-by: Agustin <agustin@pop-os.localdomain>",
          "timestamp": "2023-09-28T13:05:44Z",
          "tree_id": "7be102c8e507e252012c09602eb40dc53ac58b2f",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/c76dbf6062ad8c8388b6757e9eef11776df5763e"
        },
        "date": 1695907863140,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 86327591,
            "range": "± 10025293",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 160722758,
            "range": "± 5636650",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 305685312,
            "range": "± 1668405",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 610176417,
            "range": "± 30822978",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 34192645,
            "range": "± 142494",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 68196170,
            "range": "± 725185",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 133107191,
            "range": "± 693459",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 282826792,
            "range": "± 2887254",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 29587229,
            "range": "± 190111",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 50414081,
            "range": "± 3008841",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 112083026,
            "range": "± 9041742",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 240609215,
            "range": "± 9535791",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 116338031,
            "range": "± 1084891",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 237628159,
            "range": "± 1068321",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 489662000,
            "range": "± 3862118",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 965298938,
            "range": "± 13270805",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 408439635,
            "range": "± 2577543",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 791166416,
            "range": "± 4121383",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1577394729,
            "range": "± 14500370",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3129941291,
            "range": "± 19651039",
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
          "distinct": true,
          "id": "c76dbf6062ad8c8388b6757e9eef11776df5763e",
          "message": "Fix verifier and CairoAIR config (#578)\n\n* Fix verifier and CairoAIR config\n\n* Add test to make sure authentication paths are being checked\n\n* Cargo format\n\n* Fix clippy for feature metal\n\n---------\n\nCo-authored-by: Agustin <agustin@pop-os.localdomain>",
          "timestamp": "2023-09-28T13:05:44Z",
          "tree_id": "7be102c8e507e252012c09602eb40dc53ac58b2f",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/c76dbf6062ad8c8388b6757e9eef11776df5763e"
        },
        "date": 1695909139469,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 855775878,
            "range": "± 701333",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 2907056288,
            "range": "± 12433189",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 1790495907,
            "range": "± 664608",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 6385150634,
            "range": "± 27563422",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 3742297796,
            "range": "± 1780422",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 13862010628,
            "range": "± 17773076",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 7805260090,
            "range": "± 4088654",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 29704982603,
            "range": "± 42108007",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 29672791,
            "range": "± 245453",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 29803329,
            "range": "± 240871",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 44311089,
            "range": "± 1292886",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 44201898,
            "range": "± 1148481",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 61139879,
            "range": "± 265286",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 61194540,
            "range": "± 288348",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 110968655,
            "range": "± 1455350",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 111077964,
            "range": "± 1398886",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 124093066,
            "range": "± 229664",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 124202052,
            "range": "± 454285",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 242265006,
            "range": "± 1511542",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 241004016,
            "range": "± 1252191",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 248564310,
            "range": "± 289078",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 248468820,
            "range": "± 485355",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 499745183,
            "range": "± 1161442",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 502605412,
            "range": "± 2399481",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 56949304,
            "range": "± 480316",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 120804877,
            "range": "± 517646",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 256985457,
            "range": "± 727943",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 521823176,
            "range": "± 2054496",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 1018954868,
            "range": "± 4013859",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 2143438950,
            "range": "± 2388065",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 4466869110,
            "range": "± 3130732",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 9268198557,
            "range": "± 5209314",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 1089935424,
            "range": "± 3660589",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 2281087641,
            "range": "± 4623205",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 4740159233,
            "range": "± 4076792",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 9826130513,
            "range": "± 7813877",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 764,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 24477,
            "range": "± 7",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 705,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 318,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 962,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 21279,
            "range": "± 11",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 2857,
            "range": "± 431",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 59651,
            "range": "± 526",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 776,
            "range": "± 1",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mrugiero@gmail.com",
            "name": "Mario Rugiero",
            "username": "Oppen"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c718c5d43f6be6d42e9965e0139edef57f6432c4",
          "message": "Release v0.2.0: Tasty Tabule (#583)",
          "timestamp": "2023-09-28T19:09:02Z",
          "tree_id": "4961ec680108af502c1c0d548018c842e143b9e1",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/c718c5d43f6be6d42e9965e0139edef57f6432c4"
        },
        "date": 1695929431008,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 88699866,
            "range": "± 14947644",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 163463734,
            "range": "± 5411616",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 306956468,
            "range": "± 3430778",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 627323729,
            "range": "± 21766896",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 34178369,
            "range": "± 419901",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 68443938,
            "range": "± 618509",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 133519495,
            "range": "± 1610073",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 283536031,
            "range": "± 2527875",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 29936213,
            "range": "± 270691",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 56977680,
            "range": "± 1619841",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 110750631,
            "range": "± 5956966",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 239599479,
            "range": "± 8218240",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 119049717,
            "range": "± 881604",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 237423409,
            "range": "± 1576155",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 492207093,
            "range": "± 8677060",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 961485958,
            "range": "± 11840444",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 407756979,
            "range": "± 3804007",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 792635999,
            "range": "± 4801348",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1582823354,
            "range": "± 13501805",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3118639666,
            "range": "± 23405284",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mrugiero@gmail.com",
            "name": "Mario Rugiero",
            "username": "Oppen"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c718c5d43f6be6d42e9965e0139edef57f6432c4",
          "message": "Release v0.2.0: Tasty Tabule (#583)",
          "timestamp": "2023-09-28T19:09:02Z",
          "tree_id": "4961ec680108af502c1c0d548018c842e143b9e1",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/c718c5d43f6be6d42e9965e0139edef57f6432c4"
        },
        "date": 1695930833335,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 1013047108,
            "range": "± 22754759",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 3032582398,
            "range": "± 12151834",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 2124785728,
            "range": "± 1602242",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 6696120955,
            "range": "± 24047823",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 4441389209,
            "range": "± 4023226",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 14576579649,
            "range": "± 59749282",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 9244584325,
            "range": "± 6983275",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 31271651039,
            "range": "± 120904632",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 36373735,
            "range": "± 273724",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 36255374,
            "range": "± 167382",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 55390984,
            "range": "± 2670775",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 52022996,
            "range": "± 996082",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 72452796,
            "range": "± 196390",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 73025897,
            "range": "± 202705",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 123294315,
            "range": "± 1824568",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 122913542,
            "range": "± 2953585",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 147379898,
            "range": "± 326440",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 146124505,
            "range": "± 330377",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 262296908,
            "range": "± 1198425",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 262975537,
            "range": "± 995055",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 290258839,
            "range": "± 262925",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 290323435,
            "range": "± 322014",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 528033091,
            "range": "± 1187089",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 527897989,
            "range": "± 1004652",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 56596417,
            "range": "± 470310",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 115944432,
            "range": "± 461630",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 240165557,
            "range": "± 380709",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 495817668,
            "range": "± 1216088",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 1186444900,
            "range": "± 2009701",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 2489863190,
            "range": "± 4584929",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 5185309308,
            "range": "± 4045686",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 10780970664,
            "range": "± 15886934",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 1266802250,
            "range": "± 3335484",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 2642318060,
            "range": "± 1553597",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 5486335576,
            "range": "± 3460490",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 11363242756,
            "range": "± 7427955",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 24,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 45,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 95,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 39,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 132,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 69,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 468,
            "range": "± 10",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 70,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 27,
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
          "distinct": false,
          "id": "314fafcf44e1764dafa748f82d6df045f309cbc6",
          "message": "Crypto: Remove repeated code in Merkle tree backends (#584)\n\n* use generics for number of bits and remove repeated code\n\n* rename backend definitions\n\n* rename hash to single\n\n* minor change\n\n* fmt\n\n* rename generic\n\n* change names",
          "timestamp": "2023-09-29T15:33:17Z",
          "tree_id": "d4f750cca8afa1f64e092e8c424cd52255cb5e86",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/314fafcf44e1764dafa748f82d6df045f309cbc6"
        },
        "date": 1696002792439,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 85141717,
            "range": "± 7796533",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 165672063,
            "range": "± 2433947",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 311368739,
            "range": "± 3929062",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 647873458,
            "range": "± 22980209",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 34067398,
            "range": "± 176065",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 68336222,
            "range": "± 641128",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 133701117,
            "range": "± 1428069",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 281693708,
            "range": "± 3611309",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 30356649,
            "range": "± 345205",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 56018206,
            "range": "± 2982489",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 114463576,
            "range": "± 4946563",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 239801513,
            "range": "± 7250762",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 118867048,
            "range": "± 820690",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 240990513,
            "range": "± 1072799",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 495748197,
            "range": "± 4286097",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 988466500,
            "range": "± 18434338",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 408126614,
            "range": "± 1820050",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 794044000,
            "range": "± 6261143",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1577719166,
            "range": "± 20428804",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3147010812,
            "range": "± 22649742",
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
          "distinct": false,
          "id": "314fafcf44e1764dafa748f82d6df045f309cbc6",
          "message": "Crypto: Remove repeated code in Merkle tree backends (#584)\n\n* use generics for number of bits and remove repeated code\n\n* rename backend definitions\n\n* rename hash to single\n\n* minor change\n\n* fmt\n\n* rename generic\n\n* change names",
          "timestamp": "2023-09-29T15:33:17Z",
          "tree_id": "d4f750cca8afa1f64e092e8c424cd52255cb5e86",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/314fafcf44e1764dafa748f82d6df045f309cbc6"
        },
        "date": 1696004271785,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 965398868,
            "range": "± 30919892",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 3382888599,
            "range": "± 41881535",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 1998073823,
            "range": "± 14270601",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 7646295969,
            "range": "± 72386258",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 4168002174,
            "range": "± 42923381",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 16559258612,
            "range": "± 154404582",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 9265553232,
            "range": "± 307144945",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 35127639495,
            "range": "± 119435170",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 32226106,
            "range": "± 424500",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 30015921,
            "range": "± 864783",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 41249233,
            "range": "± 1898298",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 46282736,
            "range": "± 3469534",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 66477228,
            "range": "± 1105834",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 69789155,
            "range": "± 815564",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 126095439,
            "range": "± 2204533",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 123737619,
            "range": "± 2799307",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 136190945,
            "range": "± 1970731",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 135138655,
            "range": "± 1727972",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 261184267,
            "range": "± 5590416",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 259174933,
            "range": "± 4252725",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 277442779,
            "range": "± 4808999",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 274307824,
            "range": "± 4333993",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 543184231,
            "range": "± 8926445",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 538866060,
            "range": "± 6372312",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 60289318,
            "range": "± 2394529",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 127803863,
            "range": "± 3492671",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 258098629,
            "range": "± 2944128",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 508851083,
            "range": "± 11159936",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 1124781579,
            "range": "± 16175590",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 2449433792,
            "range": "± 50079647",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 5109385060,
            "range": "± 108154942",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 10520917855,
            "range": "± 230662394",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 1239145981,
            "range": "± 12820086",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 2660660079,
            "range": "± 47390924",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 5665520897,
            "range": "± 87733786",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 11572558849,
            "range": "± 193443152",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 3602,
            "range": "± 112",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 452300,
            "range": "± 33951",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 2279,
            "range": "± 173",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 641,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 2815,
            "range": "± 99",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 410071,
            "range": "± 19128",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 7568,
            "range": "± 1907",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 833483,
            "range": "± 78682",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 3695,
            "range": "± 127",
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
          "distinct": false,
          "id": "c6067144c5928e4910ca87d0eb786bd97c3732b2",
          "message": "Stark: Make batch commit of trace columns compatible with SHARP  (#581)\n\n* add test\n\n* make trace commitment SHARP compatible\n\n* wip\n\n* use powers of a single challenge for the boundary and transition coefficients\n\n* add permutation to match sharp compatible commitments on the trace\n\n* change trait bound from ByteConversion to Serializable\n\n* minor refactor\n\n* fmt, clippy\n\n* move std feature to inner trait function in Serializable",
          "timestamp": "2023-10-02T19:35:59Z",
          "tree_id": "a7496082d9eef839532db2966130a671ac4426cb",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/c6067144c5928e4910ca87d0eb786bd97c3732b2"
        },
        "date": 1696276612386,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 83403699,
            "range": "± 5526067",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 165142223,
            "range": "± 8779830",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 317765635,
            "range": "± 4112425",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 648096458,
            "range": "± 22493475",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 34827391,
            "range": "± 346273",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 68896603,
            "range": "± 550498",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 136716405,
            "range": "± 1223550",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 283102604,
            "range": "± 1910711",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 26993111,
            "range": "± 1029965",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 55026486,
            "range": "± 2483773",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 109736632,
            "range": "± 4055393",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 239636826,
            "range": "± 9694550",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 121483338,
            "range": "± 1169188",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 243093326,
            "range": "± 1420446",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 492102770,
            "range": "± 7948565",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 986900041,
            "range": "± 18565262",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 409161218,
            "range": "± 3988399",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 808190458,
            "range": "± 5277530",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1586043312,
            "range": "± 9269395",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3154082271,
            "range": "± 20133851",
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
          "distinct": false,
          "id": "c6067144c5928e4910ca87d0eb786bd97c3732b2",
          "message": "Stark: Make batch commit of trace columns compatible with SHARP  (#581)\n\n* add test\n\n* make trace commitment SHARP compatible\n\n* wip\n\n* use powers of a single challenge for the boundary and transition coefficients\n\n* add permutation to match sharp compatible commitments on the trace\n\n* change trait bound from ByteConversion to Serializable\n\n* minor refactor\n\n* fmt, clippy\n\n* move std feature to inner trait function in Serializable",
          "timestamp": "2023-10-02T19:35:59Z",
          "tree_id": "a7496082d9eef839532db2966130a671ac4426cb",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/c6067144c5928e4910ca87d0eb786bd97c3732b2"
        },
        "date": 1696277707270,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 943430608,
            "range": "± 2749854",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 1843149149,
            "range": "± 13102833",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 1973073422,
            "range": "± 658031",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 4494006034,
            "range": "± 13292971",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 4123112231,
            "range": "± 2720044",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 9841200069,
            "range": "± 19202547",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 8597651422,
            "range": "± 3211832",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 21563625749,
            "range": "± 22566251",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 33547232,
            "range": "± 128212",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 33442980,
            "range": "± 95446",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 52732417,
            "range": "± 600289",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 50186266,
            "range": "± 1183534",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 66353527,
            "range": "± 111665",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 66734449,
            "range": "± 139296",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 110783018,
            "range": "± 619552",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 111517845,
            "range": "± 520756",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 133632254,
            "range": "± 262949",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 133486128,
            "range": "± 485576",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 226569460,
            "range": "± 354608",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 226492825,
            "range": "± 1102723",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 264869968,
            "range": "± 1213498",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 265717945,
            "range": "± 2222681",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 448478108,
            "range": "± 774483",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 448859127,
            "range": "± 607325",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 45199498,
            "range": "± 379933",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 91864156,
            "range": "± 227209",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 185740983,
            "range": "± 405764",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 368649434,
            "range": "± 1029626",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 1085672101,
            "range": "± 2266549",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 2270857421,
            "range": "± 2445939",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 4708098964,
            "range": "± 2927186",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 9749591118,
            "range": "± 60106079",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 1152843891,
            "range": "± 2696257",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 2397136904,
            "range": "± 1688737",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 4967543322,
            "range": "± 2718641",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 10271594353,
            "range": "± 7185094",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 2204,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 282220,
            "range": "± 150",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 1033,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 529,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 1526,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 132448,
            "range": "± 33",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 3748,
            "range": "± 1820",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 391959,
            "range": "± 2924",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 2159,
            "range": "± 10",
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
          "id": "64053db4df9d47bc8bea49fe6937ab7f8ba16dbe",
          "message": "Update README.md (#594)",
          "timestamp": "2023-10-05T17:15:29-03:00",
          "tree_id": "596898ce1850264e09d63af6daf48a0572784be3",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/64053db4df9d47bc8bea49fe6937ab7f8ba16dbe"
        },
        "date": 1696537276039,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 105416083,
            "range": "± 6894606",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 166071250,
            "range": "± 15707440",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 331588572,
            "range": "± 8379006",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 628928499,
            "range": "± 11831121",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 34374146,
            "range": "± 2512960",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 69512065,
            "range": "± 1461756",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 137043849,
            "range": "± 1426049",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 282656135,
            "range": "± 1079324",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 27367794,
            "range": "± 768664",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 54439664,
            "range": "± 3241184",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 110477537,
            "range": "± 4957848",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 241406958,
            "range": "± 10110043",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 116546708,
            "range": "± 899429",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 238326062,
            "range": "± 1253507",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 488998354,
            "range": "± 5973277",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 978526395,
            "range": "± 15625568",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 406695854,
            "range": "± 3562845",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 799798729,
            "range": "± 3711681",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1581337458,
            "range": "± 10311120",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3127619104,
            "range": "± 21471206",
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
          "id": "f050c5cc435052a54cd4e6b432f8518a5ec284f9",
          "message": "Update README.md",
          "timestamp": "2023-10-05T17:28:25-03:00",
          "tree_id": "f61c45d58902478a449ff7dba233887f2389b65d",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/f050c5cc435052a54cd4e6b432f8518a5ec284f9"
        },
        "date": 1696538046700,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 108429196,
            "range": "± 8623790",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 161165305,
            "range": "± 13692370",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 335715000,
            "range": "± 14373826",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 631569083,
            "range": "± 7357218",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 34031897,
            "range": "± 2266495",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 69500655,
            "range": "± 3334222",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 133865063,
            "range": "± 1141709",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 283096427,
            "range": "± 2894620",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 27411813,
            "range": "± 1508844",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 56379709,
            "range": "± 2008751",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 106936235,
            "range": "± 5671082",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 235814861,
            "range": "± 13260448",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 115252160,
            "range": "± 2483144",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 236789819,
            "range": "± 3100484",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 521787229,
            "range": "± 18176326",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 978613916,
            "range": "± 12115677",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 411031468,
            "range": "± 3507826",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 792417250,
            "range": "± 4451758",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1580398625,
            "range": "± 11547127",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3131240145,
            "range": "± 13237909",
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
          "id": "64053db4df9d47bc8bea49fe6937ab7f8ba16dbe",
          "message": "Update README.md (#594)",
          "timestamp": "2023-10-05T17:15:29-03:00",
          "tree_id": "596898ce1850264e09d63af6daf48a0572784be3",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/64053db4df9d47bc8bea49fe6937ab7f8ba16dbe"
        },
        "date": 1696538609275,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 936690041,
            "range": "± 19365934",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 3026811121,
            "range": "± 44243234",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 1948533585,
            "range": "± 37426098",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 6516730501,
            "range": "± 81545881",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 3982298224,
            "range": "± 60245521",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 14124865646,
            "range": "± 76994117",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 8398063738,
            "range": "± 80185855",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 31286904677,
            "range": "± 176045535",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 29698072,
            "range": "± 1225255",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 29295372,
            "range": "± 1199387",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 40873931,
            "range": "± 2571776",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 41734355,
            "range": "± 1281287",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 65739295,
            "range": "± 1952404",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 66552922,
            "range": "± 1081172",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 109987750,
            "range": "± 2995838",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 113623728,
            "range": "± 2085461",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 131350072,
            "range": "± 3499240",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 136115277,
            "range": "± 2851547",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 242596721,
            "range": "± 3854168",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 241816041,
            "range": "± 4429382",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 272271853,
            "range": "± 2580713",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 272265374,
            "range": "± 4494629",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 496534017,
            "range": "± 7180102",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 501331008,
            "range": "± 9385125",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 49658311,
            "range": "± 800312",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 112513480,
            "range": "± 1421608",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 227410342,
            "range": "± 2258106",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 464384000,
            "range": "± 7398751",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 1057972593,
            "range": "± 15503095",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 2272928888,
            "range": "± 57437814",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 4716400282,
            "range": "± 54233530",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 9864070120,
            "range": "± 101350267",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 1173623958,
            "range": "± 23519479",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 2456137154,
            "range": "± 55353573",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 5114836809,
            "range": "± 95255515",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 10414411188,
            "range": "± 114734592",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 396,
            "range": "± 25",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 6340,
            "range": "± 370",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 466,
            "range": "± 22",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 195,
            "range": "± 9",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 719,
            "range": "± 43",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 6218,
            "range": "± 310",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 2069,
            "range": "± 101",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 18928,
            "range": "± 1152",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 399,
            "range": "± 25",
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
          "id": "bea1b8d6fe02e955084d82d6cf4a7f5ee1ce84c3",
          "message": "Fix clippy, remove partial ord from babybear (#591)",
          "timestamp": "2023-10-05T20:11:55Z",
          "tree_id": "df84e4384e23b03e00feebc2c6b0925531d9e7b7",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/bea1b8d6fe02e955084d82d6cf4a7f5ee1ce84c3"
        },
        "date": 1696539115438,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 100217087,
            "range": "± 12742194",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 163331664,
            "range": "± 2960845",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 317081521,
            "range": "± 2818916",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 632523667,
            "range": "± 16605507",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 34092490,
            "range": "± 1314002",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 68919545,
            "range": "± 820217",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 135983827,
            "range": "± 1295110",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 284132240,
            "range": "± 2780247",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 26980100,
            "range": "± 845595",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 55509475,
            "range": "± 1314052",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 104777582,
            "range": "± 5677365",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 236326201,
            "range": "± 12319000",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 116596688,
            "range": "± 891835",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 238143152,
            "range": "± 987779",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 486245375,
            "range": "± 10016553",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 978441937,
            "range": "± 14805624",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 403645291,
            "range": "± 2546538",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 791540541,
            "range": "± 6342442",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1553451396,
            "range": "± 10789155",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3081888542,
            "range": "± 24975099",
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
          "id": "f050c5cc435052a54cd4e6b432f8518a5ec284f9",
          "message": "Update README.md",
          "timestamp": "2023-10-05T17:28:25-03:00",
          "tree_id": "f61c45d58902478a449ff7dba233887f2389b65d",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/f050c5cc435052a54cd4e6b432f8518a5ec284f9"
        },
        "date": 1696539558881,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 1026072228,
            "range": "± 21307782",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 3585344949,
            "range": "± 16170892",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 2203156961,
            "range": "± 40198562",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 7899193499,
            "range": "± 52407109",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 4352858634,
            "range": "± 62793581",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 16992265424,
            "range": "± 120208401",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 9019217091,
            "range": "± 232403747",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 36493336707,
            "range": "± 141280322",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 35629742,
            "range": "± 1030335",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 36464768,
            "range": "± 603772",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 64477199,
            "range": "± 1948124",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 66985053,
            "range": "± 2730093",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 73280461,
            "range": "± 1241073",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 72201359,
            "range": "± 2970746",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 141282955,
            "range": "± 1942165",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 136948929,
            "range": "± 1510329",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 146030194,
            "range": "± 2952726",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 141803005,
            "range": "± 2483062",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 273878610,
            "range": "± 6672837",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 277258013,
            "range": "± 4404284",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 291880772,
            "range": "± 11156326",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 290008453,
            "range": "± 5083807",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 562545406,
            "range": "± 5694704",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 562413457,
            "range": "± 6362717",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 65176276,
            "range": "± 2457211",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 136013933,
            "range": "± 1962422",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 268087729,
            "range": "± 3970038",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 533980304,
            "range": "± 7097254",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 1199588963,
            "range": "± 35885232",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 2425405798,
            "range": "± 45600740",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 5159747561,
            "range": "± 91755380",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 10678352462,
            "range": "± 228835923",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 1306400492,
            "range": "± 36660997",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 2671508814,
            "range": "± 51293359",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 5475304181,
            "range": "± 119727543",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 11405167047,
            "range": "± 64755058",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 831,
            "range": "± 74",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 27701,
            "range": "± 1805",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 814,
            "range": "± 53",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 391,
            "range": "± 26",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 1116,
            "range": "± 67",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 25404,
            "range": "± 1703",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 2918,
            "range": "± 496",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 71635,
            "range": "± 4676",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 881,
            "range": "± 76",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}