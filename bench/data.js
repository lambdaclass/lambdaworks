window.BENCHMARK_DATA = {
  "lastUpdate": 1769802882371,
  "repoUrl": "https://github.com/lambdaclass/lambdaworks",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "56092489+ColoCarletti@users.noreply.github.com",
            "name": "Joaquin Carletti",
            "username": "ColoCarletti"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "3046f0e663d9182cd8798f8e9488ad300ffa4e46",
          "message": "Remove Metal feature (#993)\n\n* rm all metal features\n\n* fix clippy\n\n* fix attribute\n\n* fix another attribute",
          "timestamp": "2025-04-04T16:50:42Z",
          "tree_id": "2aaf079d5f553baf9a3a9e9e5307664a03c827cc",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/3046f0e663d9182cd8798f8e9488ad300ffa4e46"
        },
        "date": 1743787195774,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 320705388,
            "range": "± 1229369",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 375578528,
            "range": "± 1347354",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 279807033,
            "range": "± 961256",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 672838962,
            "range": "± 1116880",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 790967290,
            "range": "± 2596968",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1410286987,
            "range": "± 882187",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1657890232,
            "range": "± 2605314",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1232780111,
            "range": "± 1180582",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2949072229,
            "range": "± 1321530",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3459300863,
            "range": "± 8619321",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6179583205,
            "range": "± 17065154",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7291301553,
            "range": "± 247093445",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5414236330,
            "range": "± 28395988",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7468724,
            "range": "± 1841",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7537959,
            "range": "± 44546",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 9602368,
            "range": "± 562231",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 9865255,
            "range": "± 143691",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 17834704,
            "range": "± 163536",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 17830367,
            "range": "± 88037",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 26109838,
            "range": "± 746716",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 26475106,
            "range": "± 1141531",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 35383803,
            "range": "± 154065",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 35704628,
            "range": "± 2810906",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 64691879,
            "range": "± 567840",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 64827620,
            "range": "± 805876",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 71253085,
            "range": "± 124218",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 71178972,
            "range": "± 58657",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 136134606,
            "range": "± 1075568",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 137532919,
            "range": "± 2683461",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 139752058,
            "range": "± 466816",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 139712869,
            "range": "± 477014",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 270812589,
            "range": "± 420818",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 272761366,
            "range": "± 1375428",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 15817354,
            "range": "± 146297",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 34006462,
            "range": "± 1676986",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 66811860,
            "range": "± 191893",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 133633502,
            "range": "± 282515",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 353461137,
            "range": "± 1402597",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 351746130,
            "range": "± 1375309",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 753209559,
            "range": "± 2292721",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1579028534,
            "range": "± 2691542",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3285878471,
            "range": "± 3601429",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 6940267323,
            "range": "± 4428984",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 375302287,
            "range": "± 298354",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 788066352,
            "range": "± 3896669",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1660458229,
            "range": "± 855585",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3439714851,
            "range": "± 4068957",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7242585864,
            "range": "± 4345635",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 19,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 90,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 60,
            "range": "± 5",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 26,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 88,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 161,
            "range": "± 7",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 238,
            "range": "± 40",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 605,
            "range": "± 39",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 22,
            "range": "± 18",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 18,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 153,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 13668,
            "range": "± 644",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 351,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "45471455+jotabulacios@users.noreply.github.com",
            "name": "jotabulacios",
            "username": "jotabulacios"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d9385675aa91e7e99c8c023e74108fe316b243c7",
          "message": "Update Readme (#987)\n\n* update kzg Readme\n\n* KZG: make sure example is ok\n\n* Add Readme for Merkle Tree\n\n* Add Readme for Circle\n\n* fix circle docs\n\n* fix circle readme\n\n* fix typo, remove unused test and fix markdown style\n\n* changes in circle readme\n\n* fix circle readme\n\n* fix clippy\n\n* fix clippy spaces\n\n---------\n\nCo-authored-by: Nicole <nicole.graus@lambdaclass.com>\nCo-authored-by: Joaquin Carletti <joaquin.carletti@lambdaclass.com>\nCo-authored-by: Diego K <43053772+diegokingston@users.noreply.github.com>",
          "timestamp": "2025-04-04T17:10:25Z",
          "tree_id": "74f03eeb691de0aa94763b5baf42b9f4ed23213b",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/d9385675aa91e7e99c8c023e74108fe316b243c7"
        },
        "date": 1743788268386,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 319852266,
            "range": "± 3805245",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 371104952,
            "range": "± 1740701",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 280876259,
            "range": "± 4949283",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 675443325,
            "range": "± 1105947",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 791678995,
            "range": "± 1828994",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1412436337,
            "range": "± 1278948",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1652765386,
            "range": "± 3284626",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1233422421,
            "range": "± 10312045",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2952086639,
            "range": "± 10618799",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3459442373,
            "range": "± 4424472",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6182994347,
            "range": "± 8128966",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7254550053,
            "range": "± 17539210",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5409131663,
            "range": "± 16148765",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7487159,
            "range": "± 5591",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7555693,
            "range": "± 11890",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 9601082,
            "range": "± 52661",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 9656556,
            "range": "± 12599",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 17727102,
            "range": "± 20747",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 17702983,
            "range": "± 47819",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 26793247,
            "range": "± 326104",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 26627344,
            "range": "± 326648",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 35393072,
            "range": "± 61487",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 35603696,
            "range": "± 80770",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 65545766,
            "range": "± 711173",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 65906135,
            "range": "± 173066",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 71450949,
            "range": "± 123146",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 70882820,
            "range": "± 676315",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 139326643,
            "range": "± 996994",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 137803051,
            "range": "± 1177147",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 140531442,
            "range": "± 882189",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 141094402,
            "range": "± 489707",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 278696365,
            "range": "± 2199133",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 272579587,
            "range": "± 526879",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 15534075,
            "range": "± 81717",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 32852702,
            "range": "± 232320",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 66750776,
            "range": "± 387176",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 135963696,
            "range": "± 2016853",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 354161761,
            "range": "± 1124515",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 350899360,
            "range": "± 1206074",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 748718896,
            "range": "± 7440503",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1577322363,
            "range": "± 1531086",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3295011856,
            "range": "± 13650702",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 6931654664,
            "range": "± 8755697",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 379027620,
            "range": "± 1035968",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 791817068,
            "range": "± 3980896",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1665178431,
            "range": "± 1568354",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3448918670,
            "range": "± 7856603",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7253263802,
            "range": "± 9209284",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 1148,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 146687,
            "range": "± 113",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 770,
            "range": "± 28",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 421,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 1116,
            "range": "± 17",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 82224,
            "range": "± 172",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 2705,
            "range": "± 2645",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 198588,
            "range": "± 6535",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 1167,
            "range": "± 8",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 20,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 154,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 13738,
            "range": "± 1295",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 355,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "45471455+jotabulacios@users.noreply.github.com",
            "name": "jotabulacios",
            "username": "jotabulacios"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a7b22e18784dc80dc755644c0a6e310d544e1dee",
          "message": "Add Binary Field (#984)\n\n* first commit\n\n* define add function\n\n* add more functions and refactor\n\n* save work\n\n* save work\n\n* fix new function,tests and add benches\n\n* change mul function\n\n* test\n\n* change comment\n\n* change mul algorithm. Mul in level 0 and 1 working\n\n* refactor some functions and fix tests\n\n* fix inverse function\n\n* add docs and update README\n\n* fix conflicts\n\n* fix clippy\n\n* fix no_std\n\n* fix tests no std\n\n* small fixex for benches\n\n* fix typo\n\n* remove num_bits from the struct. remove set_num_level. move mul\n\n* improve add_elements()\n\n* impl Eq instead of equalls function\n\n* derive default\n\n* fix prefix in test\n\n* add readme\n\n* omit mul lifetimes\n\n* use vector of random elements for benches\n\n* fix doc\n\n---------\n\nCo-authored-by: Nicole <nicole@Nicoles-MacBook-Air.local>\nCo-authored-by: Nicole <nicole@Nicoles-Air.fibertel.com.ar>\nCo-authored-by: Nicole <nicole.graus@lambdaclass.com>\nCo-authored-by: Diego K <43053772+diegokingston@users.noreply.github.com>",
          "timestamp": "2025-04-04T22:04:46Z",
          "tree_id": "c004e1007c6a9c30eae0542f5689f1dd6c537503",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/a7b22e18784dc80dc755644c0a6e310d544e1dee"
        },
        "date": 1743805940142,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 322536981,
            "range": "± 677708",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 381470548,
            "range": "± 1184480",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 282006174,
            "range": "± 5939037",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 675795635,
            "range": "± 659215",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 795681046,
            "range": "± 2425038",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1411782190,
            "range": "± 1751973",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1651376887,
            "range": "± 2140153",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1233122116,
            "range": "± 3254844",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2946309310,
            "range": "± 1941423",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3432717624,
            "range": "± 11089516",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6173099230,
            "range": "± 8414029",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7267160590,
            "range": "± 13302343",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5403850651,
            "range": "± 9004410",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7477272,
            "range": "± 7846",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7567425,
            "range": "± 203683",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 9944972,
            "range": "± 20392",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 10014529,
            "range": "± 13387",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 17618702,
            "range": "± 30856",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 17571476,
            "range": "± 20902",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 24371876,
            "range": "± 236123",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 24059528,
            "range": "± 206987",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 35327920,
            "range": "± 386990",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 35556992,
            "range": "± 194748",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 65557303,
            "range": "± 809469",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 67348023,
            "range": "± 923589",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 71434530,
            "range": "± 133214",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 71037702,
            "range": "± 216806",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 136201709,
            "range": "± 1189575",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 135977087,
            "range": "± 2000220",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 138991812,
            "range": "± 1846913",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 138849164,
            "range": "± 100573",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 272816251,
            "range": "± 1189382",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 271332338,
            "range": "± 557768",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 15574852,
            "range": "± 132202",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 32855737,
            "range": "± 144069",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 65461659,
            "range": "± 167480",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 134686502,
            "range": "± 1861292",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 345398332,
            "range": "± 3959000",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 349246068,
            "range": "± 882945",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 746529539,
            "range": "± 7013100",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1578407282,
            "range": "± 3122986",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3290550733,
            "range": "± 7700902",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 6920528032,
            "range": "± 9116519",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 374258082,
            "range": "± 2251004",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 794782921,
            "range": "± 4374850",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1654737887,
            "range": "± 1901458",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3434244481,
            "range": "± 10075935",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7233266618,
            "range": "± 9545793",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 48,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 443,
            "range": "± 15",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 104,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 59,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 173,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 538,
            "range": "± 41",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 612,
            "range": "± 94",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 2075,
            "range": "± 159",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 51,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 13,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 13,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 45,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 6273,
            "range": "± 110",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 32,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 9886,
            "range": "± 449",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "erhany96@gmail.com",
            "name": "erhant",
            "username": "erhant"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "7fe890e658ef17cf826c917ca7a71cf6a8e139f4",
          "message": "fix: Circom Adapter (Groth16 over BLS12-381) (#991)\n\n* migrate fixes from other PR, todo tutorial guide fixes\n\n* update tutorial docs as well\n\n* update readme\n\n* fix lint\n\n* fix export witness.json doc\n\n* Remove Metal feature (#993)\n\n* rm all metal features\n\n* fix clippy\n\n* fix attribute\n\n* fix another attribute\n\n* Update Readme (#987)\n\n* update kzg Readme\n\n* KZG: make sure example is ok\n\n* Add Readme for Merkle Tree\n\n* Add Readme for Circle\n\n* fix circle docs\n\n* fix circle readme\n\n* fix typo, remove unused test and fix markdown style\n\n* changes in circle readme\n\n* fix circle readme\n\n* fix clippy\n\n* fix clippy spaces\n\n---------\n\nCo-authored-by: Nicole <nicole.graus@lambdaclass.com>\nCo-authored-by: Joaquin Carletti <joaquin.carletti@lambdaclass.com>\nCo-authored-by: Diego K <43053772+diegokingston@users.noreply.github.com>\n\n---------\n\nCo-authored-by: Joaquin Carletti <56092489+ColoCarletti@users.noreply.github.com>\nCo-authored-by: jotabulacios <45471455+jotabulacios@users.noreply.github.com>\nCo-authored-by: Nicole <nicole.graus@lambdaclass.com>\nCo-authored-by: Joaquin Carletti <joaquin.carletti@lambdaclass.com>\nCo-authored-by: Diego K <43053772+diegokingston@users.noreply.github.com>",
          "timestamp": "2025-04-07T10:31:41-03:00",
          "tree_id": "97f48d8c59fcf51b4577e7534ef45b2d59210387",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/7fe890e658ef17cf826c917ca7a71cf6a8e139f4"
        },
        "date": 1744034142491,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 321437400,
            "range": "± 348215",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 380145457,
            "range": "± 1461575",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 280568088,
            "range": "± 394822",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 676218495,
            "range": "± 347059",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 802038191,
            "range": "± 3380461",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1415170326,
            "range": "± 1381686",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1667611437,
            "range": "± 3930616",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1235252309,
            "range": "± 1199095",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2957640592,
            "range": "± 2451027",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3514601085,
            "range": "± 13768918",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6195636016,
            "range": "± 8128520",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7407546036,
            "range": "± 28219330",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5419618752,
            "range": "± 3876031",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 8222883,
            "range": "± 8167",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 8305733,
            "range": "± 11553",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 11417889,
            "range": "± 307054",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 11524311,
            "range": "± 176140",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 19698747,
            "range": "± 71650",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 19718337,
            "range": "± 97455",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 34685528,
            "range": "± 1095145",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 33501042,
            "range": "± 754484",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 38611651,
            "range": "± 61391",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 39391628,
            "range": "± 104268",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 74704448,
            "range": "± 701171",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 75474289,
            "range": "± 346220",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 78286128,
            "range": "± 136024",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 77947828,
            "range": "± 329831",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 154740630,
            "range": "± 2445437",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 154354360,
            "range": "± 963613",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 152519534,
            "range": "± 363491",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 152136820,
            "range": "± 187903",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 310582411,
            "range": "± 3168153",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 303174745,
            "range": "± 3898192",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 17386870,
            "range": "± 477977",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 37379851,
            "range": "± 726893",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 73903717,
            "range": "± 887313",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 148636900,
            "range": "± 1040597",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 380034545,
            "range": "± 3996013",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 358053518,
            "range": "± 1218181",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 758901629,
            "range": "± 1992783",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1589431581,
            "range": "± 2117222",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3334508358,
            "range": "± 7077874",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7026650519,
            "range": "± 13257586",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 383769972,
            "range": "± 732942",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 806184179,
            "range": "± 3276873",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1679221092,
            "range": "± 3563109",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3493952437,
            "range": "± 7253656",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7350396887,
            "range": "± 9856155",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 526,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 33821,
            "range": "± 49",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 412,
            "range": "± 14",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 206,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 714,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 24807,
            "range": "± 422",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 1659,
            "range": "± 1626",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 61620,
            "range": "± 679",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 543,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 12,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 68,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 83,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 9480,
            "range": "± 1249",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 43,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "nicole.graus@lambdaclass.com",
            "name": "Nicole Graus",
            "username": "nicole-graus"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ca84326212a16046c25764a04f09fbd25d4a5a5b",
          "message": "Add RSA and Schnorr Examples (#989)\n\n* first commit. adds basic rsa\n\n* refactor and update README\n\n* finish schnorr signature. rand test not working\n\n* fix comments\n\n* Change random sampl in groth16. Schnorr worikng\n\n* change sample field elem\n\n* fix clippy\n\n* fix clippy\n\n* refactor rsa and update README\n\n* remove comments\n\n* refactor schnorr\n\n* fix clippy\n\n* update ubuntu version for ci\n\n* Update README.md\n\n* Update README.md\n\n* change p and q values in the test\n\n* Update README.md\n\n* Update examples/schnorr-signature/README.md\n\nCo-authored-by: Pablo Deymonnaz <deymonnaz@gmail.com>\n\n---------\n\nCo-authored-by: jotabulacios <jbulacios@fi.uba.ar>\nCo-authored-by: Diego K <43053772+diegokingston@users.noreply.github.com>\nCo-authored-by: Pablo Deymonnaz <deymonnaz@gmail.com>",
          "timestamp": "2025-04-07T20:32:33Z",
          "tree_id": "bae795434cd5db4edc88ea2d19b5698ea4aa7e2c",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/ca84326212a16046c25764a04f09fbd25d4a5a5b"
        },
        "date": 1744059616751,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 323302174,
            "range": "± 310320",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 378198393,
            "range": "± 1889551",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 282302187,
            "range": "± 579703",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 679977030,
            "range": "± 555917",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 800838249,
            "range": "± 1952424",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1425839785,
            "range": "± 2514275",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1669592523,
            "range": "± 3375890",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1242617718,
            "range": "± 1801842",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2979299365,
            "range": "± 4949856",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3481898841,
            "range": "± 6663047",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6242153472,
            "range": "± 1656559",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7351475668,
            "range": "± 4358052",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5533157933,
            "range": "± 5560622",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 8291674,
            "range": "± 8822",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 8314536,
            "range": "± 8702",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 10334249,
            "range": "± 73145",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 10462067,
            "range": "± 41279",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 18895744,
            "range": "± 72045",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 18786245,
            "range": "± 63210",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 28289380,
            "range": "± 553212",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 27904926,
            "range": "± 689561",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 37860062,
            "range": "± 113373",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 38433200,
            "range": "± 140895",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 69951362,
            "range": "± 690983",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 69917365,
            "range": "± 452670",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 77564187,
            "range": "± 33983",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 77410617,
            "range": "± 132595",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 144575783,
            "range": "± 666787",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 145391265,
            "range": "± 509476",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 151334758,
            "range": "± 230203",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 152036489,
            "range": "± 635478",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 289779727,
            "range": "± 599516",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 292670351,
            "range": "± 685182",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 15200285,
            "range": "± 199663",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 33801645,
            "range": "± 123245",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 69162244,
            "range": "± 228661",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 139832541,
            "range": "± 472688",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 363144425,
            "range": "± 868273",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 355590805,
            "range": "± 1434883",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 755813997,
            "range": "± 1731425",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1594064923,
            "range": "± 2063410",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3328267097,
            "range": "± 3029485",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7024047657,
            "range": "± 2048699",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 379896032,
            "range": "± 943930",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 798572850,
            "range": "± 1282477",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1677234138,
            "range": "± 1460907",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3489203738,
            "range": "± 1426865",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7333133452,
            "range": "± 2950916",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 14,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 52,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 24,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 79,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 32,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 242,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 44,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 5,
            "range": "± 55",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 17,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 111,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 11799,
            "range": "± 1536",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 209,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "70894690+0xLucqs@users.noreply.github.com",
            "name": "0xLucqs",
            "username": "0xLucqs"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "201f08d93652006d30cce0dbda83c68fa20f4213",
          "message": "fix(coset): shl overflow when getting the points (#994)\n\n* fix(coset): shl overflow when getting the points\n\n* impl suggestion",
          "timestamp": "2025-04-08T20:16:18Z",
          "tree_id": "f8801b9b7fb0e19c401c46b50ae9b84edce6c97e",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/201f08d93652006d30cce0dbda83c68fa20f4213"
        },
        "date": 1744145059078,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 322612516,
            "range": "± 783501",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 375172800,
            "range": "± 1755920",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 281936890,
            "range": "± 254910",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 677832008,
            "range": "± 13655791",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 794145062,
            "range": "± 13243424",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1423013668,
            "range": "± 2821673",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1669372326,
            "range": "± 13723412",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1242990934,
            "range": "± 673322",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2970303244,
            "range": "± 2855584",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3466325400,
            "range": "± 7093704",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6222552366,
            "range": "± 14197329",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7317714054,
            "range": "± 19869092",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5452381623,
            "range": "± 1885502",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7533338,
            "range": "± 60821",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7599272,
            "range": "± 18669",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 12299973,
            "range": "± 1477743",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 10896407,
            "range": "± 297657",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 17683720,
            "range": "± 71712",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 17227183,
            "range": "± 38677",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 28665738,
            "range": "± 2466855",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 26790666,
            "range": "± 1521028",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 34946833,
            "range": "± 94476",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 35188797,
            "range": "± 119149",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 66806435,
            "range": "± 1623807",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 66642918,
            "range": "± 3325895",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 71874222,
            "range": "± 164695",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 71365370,
            "range": "± 168413",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 139299207,
            "range": "± 962149",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 138070290,
            "range": "± 836855",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 142398779,
            "range": "± 457154",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 142181162,
            "range": "± 176661",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 280338474,
            "range": "± 1033178",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 278745190,
            "range": "± 1571992",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 15010369,
            "range": "± 626637",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 34196270,
            "range": "± 241280",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 67750478,
            "range": "± 416911",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 137221681,
            "range": "± 639436",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 361147163,
            "range": "± 11347788",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 359189887,
            "range": "± 1607505",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 759419329,
            "range": "± 2488516",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1591326735,
            "range": "± 3637246",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3310404885,
            "range": "± 3380166",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 6984313813,
            "range": "± 15248299",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 383772320,
            "range": "± 2871366",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 801865129,
            "range": "± 2779209",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1672680649,
            "range": "± 4350940",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3469673705,
            "range": "± 12375245",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7318440621,
            "range": "± 16088436",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 12,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 30,
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
            "value": 83,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 52,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 297,
            "range": "± 5",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 285,
            "range": "± 9",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 11,
            "range": "± 33",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 13,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 14,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 49,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 7132,
            "range": "± 1053",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 31,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 9922,
            "range": "± 1124",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "nicole.graus@lambdaclass.com",
            "name": "Nicole Graus",
            "username": "nicole-graus"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "83c041d722f6401d418f487ffc095046cef3ee33",
          "message": "Pohlig Hellman Attack Example (#995)\n\n* save work. Add files\n\n* Save work\n\n* Curve 1 and curve 2 working just for small k\n\n* update ubuntu version for ci\n\n* save work\n\n* save work\n\n* new group over bls12-381\n\n* refactor\n\n* small refactor\n\n* Add Readme and remove comments\n\n* brute force test\n\n* print iterations brute force\n\n* add brute-force comment to readme\n\n* fix clippy\n\n* Update README.md\n\n* change return of function\n\n* fix clippy\n\n* change chinese theorem error\n\n* fix clippy\n\n---------\n\nCo-authored-by: jotabulacios <jbulacios@fi.uba.ar>\nCo-authored-by: Diego K <43053772+diegokingston@users.noreply.github.com>\nCo-authored-by: diegokingston <dkingston@fi.uba.ar>",
          "timestamp": "2025-04-08T20:16:51Z",
          "tree_id": "77bf8329b60320dd42d600228559cef253915074",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/83c041d722f6401d418f487ffc095046cef3ee33"
        },
        "date": 1744145122407,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 337032598,
            "range": "± 341044",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 381483433,
            "range": "± 1256221",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 283310648,
            "range": "± 2804886",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 704873522,
            "range": "± 1115648",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 793288212,
            "range": "± 3021807",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1475980310,
            "range": "± 2772032",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1675442743,
            "range": "± 2776426",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1243963546,
            "range": "± 693660",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 3085603525,
            "range": "± 3928348",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3489619301,
            "range": "± 5515004",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6469173555,
            "range": "± 12242973",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7345325930,
            "range": "± 14801476",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5458799022,
            "range": "± 5324731",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7535012,
            "range": "± 4997",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7589493,
            "range": "± 5072",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 9885412,
            "range": "± 244086",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 10617170,
            "range": "± 756745",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 17681854,
            "range": "± 95454",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 17663075,
            "range": "± 56199",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 33282154,
            "range": "± 1042732",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 32422792,
            "range": "± 629797",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 35741599,
            "range": "± 152664",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 35891305,
            "range": "± 124162",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 70867350,
            "range": "± 1047052",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 72034422,
            "range": "± 814846",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 72028951,
            "range": "± 217272",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 72009037,
            "range": "± 134463",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 146744415,
            "range": "± 903954",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 144872484,
            "range": "± 1746349",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 143579239,
            "range": "± 341996",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 143657769,
            "range": "± 213450",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 289237882,
            "range": "± 1526873",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 289097187,
            "range": "± 2251628",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 16802329,
            "range": "± 176758",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 36269099,
            "range": "± 293885",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 72344885,
            "range": "± 455678",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 144187979,
            "range": "± 576753",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 378376989,
            "range": "± 3084053",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 373914343,
            "range": "± 819878",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 792624341,
            "range": "± 2955407",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1659470692,
            "range": "± 2034017",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3444698418,
            "range": "± 2100483",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7276915840,
            "range": "± 8618775",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 396680461,
            "range": "± 1630153",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 825378837,
            "range": "± 2117927",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1728524137,
            "range": "± 3164005",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3598299240,
            "range": "± 6189967",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7580173498,
            "range": "± 8160892",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 1118,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 143324,
            "range": "± 93",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 787,
            "range": "± 13",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 395,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 1223,
            "range": "± 14",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 103113,
            "range": "± 6851",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 2733,
            "range": "± 2190",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 198259,
            "range": "± 2451",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 1140,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 13,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 14,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 49,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 8708,
            "range": "± 1263",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 31,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
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
          "id": "3e413875bb4f9b45883678cf05a6184c2fa344a7",
          "message": "Update README.md (#996)",
          "timestamp": "2025-04-09T15:03:41Z",
          "tree_id": "d9561f498cfb89b926cd67b1b68143bc679e0227",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/3e413875bb4f9b45883678cf05a6184c2fa344a7"
        },
        "date": 1744212724057,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 324920723,
            "range": "± 714090",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 389038579,
            "range": "± 2439642",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 280658658,
            "range": "± 155014",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 681160668,
            "range": "± 248528",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 812031122,
            "range": "± 3655666",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1425623370,
            "range": "± 2525518",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1690466477,
            "range": "± 9088668",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1233969790,
            "range": "± 2262847",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2979013463,
            "range": "± 1909816",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3537440924,
            "range": "± 16510532",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6248667893,
            "range": "± 3209709",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7418915661,
            "range": "± 21389826",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5414230850,
            "range": "± 4142559",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 8224964,
            "range": "± 3977",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 8327296,
            "range": "± 14214",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 15235837,
            "range": "± 613764",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 15942800,
            "range": "± 1397765",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 19346030,
            "range": "± 80631",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 19246494,
            "range": "± 158896",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 36265454,
            "range": "± 861308",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 36820588,
            "range": "± 715920",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 38403885,
            "range": "± 151439",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 38858659,
            "range": "± 101715",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 74550757,
            "range": "± 756336",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 73759444,
            "range": "± 762117",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 78384524,
            "range": "± 224747",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 77856418,
            "range": "± 220158",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 149600278,
            "range": "± 687720",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 149696164,
            "range": "± 1468261",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 153582268,
            "range": "± 474294",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 154268345,
            "range": "± 287543",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 294645286,
            "range": "± 1943735",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 295609179,
            "range": "± 3411762",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 18537299,
            "range": "± 896767",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 36569058,
            "range": "± 1253456",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 74686485,
            "range": "± 3202358",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 149105815,
            "range": "± 8062333",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 377799909,
            "range": "± 7356768",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 363425586,
            "range": "± 2688926",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 766702373,
            "range": "± 1698843",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1600311370,
            "range": "± 4699138",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3335901351,
            "range": "± 8207817",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7067173707,
            "range": "± 19563159",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 390649595,
            "range": "± 1307099",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 808225955,
            "range": "± 3076283",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1688784363,
            "range": "± 3784791",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3507734586,
            "range": "± 9986787",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7397674088,
            "range": "± 14892021",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 11,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 27,
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
            "value": 83,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 50,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 420,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 396,
            "range": "± 24",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 10,
            "range": "± 33",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 16,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 114,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 11709,
            "range": "± 141",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 181,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "70894690+0xLucqs@users.noreply.github.com",
            "name": "0xLucqs",
            "username": "0xLucqs"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "28d98d12796324d4e5ac0bd65822b60044833f4f",
          "message": "feat(fft): add fast mul for fft friendly fields (#997)\n\n* feat(fft): add fast multiplication\n\n* chore: add fast_mul benchmark\n\n* impl suggestion\n\n* chore: impl suggestion",
          "timestamp": "2025-04-24T11:42:11-03:00",
          "tree_id": "6ca8fe89487eb084ce66efe5ed933d8a0469ee79",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/28d98d12796324d4e5ac0bd65822b60044833f4f"
        },
        "date": 1745507172482,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 323095920,
            "range": "± 1261552",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 381310400,
            "range": "± 872291",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 284164188,
            "range": "± 4116159",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 679369981,
            "range": "± 810523",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 797459457,
            "range": "± 3780285",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1423521246,
            "range": "± 5238649",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1679774753,
            "range": "± 4278875",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1245748884,
            "range": "± 700869",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2975875355,
            "range": "± 2490922",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3489698546,
            "range": "± 10616458",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6234688392,
            "range": "± 10017355",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7338377516,
            "range": "± 18712246",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5463540080,
            "range": "± 5381404",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7535564,
            "range": "± 13506",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7610741,
            "range": "± 9731",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 9874940,
            "range": "± 38479",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 9957001,
            "range": "± 39111",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 17565090,
            "range": "± 59018",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 17596298,
            "range": "± 106852",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 29561814,
            "range": "± 785516",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 28986776,
            "range": "± 838003",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 35319624,
            "range": "± 141279",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 35863407,
            "range": "± 150121",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 69515687,
            "range": "± 1070546",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 72622649,
            "range": "± 919373",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 72790486,
            "range": "± 380722",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 72645348,
            "range": "± 305096",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 147070723,
            "range": "± 2338026",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 147494705,
            "range": "± 3491661",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 143530041,
            "range": "± 332839",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 142656895,
            "range": "± 867997",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 292109333,
            "range": "± 1138652",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 291708799,
            "range": "± 3143061",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 17541361,
            "range": "± 212414",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 36253166,
            "range": "± 1459691",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 73590100,
            "range": "± 2807304",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 150340022,
            "range": "± 2419909",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 374286957,
            "range": "± 2555151",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 356441569,
            "range": "± 1384057",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 761023412,
            "range": "± 1855627",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1603843893,
            "range": "± 6280452",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3344561007,
            "range": "± 7075151",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7036811867,
            "range": "± 16790565",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 382696252,
            "range": "± 1571526",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 806443815,
            "range": "± 2875034",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1688585365,
            "range": "± 3490387",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3502721466,
            "range": "± 9651174",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7396696375,
            "range": "± 19519533",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 541,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 34669,
            "range": "± 86",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 491,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 225,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 704,
            "range": "± 5",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 22367,
            "range": "± 102",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 143710,
            "range": "± 982",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1640725,
            "range": "± 10187",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 1764,
            "range": "± 1524",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 64755,
            "range": "± 689",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 558,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 13,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 10454,
            "range": "± 136",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 90,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 10177,
            "range": "± 1624",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 68,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
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
          "id": "7a336ca3ba8d64838b443bd8efca463c8f9c73b9",
          "message": "Improve documentation (#999)\n\n* add explanation\n\n* fix format\n\n* add more comments\n\n* explain fft\n\n* continue docs\n\n* add more docs\n\n* Update README.md",
          "timestamp": "2025-04-25T13:14:46Z",
          "tree_id": "022c7aafee18c3201d20dc2f72866b2976de4463",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/7a336ca3ba8d64838b443bd8efca463c8f9c73b9"
        },
        "date": 1745588590282,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 325167512,
            "range": "± 357543",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 383158347,
            "range": "± 1617891",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 283667452,
            "range": "± 673043",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 681873524,
            "range": "± 709042",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 800844884,
            "range": "± 2440366",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1426885051,
            "range": "± 2199439",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1685459138,
            "range": "± 6215426",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1245741005,
            "range": "± 1225876",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2979830104,
            "range": "± 2748076",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3512155368,
            "range": "± 11844782",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6243942783,
            "range": "± 3071006",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7412213765,
            "range": "± 12457020",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5459712657,
            "range": "± 4943137",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 8290321,
            "range": "± 4696",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 8370760,
            "range": "± 3293",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 14170538,
            "range": "± 818168",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 13933441,
            "range": "± 1034893",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 19307683,
            "range": "± 59690",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 19274290,
            "range": "± 88828",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 36390917,
            "range": "± 581987",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 35852131,
            "range": "± 584930",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 38590353,
            "range": "± 97914",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 38887320,
            "range": "± 154665",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 74765705,
            "range": "± 999177",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 75525460,
            "range": "± 869899",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 78438497,
            "range": "± 366840",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 78001815,
            "range": "± 258709",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 151883276,
            "range": "± 708111",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 151727353,
            "range": "± 1254970",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 154930578,
            "range": "± 417839",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 154533924,
            "range": "± 739885",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 300723530,
            "range": "± 2506367",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 301823566,
            "range": "± 1642341",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 18338046,
            "range": "± 410092",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 37397722,
            "range": "± 478214",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 74264302,
            "range": "± 1059452",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 150819057,
            "range": "± 2524344",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 381178755,
            "range": "± 1610092",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 367148607,
            "range": "± 1102936",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 774064122,
            "range": "± 2943004",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1612194695,
            "range": "± 2441785",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3354505499,
            "range": "± 6630940",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7088948617,
            "range": "± 8311963",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 396038895,
            "range": "± 1603365",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 818248907,
            "range": "± 2089020",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1700098707,
            "range": "± 4653184",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3512063378,
            "range": "± 10489721",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7399620535,
            "range": "± 8445734",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 49,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 411,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 106,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 62,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 178,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 525,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 143348,
            "range": "± 925",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1634355,
            "range": "± 19974",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 539,
            "range": "± 102",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 2030,
            "range": "± 9",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 50,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 13,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 14,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 48,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 7560,
            "range": "± 423",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 31,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "76252340+MarcosNicolau@users.noreply.github.com",
            "name": "Marcos Nicolau",
            "username": "MarcosNicolau"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "5f8f2cfcc8a1a22f77e8dff2d581f1166eefb80b",
          "message": "refactor: remove default contraint in merkle tree trait (#1004)",
          "timestamp": "2025-05-05T17:24:46Z",
          "tree_id": "99bb8089f5ac9fb827806203d60745320423ba07",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/5f8f2cfcc8a1a22f77e8dff2d581f1166eefb80b"
        },
        "date": 1746467642382,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 322695457,
            "range": "± 611075",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 379666224,
            "range": "± 1708063",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 283159377,
            "range": "± 280436",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 678621383,
            "range": "± 1024919",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 801010091,
            "range": "± 3717688",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1422804968,
            "range": "± 1623345",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1685157329,
            "range": "± 4821781",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1248506328,
            "range": "± 1446382",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2970836960,
            "range": "± 4518248",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3511009894,
            "range": "± 13818676",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6218981042,
            "range": "± 2679661",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7348890928,
            "range": "± 45362648",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5471409863,
            "range": "± 5267401",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7547554,
            "range": "± 3567",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7613059,
            "range": "± 17813",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 11110302,
            "range": "± 986468",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 10456885,
            "range": "± 935224",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 17720959,
            "range": "± 117143",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 17509571,
            "range": "± 60419",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 26884118,
            "range": "± 3893736",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 27320635,
            "range": "± 1262458",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 35282223,
            "range": "± 88192",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 35364523,
            "range": "± 263505",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 70209580,
            "range": "± 2554368",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 68429936,
            "range": "± 3239328",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 71839950,
            "range": "± 358314",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 71613864,
            "range": "± 166352",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 143328198,
            "range": "± 2573163",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 144476560,
            "range": "± 2732008",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 142982342,
            "range": "± 358916",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 142746211,
            "range": "± 204361",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 293947899,
            "range": "± 5164140",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 285371374,
            "range": "± 2739342",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 15455079,
            "range": "± 1100988",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 34623605,
            "range": "± 876922",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 79039795,
            "range": "± 2884411",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 155956312,
            "range": "± 4109799",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 386660504,
            "range": "± 4930568",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 367288520,
            "range": "± 5024959",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 761519817,
            "range": "± 4583372",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1602598212,
            "range": "± 5446278",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3339587740,
            "range": "± 9652549",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7032887949,
            "range": "± 15356491",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 381480913,
            "range": "± 1204371",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 800032601,
            "range": "± 4976345",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1679026410,
            "range": "± 3301481",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3498004477,
            "range": "± 10242867",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7363066169,
            "range": "± 23772139",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 260,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 8397,
            "range": "± 19",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 288,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 174,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 410,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 7117,
            "range": "± 24",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 142690,
            "range": "± 370",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1643673,
            "range": "± 9220",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 1297,
            "range": "± 1151",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 22317,
            "range": "± 451",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 266,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 13,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 76,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 86,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 10357,
            "range": "± 1017",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 47,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "45471455+jotabulacios@users.noreply.github.com",
            "name": "jotabulacios",
            "username": "jotabulacios"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c9d02b585f9dafee732eb6f2e30b7028b546e79e",
          "message": "Add big uint conversion (#1002)\n\n* first iteration to add BigUint conversion\n\n* remove comments\n\n* refactor\n\n* refactor\n\n* save work. tests working for different fields\n\n* small refactor\n\n* conversion not working for baby bear u32\n\n* all tests working for every field\n\n* remove comments in u64_prime_field file\n\n* remove commented code\n\n* fix cargo check no-default-features\n\n* remove changes in unsigned_integer\n\n* fix ensure-no-std\n\n* remove Ok and unwrap\n\n* fix ci set up job\n\n* fix compile error\n\n* add BN254 conversion test\n\n* change test for bn254\n\n---------\n\nCo-authored-by: Nicole <nicole@Nicoles-MacBook-Air.local>\nCo-authored-by: Nicole <nicole.graus@lambdaclass.com>\nCo-authored-by: Diego K <43053772+diegokingston@users.noreply.github.com>",
          "timestamp": "2025-05-09T17:05:53-03:00",
          "tree_id": "411624d8e8ce1278066bf25ab94b6ec368be30ab",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/c9d02b585f9dafee732eb6f2e30b7028b546e79e"
        },
        "date": 1746822596607,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 323451059,
            "range": "± 268299",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 376807613,
            "range": "± 1313548",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 282823402,
            "range": "± 4229959",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 679629634,
            "range": "± 388299",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 793103466,
            "range": "± 1689426",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1423896294,
            "range": "± 4353360",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1667357554,
            "range": "± 2933823",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1243888081,
            "range": "± 1182913",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2981132872,
            "range": "± 14664835",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3491020927,
            "range": "± 12755257",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6237050535,
            "range": "± 3669578",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7378530124,
            "range": "± 8954215",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5455713687,
            "range": "± 3948262",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 8283673,
            "range": "± 12800",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 8365016,
            "range": "± 10269",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 10708471,
            "range": "± 263766",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 10559195,
            "range": "± 84566",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 18851283,
            "range": "± 78314",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 18851605,
            "range": "± 91511",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 29964823,
            "range": "± 1297784",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 29475991,
            "range": "± 1098928",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 37847179,
            "range": "± 78574",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 38543573,
            "range": "± 123975",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 70124996,
            "range": "± 889107",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 71117875,
            "range": "± 1337717",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 78066603,
            "range": "± 191719",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 77446273,
            "range": "± 103521",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 147490149,
            "range": "± 1618457",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 148394439,
            "range": "± 867443",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 153641567,
            "range": "± 404252",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 152971383,
            "range": "± 191232",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 294984697,
            "range": "± 1790160",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 295303295,
            "range": "± 1052874",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 16497744,
            "range": "± 258775",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 35447765,
            "range": "± 817360",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 71884891,
            "range": "± 759761",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 145665541,
            "range": "± 2061222",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 362358587,
            "range": "± 3427299",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 357871222,
            "range": "± 1944202",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 762395212,
            "range": "± 1794992",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1598158132,
            "range": "± 2698045",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3342636358,
            "range": "± 6681481",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7032456476,
            "range": "± 8214333",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 384676977,
            "range": "± 1797321",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 800995608,
            "range": "± 2764245",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1684298581,
            "range": "± 3068358",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3480717808,
            "range": "± 3908788",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7347177147,
            "range": "± 15737995",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 114,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 1785,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 189,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 103,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 306,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 1664,
            "range": "± 9",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 143733,
            "range": "± 348",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1634228,
            "range": "± 13567",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 838,
            "range": "± 11",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 7191,
            "range": "± 70",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 119,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 16,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 133,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 11765,
            "range": "± 1220",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 192,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "70894690+0xLucqs@users.noreply.github.com",
            "name": "0xLucqs",
            "username": "0xLucqs"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3a04406abfa059bd5688dd0622188d516b9207bd",
          "message": "Feat/fast division (#1001)\n\n* feat(fft): add fast multiplication\n\n* chore: add fast_mul benchmark\n\n* impl suggestion\n\n* chore: impl suggestion\n\n* chore: add fast_mul benchmark\n\n* feat(fft): fast division\n\n* refacto: proper error handling\n\n* refacto: rename inversion function\n\n* test: add tests for fast division helper functions\n\n* doc: explain where fast division is taken from\n\n---------\n\nCo-authored-by: Diego K <43053772+diegokingston@users.noreply.github.com>",
          "timestamp": "2025-05-15T11:21:27-03:00",
          "tree_id": "a401827c78813d6302976cb93c864f92ff017538",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/3a04406abfa059bd5688dd0622188d516b9207bd"
        },
        "date": 1747320346474,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 323651543,
            "range": "± 1347806",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 377760438,
            "range": "± 5438045",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 281258420,
            "range": "± 414580",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 680871017,
            "range": "± 1593472",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 794346637,
            "range": "± 3680509",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1426878837,
            "range": "± 2642264",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1664871175,
            "range": "± 7061273",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1234675450,
            "range": "± 781125",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2980746943,
            "range": "± 2749213",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3472252181,
            "range": "± 15622923",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6241157860,
            "range": "± 6483124",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7328442952,
            "range": "± 21082462",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5418122047,
            "range": "± 3827502",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 8294349,
            "range": "± 10955",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 8324635,
            "range": "± 7575",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 10548718,
            "range": "± 565785",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 10565931,
            "range": "± 85847",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 19237325,
            "range": "± 99834",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 19090906,
            "range": "± 87802",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 30147093,
            "range": "± 889764",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 29569539,
            "range": "± 698940",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 38045423,
            "range": "± 412428",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 38700087,
            "range": "± 111030",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 70792684,
            "range": "± 1858007",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 71763221,
            "range": "± 1104188",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 77901976,
            "range": "± 149738",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 77577288,
            "range": "± 70800",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 151523400,
            "range": "± 2341193",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 151679560,
            "range": "± 5131143",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 153066959,
            "range": "± 328225",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 152323824,
            "range": "± 158579",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 298841401,
            "range": "± 3845369",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 297299902,
            "range": "± 4046873",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 16843391,
            "range": "± 371024",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 36288315,
            "range": "± 661660",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 73578024,
            "range": "± 1583633",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 151147337,
            "range": "± 3069945",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 384451913,
            "range": "± 2802015",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 360296155,
            "range": "± 2186425",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 764778711,
            "range": "± 2073004",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1607611338,
            "range": "± 3010162",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3352624023,
            "range": "± 15148626",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7078324717,
            "range": "± 15005930",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 385570603,
            "range": "± 1541553",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 805398869,
            "range": "± 7281180",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1693593662,
            "range": "± 3103706",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3510503667,
            "range": "± 14395872",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7377033939,
            "range": "± 11796395",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 117,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 1903,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 178,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 116,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 301,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 1642,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 144007,
            "range": "± 403",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1571401,
            "range": "± 99375",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 810686,
            "range": "± 5528",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1639385,
            "range": "± 24018",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 872,
            "range": "± 378",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 7335,
            "range": "± 117",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 121,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 16,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 128,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 11837,
            "range": "± 1418",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 185,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "45471455+jotabulacios@users.noreply.github.com",
            "name": "jotabulacios",
            "username": "jotabulacios"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "055dc6f600786dd49f4ff49383fb94ff5f629992",
          "message": "Add quadratic and cubic sumcheck (#1003)\n\n* first commit, add quadratic and cubic\n\n* fix quadratic and cubic sumcheck implementation\n\n* refactor code to use wrappers\n\n* save work\n\n* small refactor\n\n* implement suggestion\n\n* add number of factors and number of variables to transcript\n\n* fix clippy\n\n* fix clippy\n\n---------\n\nCo-authored-by: Diego K <43053772+diegokingston@users.noreply.github.com>",
          "timestamp": "2025-05-23T21:05:22Z",
          "tree_id": "d7ece2de75dabfc66391f1d65ae7ac0e5f588916",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/055dc6f600786dd49f4ff49383fb94ff5f629992"
        },
        "date": 1748036131031,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 326496974,
            "range": "± 438535",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 370473084,
            "range": "± 2057600",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 279833211,
            "range": "± 249176",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 685084167,
            "range": "± 1356756",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 778619601,
            "range": "± 1773755",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1436109821,
            "range": "± 750294",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1642109086,
            "range": "± 5425699",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1230921589,
            "range": "± 6874242",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 3003171093,
            "range": "± 3646826",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3434759053,
            "range": "± 11654593",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6281247560,
            "range": "± 6053618",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7255101011,
            "range": "± 31926338",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5391957695,
            "range": "± 2193662",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7982502,
            "range": "± 9125",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7988857,
            "range": "± 6945",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 10760900,
            "range": "± 461432",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 10239695,
            "range": "± 62862",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 18148284,
            "range": "± 77625",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 18180202,
            "range": "± 48963",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 27906144,
            "range": "± 1798682",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 27001470,
            "range": "± 509727",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 37256775,
            "range": "± 532440",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 37008792,
            "range": "± 143783",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 67117391,
            "range": "± 544298",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 67528725,
            "range": "± 1092948",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 74617386,
            "range": "± 140251",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 74609511,
            "range": "± 300154",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 140176752,
            "range": "± 1681715",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 142683585,
            "range": "± 3404708",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 151740687,
            "range": "± 159612",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 151544508,
            "range": "± 207066",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 293685774,
            "range": "± 4258666",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 289174893,
            "range": "± 3285835",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 15739747,
            "range": "± 294617",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 33651440,
            "range": "± 643980",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 68485500,
            "range": "± 962095",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 136081483,
            "range": "± 1168278",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 357297181,
            "range": "± 4718925",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 356095104,
            "range": "± 926830",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 763605224,
            "range": "± 2668117",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1608294675,
            "range": "± 2081607",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3349213751,
            "range": "± 5320388",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7084137048,
            "range": "± 8029157",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 389087602,
            "range": "± 1085388",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 808083947,
            "range": "± 1913828",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1697281784,
            "range": "± 6104393",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3509655096,
            "range": "± 6328631",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7393626168,
            "range": "± 17853875",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 115,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 1840,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 173,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 104,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 311,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 1570,
            "range": "± 192",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 144982,
            "range": "± 1136",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1547321,
            "range": "± 5389",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 815367,
            "range": "± 1753",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1672401,
            "range": "± 8214",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 807,
            "range": "± 14",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 7399,
            "range": "± 233",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 119,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 13,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 14,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 48,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 6863,
            "range": "± 695",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 31,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "45471455+jotabulacios@users.noreply.github.com",
            "name": "jotabulacios",
            "username": "jotabulacios"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ad2a380641758cfbbe2de6f4a2d8e27684f0fed7",
          "message": "fix clippy for new rust version (#1013)",
          "timestamp": "2025-08-01T21:05:21Z",
          "tree_id": "f84f62fc85c1e1ea5772038679bacf93f88bdb68",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/ad2a380641758cfbbe2de6f4a2d8e27684f0fed7"
        },
        "date": 1754084121312,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 326255928,
            "range": "± 939803",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 369905715,
            "range": "± 594165",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 279918677,
            "range": "± 517863",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 685061449,
            "range": "± 401403",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 784183788,
            "range": "± 1889981",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1435415520,
            "range": "± 471741",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1652110374,
            "range": "± 10000424",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1229965637,
            "range": "± 637220",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 3000409988,
            "range": "± 1215925",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3462072077,
            "range": "± 23803410",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6282392621,
            "range": "± 3316907",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7302940268,
            "range": "± 51380774",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5401374650,
            "range": "± 1015956",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7984778,
            "range": "± 5238",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 8002408,
            "range": "± 8754",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 10117812,
            "range": "± 143285",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 10105146,
            "range": "± 171507",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 18284352,
            "range": "± 43257",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 18263368,
            "range": "± 126630",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 28601307,
            "range": "± 676882",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 28027605,
            "range": "± 549105",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 37646908,
            "range": "± 104867",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 37105755,
            "range": "± 74237",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 73362296,
            "range": "± 2625624",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 70230220,
            "range": "± 2546897",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 74917747,
            "range": "± 221190",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 74634070,
            "range": "± 124648",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 144976371,
            "range": "± 1219601",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 144467825,
            "range": "± 1143031",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 153115114,
            "range": "± 660368",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 153397623,
            "range": "± 1002267",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 294342794,
            "range": "± 1591177",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 297526550,
            "range": "± 3486689",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 16776380,
            "range": "± 143093",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 35856901,
            "range": "± 441807",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 72059970,
            "range": "± 1421281",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 140824082,
            "range": "± 890820",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 368940861,
            "range": "± 2559109",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 359683099,
            "range": "± 1662306",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 765859213,
            "range": "± 1812981",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1610462797,
            "range": "± 2047693",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3353514823,
            "range": "± 1999644",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7084306723,
            "range": "± 4945241",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 386847105,
            "range": "± 4092238",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 808214464,
            "range": "± 2599220",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1700979431,
            "range": "± 7115410",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3550286267,
            "range": "± 10650763",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7463239684,
            "range": "± 28967920",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 12,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 31,
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
            "value": 85,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 53,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 145307,
            "range": "± 729",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1545362,
            "range": "± 7383",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 816282,
            "range": "± 2472",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1744462,
            "range": "± 15641",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 293,
            "range": "± 22",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 291,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 10,
            "range": "± 36",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 13,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 10408,
            "range": "± 30",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 90,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 10176,
            "range": "± 477",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 69,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "pdeymon@fi.uba.ar",
            "name": "Pablo Deymonnaz",
            "username": "pablodeymo"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "4b018546d485d2183229af08050da6828e44727f",
          "message": "Fix bincode repo change (#1023)\n\n* Fix bincode inclusion\n\n* Add feature serde\n\n* fix clippy\n\n* fix warnings in benches and solve bincode dependency\n\n* use crates io for bincode\n\n---------\n\nCo-authored-by: jotabulacios <jbulacios@fi.uba.ar>",
          "timestamp": "2025-09-04T10:39:44-03:00",
          "tree_id": "92dc6ad2e4737096fc389a707999cd69ee64d351",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/4b018546d485d2183229af08050da6828e44727f"
        },
        "date": 1756994647048,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 325991755,
            "range": "± 681598",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 374448513,
            "range": "± 3757645",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 277883420,
            "range": "± 2144164",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 685338373,
            "range": "± 467234",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 805259952,
            "range": "± 4998892",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1436433999,
            "range": "± 1251909",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1679420004,
            "range": "± 12453102",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1223299184,
            "range": "± 4274812",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 3001160329,
            "range": "± 2126221",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3496676362,
            "range": "± 28102858",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6285745381,
            "range": "± 3353500",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7399146270,
            "range": "± 26025662",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5390031430,
            "range": "± 5988611",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7975984,
            "range": "± 4911",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7995500,
            "range": "± 5253",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 10100023,
            "range": "± 36995",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 10340819,
            "range": "± 438263",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 18150525,
            "range": "± 74325",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 18224592,
            "range": "± 107672",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 27015866,
            "range": "± 1754348",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 26961971,
            "range": "± 945042",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 37349926,
            "range": "± 69767",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 37078828,
            "range": "± 73917",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 71806229,
            "range": "± 810077",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 71588700,
            "range": "± 963589",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 75228231,
            "range": "± 309974",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 75085703,
            "range": "± 131621",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 153828378,
            "range": "± 3316296",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 151642432,
            "range": "± 2680587",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 153118461,
            "range": "± 366388",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 153301852,
            "range": "± 228070",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 309341257,
            "range": "± 5463317",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 314997063,
            "range": "± 6437859",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 18671865,
            "range": "± 714154",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 39585478,
            "range": "± 1104319",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 81110671,
            "range": "± 3985246",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 159675731,
            "range": "± 4888886",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 396307028,
            "range": "± 4988328",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 357561710,
            "range": "± 1809150",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 768588543,
            "range": "± 3089943",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1623667653,
            "range": "± 5221858",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3396026948,
            "range": "± 5832406",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7144512983,
            "range": "± 7067355",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 385163105,
            "range": "± 2373397",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 811093379,
            "range": "± 2634027",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1705919182,
            "range": "± 4690839",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3540674157,
            "range": "± 9615678",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7463924191,
            "range": "± 17757604",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 14,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 51,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 24,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 75,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 32,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 145308,
            "range": "± 298",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1543587,
            "range": "± 14118",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 814917,
            "range": "± 1820",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1713758,
            "range": "± 33265",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 251,
            "range": "± 14",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 44,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 5,
            "range": "± 60",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 13,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 10334,
            "range": "± 25",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 88,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 10157,
            "range": "± 961",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 64,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 9727,
            "range": "± 1211",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "nicole.graus@lambdaclass.com",
            "name": "Nicole Graus",
            "username": "nicole-graus"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "575c4a208e4182fa5998965808b953f512124cb7",
          "message": "Add GKR Protocol (#1011)\n\n* save work, protocol structure now is following the post style\n\n* add prints to debug implementation\n\n* refactor to avoid unwraps\n\n* more refactor\n\n* more refactor\n\n* refactor\n\n* add readme\n\n* Fix readme. Add struct Prover\n\n* remove extra cargo.toml\n\n* remove claimed_sum from sumcheck_proof. Fix test. Add documentation\n\n* remove clones and fix clippy\n\n* add documentation\n\n* add check mark for gkr in readme\n\n* add degree check on g_j\n\n* add verifier checks: proof structure match circuit structure\n\n* fix clippy for new rust version (#1013)\n\n* fix clippy\n\n* fix readme\n\n* rename modulus\n\n* avoid repetitive computation\n\n* check terms has len 2\n\n* remove clone term_1 and term_2\n\n* add check number of inputs\n\n* fix type complexity\n\n---------\n\nCo-authored-by: jotabulacios <jbulacios@fi.uba.ar>\nCo-authored-by: jotabulacios <45471455+jotabulacios@users.noreply.github.com>\nCo-authored-by: Diego K <43053772+diegokingston@users.noreply.github.com>",
          "timestamp": "2025-09-04T17:41:15Z",
          "tree_id": "4a0dbcbe5f6bff84ac8a4a145fdbd56730252eb7",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/575c4a208e4182fa5998965808b953f512124cb7"
        },
        "date": 1757009404290,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 327349357,
            "range": "± 225229",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 369173127,
            "range": "± 944363",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 278358638,
            "range": "± 3996838",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 686881448,
            "range": "± 450374",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 777632938,
            "range": "± 1958864",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1439667581,
            "range": "± 6619044",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1632517436,
            "range": "± 6433470",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1221655289,
            "range": "± 937573",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 3007052303,
            "range": "± 7970711",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3424299583,
            "range": "± 4418343",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6293588909,
            "range": "± 2019723",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7240206657,
            "range": "± 21137611",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5357256651,
            "range": "± 10105561",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 8005486,
            "range": "± 4484",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 8005962,
            "range": "± 12041",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 10621194,
            "range": "± 98679",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 10575369,
            "range": "± 67678",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 19365209,
            "range": "± 125406",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 19234876,
            "range": "± 103713",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 32380989,
            "range": "± 1786760",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 30467225,
            "range": "± 1327679",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 38214504,
            "range": "± 49895",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 38218622,
            "range": "± 83004",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 71712421,
            "range": "± 1102960",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 69059180,
            "range": "± 523308",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 75501772,
            "range": "± 353384",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 75331493,
            "range": "± 94651",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 141133423,
            "range": "± 1640193",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 142705663,
            "range": "± 566601",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 151238136,
            "range": "± 228972",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 151140655,
            "range": "± 98154",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 285776897,
            "range": "± 1171319",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 289713741,
            "range": "± 700536",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 15281536,
            "range": "± 358394",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 33943059,
            "range": "± 191313",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 68260684,
            "range": "± 330574",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 137590252,
            "range": "± 1099558",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 350515501,
            "range": "± 2421887",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 362356571,
            "range": "± 844229",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 769775233,
            "range": "± 2815799",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1612756789,
            "range": "± 2208520",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3354180619,
            "range": "± 7685594",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7081259770,
            "range": "± 10563601",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 389717837,
            "range": "± 2412829",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 810352827,
            "range": "± 2141090",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1691467127,
            "range": "± 4272275",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3509554173,
            "range": "± 2748299",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7386491497,
            "range": "± 5383268",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 242,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 7794,
            "range": "± 7",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 278,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 195,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 466,
            "range": "± 5",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 6462,
            "range": "± 72",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 145173,
            "range": "± 193",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1578950,
            "range": "± 5956",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 816826,
            "range": "± 1654",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1675230,
            "range": "± 12407",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 1245,
            "range": "± 1077",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 22756,
            "range": "± 313",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 249,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 12,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 14,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 48,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 7236,
            "range": "± 525",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 31,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 9669,
            "range": "± 262",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "56092489+ColoCarletti@users.noreply.github.com",
            "name": "Joaquin Carletti",
            "username": "ColoCarletti"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "160bb5ca435696a41387c8cd95ad1f451e497a73",
          "message": "Fix clippy (#1029)\n\n* fix clippy\n\n* remove unused import\n\n* fix",
          "timestamp": "2025-09-22T13:55:28Z",
          "tree_id": "35c4006d18ca2209259c1fd0ec20c1ae7f75ef99",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/160bb5ca435696a41387c8cd95ad1f451e497a73"
        },
        "date": 1758551006111,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 325335444,
            "range": "± 125867",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 369983647,
            "range": "± 985942",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 277303546,
            "range": "± 165830",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 684694098,
            "range": "± 678642",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 778176882,
            "range": "± 1775762",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1434791074,
            "range": "± 2940528",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1638901678,
            "range": "± 4906497",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1220013815,
            "range": "± 2853602",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2995143977,
            "range": "± 3485241",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3421099090,
            "range": "± 7458792",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6266438925,
            "range": "± 3331942",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7254859249,
            "range": "± 10861104",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5358648266,
            "range": "± 7326848",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 8168692,
            "range": "± 5196",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 8179494,
            "range": "± 5922",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 10447423,
            "range": "± 332621",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 11518927,
            "range": "± 843071",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 18489389,
            "range": "± 69462",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 18383582,
            "range": "± 84599",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 27534711,
            "range": "± 250836",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 27370098,
            "range": "± 321637",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 37211194,
            "range": "± 116104",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 37496783,
            "range": "± 174741",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 69841715,
            "range": "± 414266",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 69643673,
            "range": "± 1306146",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 75099049,
            "range": "± 132341",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 75235898,
            "range": "± 171668",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 146259185,
            "range": "± 813673",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 147469116,
            "range": "± 992128",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 151743604,
            "range": "± 173893",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 151569894,
            "range": "± 245275",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 295296106,
            "range": "± 1103585",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 294863150,
            "range": "± 429504",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 17265704,
            "range": "± 263477",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 36519140,
            "range": "± 366020",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 74608768,
            "range": "± 5225600",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 146583199,
            "range": "± 751191",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 380952927,
            "range": "± 2779776",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 363822408,
            "range": "± 2322157",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 774116277,
            "range": "± 2971178",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1614958369,
            "range": "± 6043447",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3360972770,
            "range": "± 2582537",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7082677363,
            "range": "± 8344820",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 390025306,
            "range": "± 1313770",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 811466603,
            "range": "± 2131396",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1697281333,
            "range": "± 889175",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3513056901,
            "range": "± 11513542",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7407719018,
            "range": "± 5551583",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 541,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 34591,
            "range": "± 379",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 436,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 237,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 743,
            "range": "± 8",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 27471,
            "range": "± 202",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 149369,
            "range": "± 332",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1568268,
            "range": "± 7377",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 837200,
            "range": "± 1554",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1736615,
            "range": "± 10807",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 1677,
            "range": "± 1405",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 68220,
            "range": "± 825",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 556,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 14,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "45471455+jotabulacios@users.noreply.github.com",
            "name": "jotabulacios",
            "username": "jotabulacios"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "060dbf6af6e0c293e80f143fc84c01df47984304",
          "message": "Add new unchecked (#1024)\n\n* add new_uncheckes for short weiestrass\n\n* remove unused files\n\n* use unchecked for stark curve generator\n\n* revert changes for vesta curve\n\n* add missing comment\n\n* make the point field private\n\n* fix clippy\n\n* add safety comments and rename function\n\n---------\n\nCo-authored-by: Nicole <nicole.graus@lambdaclass.com>",
          "timestamp": "2025-09-26T12:41:53Z",
          "tree_id": "58fde0f7f49b7bb1c20f6adb2b69cfd487ca2a32",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/060dbf6af6e0c293e80f143fc84c01df47984304"
        },
        "date": 1758892260395,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 325215580,
            "range": "± 847396",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 373657068,
            "range": "± 2434340",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 278297585,
            "range": "± 216465",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 682379484,
            "range": "± 1547866",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 785330520,
            "range": "± 5010239",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1430639896,
            "range": "± 2018894",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1641789202,
            "range": "± 4857329",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1223414376,
            "range": "± 926179",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2990728001,
            "range": "± 1673318",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3462773251,
            "range": "± 27530512",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6254017292,
            "range": "± 6648237",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7276449166,
            "range": "± 28044908",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5363226478,
            "range": "± 2129329",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7802390,
            "range": "± 7737",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7855133,
            "range": "± 3591",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 12414636,
            "range": "± 329671",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 12445642,
            "range": "± 406244",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 18406185,
            "range": "± 25581",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 18406602,
            "range": "± 26824",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 28133954,
            "range": "± 1611770",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 25844952,
            "range": "± 1778691",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 36405996,
            "range": "± 273027",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 36731186,
            "range": "± 194837",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 71598287,
            "range": "± 2491460",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 73062654,
            "range": "± 2116195",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 74007339,
            "range": "± 182959",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 73830442,
            "range": "± 290963",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 145939467,
            "range": "± 2058601",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 144493665,
            "range": "± 1143129",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 147382138,
            "range": "± 149852",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 147205967,
            "range": "± 179827",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 289140208,
            "range": "± 1155299",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 296248833,
            "range": "± 2802542",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 16989761,
            "range": "± 659266",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 35842569,
            "range": "± 462530",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 69181791,
            "range": "± 547365",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 143995557,
            "range": "± 1552718",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 378716909,
            "range": "± 5075004",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 355185973,
            "range": "± 4450696",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 764245901,
            "range": "± 5096431",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1612145494,
            "range": "± 3757159",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3339393239,
            "range": "± 3569650",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7105842097,
            "range": "± 14807527",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 382037081,
            "range": "± 1024800",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 804546863,
            "range": "± 1461616",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1700177771,
            "range": "± 3290557",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3544380419,
            "range": "± 8550266",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7422766439,
            "range": "± 23445052",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 20,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 94,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 63,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 29,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 91,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 165,
            "range": "± 5",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 149398,
            "range": "± 235",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1577888,
            "range": "± 15509",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 837095,
            "range": "± 1089",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1750395,
            "range": "± 19949",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 266,
            "range": "± 52",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 646,
            "range": "± 142",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 22,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 13,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 15,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 49,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 7157,
            "range": "± 647",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 32,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "56092489+ColoCarletti@users.noreply.github.com",
            "name": "Joaquin Carletti",
            "username": "ColoCarletti"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "5c451b961aacd957e6d837f23550cfa3144241d0",
          "message": "Release 0.13.0 (#1032)\n\n* update version\n\n* add publish workflow\n\n* update dependencies versions\n\n* fix name\n\n* add --workspace",
          "timestamp": "2025-09-26T22:11:09Z",
          "tree_id": "f6fa21ee6bc287b148793d5475cf7feb52221341",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/5c451b961aacd957e6d837f23550cfa3144241d0"
        },
        "date": 1758926412276,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 325963016,
            "range": "± 3399029",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 371738149,
            "range": "± 1561326",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 278814470,
            "range": "± 7929519",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 683930195,
            "range": "± 678225",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 784875772,
            "range": "± 4714139",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1431549352,
            "range": "± 866932",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1640861027,
            "range": "± 5503930",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1224447072,
            "range": "± 1196037",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2993522190,
            "range": "± 11282622",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3412001808,
            "range": "± 15221984",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6259757545,
            "range": "± 8681837",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7226452713,
            "range": "± 8290036",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5365975769,
            "range": "± 2925741",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7806927,
            "range": "± 5786",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7852937,
            "range": "± 20188",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 11733822,
            "range": "± 1335452",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 11315285,
            "range": "± 278017",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 18110453,
            "range": "± 49818",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 18093574,
            "range": "± 40461",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 30587502,
            "range": "± 1034200",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 30614571,
            "range": "± 767048",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 36459325,
            "range": "± 155532",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 36610808,
            "range": "± 46152",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 70282343,
            "range": "± 726581",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 70748278,
            "range": "± 952778",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 73801286,
            "range": "± 184280",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 73530496,
            "range": "± 78973",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 143704585,
            "range": "± 469241",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 143309730,
            "range": "± 405695",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 146940510,
            "range": "± 221954",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 146619131,
            "range": "± 231523",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 286076963,
            "range": "± 1299961",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 286364706,
            "range": "± 821430",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 16264109,
            "range": "± 257096",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 35221517,
            "range": "± 172517",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 71058491,
            "range": "± 338395",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 140004925,
            "range": "± 622099",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 367674065,
            "range": "± 818266",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 361810685,
            "range": "± 1080391",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 769739185,
            "range": "± 10838758",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1607312731,
            "range": "± 1350613",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3343970614,
            "range": "± 9667806",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7069251321,
            "range": "± 8776698",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 388927278,
            "range": "± 1511521",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 812653115,
            "range": "± 2123301",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1699394986,
            "range": "± 1993612",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3500682480,
            "range": "± 3594168",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7373802937,
            "range": "± 5009223",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 12,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 30,
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
            "value": 83,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 51,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 149089,
            "range": "± 376",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1553940,
            "range": "± 7253",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 834314,
            "range": "± 1458",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1688453,
            "range": "± 17068",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 269,
            "range": "± 11",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 269,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 10,
            "range": "± 38",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 13,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 14,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 48,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 7159,
            "range": "± 985",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 32,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 9754,
            "range": "± 121",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "56092489+ColoCarletti@users.noreply.github.com",
            "name": "Joaquin Carletti",
            "username": "ColoCarletti"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "980313e6e5d3962aa15cbfbe9c73e534af2dfc82",
          "message": "fix publish workflow (#1033)",
          "timestamp": "2025-09-29T14:07:47Z",
          "tree_id": "1715dc3b4f1804b05b1bdbb785bf06720cfd136f",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/980313e6e5d3962aa15cbfbe9c73e534af2dfc82"
        },
        "date": 1759156644327,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 325276909,
            "range": "± 299122",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 373799212,
            "range": "± 1280792",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 276761830,
            "range": "± 3240462",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 684123108,
            "range": "± 458835",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 784548181,
            "range": "± 860109",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1432390908,
            "range": "± 2789501",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1648173519,
            "range": "± 1888834",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1217575532,
            "range": "± 3284841",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2993699711,
            "range": "± 719515",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3440730954,
            "range": "± 10184483",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6268181109,
            "range": "± 3354598",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7305123303,
            "range": "± 13595186",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5340979849,
            "range": "± 11014483",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7792357,
            "range": "± 26787",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7845413,
            "range": "± 4191",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 9905692,
            "range": "± 54020",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 9935119,
            "range": "± 37029",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 17946159,
            "range": "± 72651",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 17854317,
            "range": "± 29782",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 27077437,
            "range": "± 1066053",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 26842568,
            "range": "± 504628",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 36099204,
            "range": "± 25086",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 36296376,
            "range": "± 29384",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 67316601,
            "range": "± 696305",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 67900197,
            "range": "± 610414",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 73460960,
            "range": "± 125364",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 73239280,
            "range": "± 97542",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 142108141,
            "range": "± 990596",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 143959311,
            "range": "± 2673369",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 146531409,
            "range": "± 321114",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 146469282,
            "range": "± 105839",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 284632593,
            "range": "± 945734",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 283681688,
            "range": "± 1164461",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 15729115,
            "range": "± 66530",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 35615464,
            "range": "± 669731",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 70791379,
            "range": "± 556050",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 139936815,
            "range": "± 910885",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 365484553,
            "range": "± 5490571",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 361663058,
            "range": "± 2990451",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 762404414,
            "range": "± 2192560",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1604253640,
            "range": "± 3479460",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3343212740,
            "range": "± 3525470",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7061624679,
            "range": "± 9724140",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 385896349,
            "range": "± 2498005",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 805836010,
            "range": "± 2063810",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1693259461,
            "range": "± 1272833",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3502638453,
            "range": "± 9288862",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7379752116,
            "range": "± 5243622",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 12,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 27,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 56,
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
            "value": 84,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 50,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 149332,
            "range": "± 202",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1576574,
            "range": "± 14090",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 837195,
            "range": "± 3444",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1749773,
            "range": "± 5302",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 279,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 278,
            "range": "± 10",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 9,
            "range": "± 37",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 14,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 10383,
            "range": "± 26",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 92,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 10130,
            "range": "± 952",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 62,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "56092489+ColoCarletti@users.noreply.github.com",
            "name": "Joaquin Carletti",
            "username": "ColoCarletti"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "6f03bf1ba13c4207faafb5b0c639b68f8b60d55f",
          "message": "Remove all-features in publish (#1034)\n\n* remove all-features in publish\n\n* rm mac specific features",
          "timestamp": "2025-09-29T15:41:05Z",
          "tree_id": "3627df9a79145407673b45e92a794163a751866f",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/6f03bf1ba13c4207faafb5b0c639b68f8b60d55f"
        },
        "date": 1759162230136,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 326052587,
            "range": "± 327303",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 378794070,
            "range": "± 1060771",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 279230377,
            "range": "± 6382372",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 684888426,
            "range": "± 7314944",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 797613594,
            "range": "± 9744927",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1433431586,
            "range": "± 645093",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1667154408,
            "range": "± 11292086",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1221477059,
            "range": "± 6207384",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2993403094,
            "range": "± 1566266",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3480509202,
            "range": "± 21420319",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6276792127,
            "range": "± 11290789",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7341362876,
            "range": "± 14007302",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5353857536,
            "range": "± 12102265",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7800649,
            "range": "± 20792",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7867376,
            "range": "± 97685",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 14969400,
            "range": "± 510732",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 15421656,
            "range": "± 388852",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 18548995,
            "range": "± 65264",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 18477744,
            "range": "± 49657",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 35407419,
            "range": "± 369150",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 35945733,
            "range": "± 254154",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 37047255,
            "range": "± 371682",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 37149369,
            "range": "± 905982",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 72676056,
            "range": "± 785625",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 72988400,
            "range": "± 220160",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 74992312,
            "range": "± 118634",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 74678301,
            "range": "± 76627",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 146134758,
            "range": "± 419783",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 144749031,
            "range": "± 361141",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 148144466,
            "range": "± 506503",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 148525673,
            "range": "± 175507",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 290099351,
            "range": "± 1499140",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 291142433,
            "range": "± 1259193",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 18126240,
            "range": "± 529617",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 36343143,
            "range": "± 936185",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 70890838,
            "range": "± 746462",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 144304341,
            "range": "± 938313",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 377452695,
            "range": "± 7593715",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 370578339,
            "range": "± 875132",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 776197736,
            "range": "± 762739",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1613468970,
            "range": "± 1173146",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3355807362,
            "range": "± 2133974",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7078061283,
            "range": "± 5411817",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 397337274,
            "range": "± 3375303",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 814209726,
            "range": "± 19613983",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1696964267,
            "range": "± 17870997",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3508199492,
            "range": "± 8024534",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7379264858,
            "range": "± 23020004",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 257,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 8230,
            "range": "± 192",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 287,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 184,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 439,
            "range": "± 12",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 6177,
            "range": "± 68",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 149213,
            "range": "± 203",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1554409,
            "range": "± 11862",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 836692,
            "range": "± 1413",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1658194,
            "range": "± 9165",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 1255,
            "range": "± 1086",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 22172,
            "range": "± 1776",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 266,
            "range": "± 15",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 13,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 16,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 78,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 10080,
            "range": "± 197",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 38,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "nicole.graus@lambdaclass.com",
            "name": "Nicole Graus",
            "username": "nicole-graus"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d9c3c04b61cda15e4b416f00ee8be3594b2b5708",
          "message": "Add Plonky3 in the Readme Table (#1035)\n\n* add plonky3\n\n* add circle for plonky3\n\n* exchange columns hal2 and plonky3\n\n* fix clippy\n\n* fix fmt\n\n---------\n\nCo-authored-by: jotabulacios <jbulacios@fi.uba.ar>",
          "timestamp": "2025-10-13T19:41:14Z",
          "tree_id": "a2dd8768cdb51093a256f03d5efa08eb67d1951a",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/d9c3c04b61cda15e4b416f00ee8be3594b2b5708"
        },
        "date": 1760386215632,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 327003901,
            "range": "± 949477",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 372022574,
            "range": "± 1166774",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 277600530,
            "range": "± 4350259",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 687967982,
            "range": "± 1073563",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 785086018,
            "range": "± 4227415",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1442572902,
            "range": "± 1855986",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1646686277,
            "range": "± 9313311",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1221192814,
            "range": "± 892421",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 3017093305,
            "range": "± 1301463",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3446167998,
            "range": "± 21249847",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6314248004,
            "range": "± 1965320",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7325057350,
            "range": "± 17558812",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5356191949,
            "range": "± 2566175",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 8215384,
            "range": "± 6980",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 8286234,
            "range": "± 10374",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 10339980,
            "range": "± 40292",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 10461441,
            "range": "± 43933",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 18975059,
            "range": "± 85143",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 19123153,
            "range": "± 97328",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 29601906,
            "range": "± 753768",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 29007544,
            "range": "± 625723",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 37952887,
            "range": "± 232341",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 38089513,
            "range": "± 107115",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 70829695,
            "range": "± 1072510",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 70545049,
            "range": "± 1064771",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 76952903,
            "range": "± 150406",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 77525153,
            "range": "± 202212",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 145733558,
            "range": "± 1678057",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 146793590,
            "range": "± 2083820",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 153579462,
            "range": "± 433381",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 153146846,
            "range": "± 410654",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 299082342,
            "range": "± 3417229",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 299439096,
            "range": "± 4130444",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 16791018,
            "range": "± 468063",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 35837178,
            "range": "± 623944",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 72010386,
            "range": "± 1448005",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 143570805,
            "range": "± 3183094",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 364989740,
            "range": "± 3545268",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 364073093,
            "range": "± 2471830",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 770399434,
            "range": "± 5166278",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1622037508,
            "range": "± 2128959",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3385749529,
            "range": "± 10682648",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7134777766,
            "range": "± 12866778",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 393985161,
            "range": "± 2542414",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 813949176,
            "range": "± 3791740",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1712152356,
            "range": "± 4667565",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3540453825,
            "range": "± 12449434",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7434124041,
            "range": "± 12618574",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 4,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 15,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 51,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 24,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 77,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 32,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 153972,
            "range": "± 329",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1577286,
            "range": "± 118712",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 858628,
            "range": "± 800",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1661229,
            "range": "± 25863",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 253,
            "range": "± 12",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 44,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 5,
            "range": "± 60",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 13,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 14,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 48,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 7456,
            "range": "± 749",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 32,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 9845,
            "range": "± 449",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "34384633+tdelabro@users.noreply.github.com",
            "name": "Timothée Delabrouille",
            "username": "tdelabro"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3b96e92c046dc7088a53dfb6f4772489a05cc11b",
          "message": "feat: support formatter alternate display (#1049)\n\n* feat: support formatter alternate display\n\n* feat: impl LowerHex and UpperHex on UnsignedElement",
          "timestamp": "2025-10-17T15:49:47-03:00",
          "tree_id": "574bc70ac6a625145ec24c1cb3e7998ea0ef98c3",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/3b96e92c046dc7088a53dfb6f4772489a05cc11b"
        },
        "date": 1760728468268,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 326213869,
            "range": "± 1078415",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 370570059,
            "range": "± 1115020",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 279011883,
            "range": "± 629632",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 684628464,
            "range": "± 1113127",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 783866900,
            "range": "± 2893513",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1432018970,
            "range": "± 1802752",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1647858640,
            "range": "± 15739586",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1227111175,
            "range": "± 13531387",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2992092458,
            "range": "± 3553204",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3448239716,
            "range": "± 19567647",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6265565996,
            "range": "± 2482725",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7334672606,
            "range": "± 23616032",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5378243807,
            "range": "± 5991506",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7805391,
            "range": "± 29109",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7859042,
            "range": "± 7321",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 10431971,
            "range": "± 416148",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 12191920,
            "range": "± 947698",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 17409705,
            "range": "± 44579",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 17399479,
            "range": "± 77643",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 31808193,
            "range": "± 2282209",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 30773175,
            "range": "± 1901507",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 34959431,
            "range": "± 352689",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 35180008,
            "range": "± 233535",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 71316851,
            "range": "± 1033492",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 71278099,
            "range": "± 1137563",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 70258482,
            "range": "± 467684",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 70190864,
            "range": "± 265152",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 145678862,
            "range": "± 2495058",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 143040457,
            "range": "± 2294540",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 140790093,
            "range": "± 343259",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 140634133,
            "range": "± 332762",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 289069488,
            "range": "± 4781889",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 292111570,
            "range": "± 5331905",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 17607601,
            "range": "± 560764",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 39204493,
            "range": "± 1652486",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 74654805,
            "range": "± 3320383",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 158644283,
            "range": "± 5602546",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 402584068,
            "range": "± 12516929",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 365543707,
            "range": "± 2434703",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 774183625,
            "range": "± 3035734",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1612825566,
            "range": "± 7261155",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3360633517,
            "range": "± 10701187",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7059319828,
            "range": "± 24622592",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 390372317,
            "range": "± 1302567",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 805640471,
            "range": "± 3562802",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1692172362,
            "range": "± 3510480",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3491686087,
            "range": "± 19636439",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7348042954,
            "range": "± 12157141",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 1114,
            "range": "± 17",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 142446,
            "range": "± 160",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 774,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 406,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 1220,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 108282,
            "range": "± 376",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 153870,
            "range": "± 515",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1578704,
            "range": "± 32011",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 858552,
            "range": "± 4529",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1702629,
            "range": "± 19172",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 2663,
            "range": "± 2613",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 201902,
            "range": "± 3115",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 1132,
            "range": "± 10",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 13,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 14,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 47,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 61,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 29,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 11357,
            "range": "± 460",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "nicole.graus@lambdaclass.com",
            "name": "Nicole Graus",
            "username": "nicole-graus"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "762daeb337fd21191c514ffcd398b51d7208b5e3",
          "message": "fix link (#1052)",
          "timestamp": "2025-10-27T15:54:10Z",
          "tree_id": "38d9d80e3ae9784c0f811ed8ed80ea33696dc3af",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/762daeb337fd21191c514ffcd398b51d7208b5e3"
        },
        "date": 1761582236999,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 324602915,
            "range": "± 234628",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 370683155,
            "range": "± 921586",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 278020704,
            "range": "± 3374680",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 682977162,
            "range": "± 1460159",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 780804267,
            "range": "± 1292401",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1431091774,
            "range": "± 4722553",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1637527791,
            "range": "± 4552033",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1224147816,
            "range": "± 888916",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2990094388,
            "range": "± 3467514",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3439084034,
            "range": "± 30141218",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6263569665,
            "range": "± 6275905",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7288746098,
            "range": "± 16693646",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5373165404,
            "range": "± 3653270",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7823904,
            "range": "± 6837",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7862765,
            "range": "± 3635",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 10358353,
            "range": "± 48972",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 10413866,
            "range": "± 21368",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 17936744,
            "range": "± 33064",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 17951752,
            "range": "± 22620",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 27998817,
            "range": "± 392304",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 27188450,
            "range": "± 252420",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 36433239,
            "range": "± 157007",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 36703375,
            "range": "± 137965",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 67074439,
            "range": "± 465092",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 66821414,
            "range": "± 427415",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 73676610,
            "range": "± 72040",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 73260795,
            "range": "± 155635",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 140085368,
            "range": "± 1694492",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 141353068,
            "range": "± 3234017",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 146818070,
            "range": "± 180678",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 146706529,
            "range": "± 92217",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 280208438,
            "range": "± 3884471",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 280130737,
            "range": "± 803742",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 15950586,
            "range": "± 42551",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 32887501,
            "range": "± 83654",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 66436958,
            "range": "± 371453",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 135233762,
            "range": "± 1779980",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 349207971,
            "range": "± 1816323",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 357384381,
            "range": "± 892621",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 760906384,
            "range": "± 4498981",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1600783894,
            "range": "± 1839298",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3335351012,
            "range": "± 2637257",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7044910654,
            "range": "± 5611240",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 383612613,
            "range": "± 633508",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 803257524,
            "range": "± 2166145",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1681506845,
            "range": "± 2771614",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3489377969,
            "range": "± 2883163",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7331910771,
            "range": "± 6829252",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 45,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 382,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 103,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 61,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 172,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 431,
            "range": "± 26",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 149119,
            "range": "± 211",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1575772,
            "range": "± 13726",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 835948,
            "range": "± 1617",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1671203,
            "range": "± 62196",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 547,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 2091,
            "range": "± 33",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 45,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 20,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 6,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 159,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 13109,
            "range": "± 841",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 333,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "nicole.graus@lambdaclass.com",
            "name": "Nicole Graus",
            "username": "nicole-graus"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "fade1668e234e575138056df99a96469e2f3d0dd",
          "message": "Implement HasDefaultTranscript for baby bear u32 degree 4 extension field (#1055)\n\n* implement HasDefaultTranscript for degree 4  baby bear u32\n\n* fix lint\n\n* fix clippy",
          "timestamp": "2025-12-10T14:35:46Z",
          "tree_id": "e08fc5ead7534c269f8625bb19afd969a17d32f7",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/fade1668e234e575138056df99a96469e2f3d0dd"
        },
        "date": 1765379135578,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 329259381,
            "range": "± 170664",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 375363779,
            "range": "± 981705",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 279579550,
            "range": "± 257626",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 690709284,
            "range": "± 3398743",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 797750355,
            "range": "± 3153216",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1447985294,
            "range": "± 1487267",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1661230128,
            "range": "± 2479831",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1227907587,
            "range": "± 3742643",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 3017968352,
            "range": "± 4299935",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3460925092,
            "range": "± 10610846",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6316506173,
            "range": "± 5789717",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7285519863,
            "range": "± 19810879",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5377086721,
            "range": "± 8191740",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7806477,
            "range": "± 9812",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7862283,
            "range": "± 6166",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 11518150,
            "range": "± 525871",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 11368146,
            "range": "± 725790",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 18286683,
            "range": "± 63076",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 18254642,
            "range": "± 56049",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 34000634,
            "range": "± 2224177",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 33162877,
            "range": "± 1187917",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 36746065,
            "range": "± 143475",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 36790808,
            "range": "± 198123",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 71917048,
            "range": "± 2475023",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 73226399,
            "range": "± 1276664",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 74477374,
            "range": "± 381499",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 73862976,
            "range": "± 319301",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 147661610,
            "range": "± 1226086",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 148563739,
            "range": "± 1473218",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 148675450,
            "range": "± 919179",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 148514398,
            "range": "± 344326",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 302831100,
            "range": "± 2896022",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 298804076,
            "range": "± 2281683",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 18969703,
            "range": "± 557282",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 38199448,
            "range": "± 502562",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 75562977,
            "range": "± 756339",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 153886602,
            "range": "± 3409392",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 369325593,
            "range": "± 4927486",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 372084137,
            "range": "± 1568904",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 786721796,
            "range": "± 6714204",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1636364491,
            "range": "± 4129420",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3392721170,
            "range": "± 7057590",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7142151105,
            "range": "± 28328583",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 393501449,
            "range": "± 1722544",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 822561113,
            "range": "± 5070804",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1717966672,
            "range": "± 6552560",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3591037939,
            "range": "± 33375102",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7469020805,
            "range": "± 98043184",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 23,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 92,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 63,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 29,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 91,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 168,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 148918,
            "range": "± 293",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1576121,
            "range": "± 14964",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 835896,
            "range": "± 2109",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1840219,
            "range": "± 12558",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 271,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 667,
            "range": "± 5",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 22,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 14,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 10372,
            "range": "± 38",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 89,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 10300,
            "range": "± 470",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 70,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 9997,
            "range": "± 703",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "56092489+ColoCarletti@users.noreply.github.com",
            "name": "Joaquin Carletti",
            "username": "ColoCarletti"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e17bcc29450cdd6d908e82ca998b20823253722a",
          "message": "Add copy to BabyBear u32 Extension Field struct (#1056)\n\n* add copy\n\n* fix clippy\n\n* fix clippy\n\n* fix clippy",
          "timestamp": "2025-12-11T20:08:28Z",
          "tree_id": "6f90e690ace7a93615dc9147c7a988b6a191ad9d",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/e17bcc29450cdd6d908e82ca998b20823253722a"
        },
        "date": 1765485385936,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 325406357,
            "range": "± 1449160",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 367525199,
            "range": "± 1396848",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 278242297,
            "range": "± 367994",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 685018077,
            "range": "± 363005",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 777713631,
            "range": "± 1089316",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1432844587,
            "range": "± 902600",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1639458180,
            "range": "± 1564130",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1223808194,
            "range": "± 721911",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2994213970,
            "range": "± 1781865",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3430968624,
            "range": "± 8243283",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6271091231,
            "range": "± 5127099",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7276624083,
            "range": "± 20112865",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5369050830,
            "range": "± 6050521",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 8065219,
            "range": "± 18642",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 8081004,
            "range": "± 15222",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 10448562,
            "range": "± 161469",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 11560672,
            "range": "± 337863",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 18474816,
            "range": "± 83177",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 18648780,
            "range": "± 70076",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 28949114,
            "range": "± 851443",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 28173462,
            "range": "± 777969",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 37400728,
            "range": "± 92193",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 37308523,
            "range": "± 111876",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 69009334,
            "range": "± 879822",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 68155552,
            "range": "± 580314",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 75255742,
            "range": "± 102992",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 75080610,
            "range": "± 183416",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 142452756,
            "range": "± 795215",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 142616483,
            "range": "± 734250",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 150887530,
            "range": "± 240194",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 150814659,
            "range": "± 145358",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 286192501,
            "range": "± 977218",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 287260855,
            "range": "± 1300607",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 15826397,
            "range": "± 222037",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 33707776,
            "range": "± 399382",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 67798973,
            "range": "± 350093",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 137835337,
            "range": "± 1438211",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 355163739,
            "range": "± 1454605",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 359807250,
            "range": "± 1312720",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 764402117,
            "range": "± 2366022",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1605633261,
            "range": "± 4717535",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3338491693,
            "range": "± 3021876",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7055355111,
            "range": "± 6787329",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 389630755,
            "range": "± 1533826",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 813572698,
            "range": "± 4922383",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1695252408,
            "range": "± 6333615",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3498384910,
            "range": "± 3543894",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7358980115,
            "range": "± 8331080",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 111,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 1779,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 184,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 123,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 315,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 1907,
            "range": "± 8",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 151613,
            "range": "± 414",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1576668,
            "range": "± 5092",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 832975,
            "range": "± 1700",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1694880,
            "range": "± 11051",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 949,
            "range": "± 551",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 7635,
            "range": "± 118",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 114,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 14,
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
          "distinct": true,
          "id": "5998a043d7613282977a92fa4bbb8456b4a68117",
          "message": "Refactor air for dyn (#1058)",
          "timestamp": "2025-12-22T17:26:08Z",
          "tree_id": "0c2de7ed3a51bd2cb5c0011de4c49287d149fed2",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/5998a043d7613282977a92fa4bbb8456b4a68117"
        },
        "date": 1766426148646,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 324724647,
            "range": "± 449404",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 372250927,
            "range": "± 923871",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 277838295,
            "range": "± 272783",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 682101775,
            "range": "± 473121",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 792027251,
            "range": "± 1638737",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1430810534,
            "range": "± 1782538",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1680508438,
            "range": "± 3361339",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1223566036,
            "range": "± 2708180",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2989018979,
            "range": "± 705364",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3511929369,
            "range": "± 5720551",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6259761299,
            "range": "± 2466897",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7442062824,
            "range": "± 8018389",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5366228136,
            "range": "± 2890874",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 8156053,
            "range": "± 6759",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 8156823,
            "range": "± 9313",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 10647376,
            "range": "± 18550",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 10644468,
            "range": "± 27822",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 18534816,
            "range": "± 73583",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 18507097,
            "range": "± 53521",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 26483422,
            "range": "± 307845",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 27170659,
            "range": "± 384524",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 37362158,
            "range": "± 26760",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 37397085,
            "range": "± 224019",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 67311486,
            "range": "± 1152236",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 66350858,
            "range": "± 662558",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 75482996,
            "range": "± 42785",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 75456592,
            "range": "± 36889",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 139584842,
            "range": "± 924681",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 140670216,
            "range": "± 848827",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 151568234,
            "range": "± 435057",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 151459889,
            "range": "± 144975",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 281865282,
            "range": "± 662773",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 281986729,
            "range": "± 597576",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 14425380,
            "range": "± 286713",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 32808339,
            "range": "± 363945",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 64888193,
            "range": "± 202035",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 133349058,
            "range": "± 3284424",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 355152160,
            "range": "± 3142192",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 353300218,
            "range": "± 1740276",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 756602876,
            "range": "± 1030408",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1597912231,
            "range": "± 3195584",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3329693128,
            "range": "± 4514017",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7017592094,
            "range": "± 28216237",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 381658121,
            "range": "± 1477383",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 810895684,
            "range": "± 3499205",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1699091898,
            "range": "± 7235972",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3516212716,
            "range": "± 4940657",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7396557256,
            "range": "± 7411707",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 254,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 8123,
            "range": "± 63",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 307,
            "range": "± 7",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 181,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 462,
            "range": "± 23",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 6581,
            "range": "± 111",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 149750,
            "range": "± 460",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1568963,
            "range": "± 8528",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 845524,
            "range": "± 3884",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1817579,
            "range": "± 42040",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 1314,
            "range": "± 1266",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 23447,
            "range": "± 649",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 263,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 17,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 6,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 131,
            "range": "± 7",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 11576,
            "range": "± 544",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 190,
            "range": "± 5",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
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
          "id": "bc6dd545e9daf5bac32e24467279b4b90367744c",
          "message": "Make AIR Dynamic, and embed them along the Prover (#1059)\n\n* make IsStarkTranscript dyn\n\n* multi_prove first draft\n\n* fix\n\n* Make PublicInputs generic again\n\n* fixes\n\n* re-enable debug code\n\n* restore commented code\n\n* clippy\n\n* fix parallel feature : swap &dyn AIR -> Box<dyn AIR>\n\n* get rid of Box\n\n* remove uneeded code\n\n* cleanup\n\n* cleanup\n\n* cleanup\n\n* Remove duplicate AsBytes trait bound\n\n* Remove uneeded trait bounds\n\n* Remove uneeded trait bounds\n\n* Remove Send + Sync bounds from PublicInputs generic PI\n\n* Remove Send + Sync bounds from PublicInputs generic PI\n\n* Apply same trait refactor to IsStarkVerifier\n\n* Remove proof_options argument from verify\n\n* Add multi_verify\n\n---------\n\nCo-authored-by: fmoletta <fedemoletta@hotmail.com>",
          "timestamp": "2026-01-05T18:18:24Z",
          "tree_id": "1b3b2c961402d83ab01ffcf92a5f577782fa9a53",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/bc6dd545e9daf5bac32e24467279b4b90367744c"
        },
        "date": 1767638851325,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 323862335,
            "range": "± 1171619",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 367718544,
            "range": "± 1392692",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 276231131,
            "range": "± 88572",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 681703306,
            "range": "± 601894",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 781667595,
            "range": "± 2586657",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1429458047,
            "range": "± 794209",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1650271155,
            "range": "± 1717573",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1217710128,
            "range": "± 460324",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2987988502,
            "range": "± 1453443",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3452850789,
            "range": "± 2351416",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6258219352,
            "range": "± 1871858",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7314189707,
            "range": "± 5963561",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5345794983,
            "range": "± 2117543",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 8080674,
            "range": "± 3563",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 8091097,
            "range": "± 7541",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 10495095,
            "range": "± 7830",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 10504075,
            "range": "± 19453",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 17994703,
            "range": "± 22353",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 18032860,
            "range": "± 38233",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 23764830,
            "range": "± 150482",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 24029969,
            "range": "± 134656",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 37109895,
            "range": "± 59555",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 36885439,
            "range": "± 42345",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 64727494,
            "range": "± 226921",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 64758643,
            "range": "± 225171",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 74660075,
            "range": "± 45349",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 74913720,
            "range": "± 39341",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 138473517,
            "range": "± 187177",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 139125395,
            "range": "± 446667",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 150885069,
            "range": "± 43112",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 150150423,
            "range": "± 155137",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 282529556,
            "range": "± 443384",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 280625971,
            "range": "± 580365",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 14568391,
            "range": "± 172779",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 32278367,
            "range": "± 107039",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 65512226,
            "range": "± 266290",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 133453488,
            "range": "± 510587",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 328709380,
            "range": "± 678512",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 351561101,
            "range": "± 726552",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 753929304,
            "range": "± 571886",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1590202551,
            "range": "± 1047511",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3324195650,
            "range": "± 1504223",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7024808199,
            "range": "± 6859644",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 380769692,
            "range": "± 548259",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 795857485,
            "range": "± 442585",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1677480610,
            "range": "± 1038150",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3481691284,
            "range": "± 1980752",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7331646469,
            "range": "± 5693316",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 20,
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
            "value": 62,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 28,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 90,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 152,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 149935,
            "range": "± 426",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1568168,
            "range": "± 9966",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 837928,
            "range": "± 1303",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1746608,
            "range": "± 6662",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 384,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 758,
            "range": "± 7",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 21,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 14,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 10419,
            "range": "± 25",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 89,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 10044,
            "range": "± 1525",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 66,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
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
          "id": "f6018beef471654b69170799b24f25af4b24def5",
          "message": "Refactor/remove naive pairing (#1068)\n\n* fix: Replace unwraps with proper error handling across provers\n\nThis commit improves error handling across the proving systems:\n\nSTARK:\n- Domain functions now return Result<Domain, ProvingError>\n- FRI commit/query phases return proper Results\n- Verifier functions handle errors gracefully (return false)\n- Extended ProvingError with FFTError, BatchInversionFailed, MerkleTreeError\n\nPLONK:\n- CommonPreprocessedInput::from_constraint_system returns Result\n- Extended ProverError with FFTError, PrimitiveRootNotFound, etc.\n\nGroth16:\n- Prover::prove returns Result<Proof, Groth16Error>\n- QuadraticArithmeticProgram::calculate_h_coefficients returns Result\n- Created Groth16Error enum with FFT, MSM, QAP error variants\n\nShared:\n- P448 Goldilocks: Implemented ByteConversion for U56x8\n- math/traits.rs: Added serialize_with_length/deserialize_with_length helpers\n- Consolidated serialization code between PLONK and Groth16 provers\n\nSumcheck/GKR: Already had proper error handling (unwraps only in tests)\n\n* refactor: Remove naive pairing implementations from BN254\n\nRemove miller_naive, line_naive, and final_exponentiation_naive functions\nthat were kept for educational/benchmark purposes. Benchmarks should\ncompare against arkworks using hyperfine instead.\n\n- Remove miller_naive (~31 lines)\n- Remove line_naive (~33 lines)\n- Remove final_exponentiation_naive (~42 lines)\n- Update benchmarks to remove naive function calls\n- Simplify documentation comments\n\nTotal: ~135 lines removed\n\n* fix: Handle Result from Prover::prove in circom example\n\nAlso run cargo fmt on prover files.\n\n* fix compiling errors. Fix clippy. Update documentation\n\n---------\n\nCo-authored-by: Nicole <nicole.graus@lambdaclass.com>",
          "timestamp": "2026-01-14T19:29:46Z",
          "tree_id": "e15ac3f2df82c857a7321f3502e3e1dd2c877114",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/f6018beef471654b69170799b24f25af4b24def5"
        },
        "date": 1768420597655,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 325261718,
            "range": "± 237088",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 371421262,
            "range": "± 2887980",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 279431569,
            "range": "± 1513071",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 682915896,
            "range": "± 259731",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 779125266,
            "range": "± 1027955",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1430522578,
            "range": "± 897510",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1642607592,
            "range": "± 5836069",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1231748501,
            "range": "± 410114",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2988876398,
            "range": "± 3386379",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3451670304,
            "range": "± 9388562",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6254295794,
            "range": "± 2910050",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7289887339,
            "range": "± 25246919",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5389904234,
            "range": "± 3790108",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 8054088,
            "range": "± 5919",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 8058992,
            "range": "± 6276",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 10926553,
            "range": "± 335766",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 10810222,
            "range": "± 457578",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 18937319,
            "range": "± 126492",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 19055085,
            "range": "± 66967",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 31626508,
            "range": "± 1608672",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 30658414,
            "range": "± 1851022",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 38012432,
            "range": "± 112461",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 37991639,
            "range": "± 86371",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 71345811,
            "range": "± 835439",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 70712443,
            "range": "± 1279406",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 76273162,
            "range": "± 1033245",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 76253694,
            "range": "± 244397",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 146455325,
            "range": "± 861680",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 143768563,
            "range": "± 662183",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 151941971,
            "range": "± 419018",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 152278449,
            "range": "± 289983",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 289619689,
            "range": "± 1623492",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 288655730,
            "range": "± 1233901",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 15993530,
            "range": "± 537479",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 34895391,
            "range": "± 797578",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 70077346,
            "range": "± 2545706",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 141620845,
            "range": "± 1696457",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 345767582,
            "range": "± 4328436",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 362213473,
            "range": "± 2418195",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 767368436,
            "range": "± 1761458",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1604680463,
            "range": "± 3937622",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3341296510,
            "range": "± 5046745",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7048830349,
            "range": "± 5605770",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 385674310,
            "range": "± 1108222",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 806083993,
            "range": "± 2473218",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1684587852,
            "range": "± 2448645",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3486254280,
            "range": "± 5956877",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7356979946,
            "range": "± 13543067",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 12,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 31,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 56,
            "range": "± 8",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 26,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 82,
            "range": "± 8",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 52,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 152327,
            "range": "± 316",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1576504,
            "range": "± 11914",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 837903,
            "range": "± 3668",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1671913,
            "range": "± 19757",
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
          "id": "fecfac6b958e9b53506e1c6d3f44527c8aa78e4c",
          "message": "Refactor/babybear u32 only (#1069)\n\n* refactor: Consolidate BabyBear to u32-only implementation\n\nRemove the u64 BabyBear implementation and keep only the efficient u32\nversion with quartic extension. This simplifies the codebase by\neliminating redundant implementations.\n\nChanges:\n- Remove babybear.rs (u64 impl) and rename babybear_u32.rs to babybear.rs\n- Remove quadratic_babybear.rs (unused u64 quadratic extension)\n- Remove quartic_babybear.rs (u64 impl) and rename quartic_babybear_u32.rs\n- Rename Degree4BabyBearU32ExtensionField to Degree4BabyBearExtensionField\n- Update all imports across math, crypto, and provers crates\n- Update benchmarks to use the new unified type names\n- Update README documentation with corrected file paths\n\n* chore: Run cargo fmt\n\n* fix typos\n\n---------\n\nCo-authored-by: Nicole <nicole.graus@lambdaclass.com>\nCo-authored-by: jotabulacios <jbulacios@fi.uba.ar>",
          "timestamp": "2026-01-15T21:05:57Z",
          "tree_id": "b9d3b12634ebae5580c580af2abc78870b709751",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/fecfac6b958e9b53506e1c6d3f44527c8aa78e4c"
        },
        "date": 1768512913871,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 326542878,
            "range": "± 794334",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 377415257,
            "range": "± 2450932",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 279536591,
            "range": "± 195392",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 685031108,
            "range": "± 541443",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 791538443,
            "range": "± 5653362",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1432940643,
            "range": "± 2188573",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1683528789,
            "range": "± 13914689",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1226927066,
            "range": "± 1204651",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2993428264,
            "range": "± 3322879",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3478309294,
            "range": "± 19053125",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6265506675,
            "range": "± 4247227",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7362690254,
            "range": "± 54850179",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5378347890,
            "range": "± 1995504",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 8074618,
            "range": "± 7763",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 8081958,
            "range": "± 9601",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 12542109,
            "range": "± 630821",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 13338638,
            "range": "± 1002775",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 18809879,
            "range": "± 74017",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 18976021,
            "range": "± 100889",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 35590065,
            "range": "± 2241727",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 35305253,
            "range": "± 2035617",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 38129526,
            "range": "± 79006",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 38203616,
            "range": "± 230892",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 73573570,
            "range": "± 2091386",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 74111037,
            "range": "± 1292708",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 76131132,
            "range": "± 248673",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 76206587,
            "range": "± 162281",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 146893150,
            "range": "± 1277769",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 151116102,
            "range": "± 3738126",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 152291803,
            "range": "± 302520",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 152214109,
            "range": "± 565929",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 294388587,
            "range": "± 2036583",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 304447949,
            "range": "± 7742413",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 17127511,
            "range": "± 846645",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 36211238,
            "range": "± 1616583",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 72023124,
            "range": "± 1736422",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 158990337,
            "range": "± 7345500",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 383150973,
            "range": "± 10096187",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 364352182,
            "range": "± 2305192",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 777381254,
            "range": "± 3540533",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1625900069,
            "range": "± 4347791",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3372609463,
            "range": "± 15875053",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7096001391,
            "range": "± 22147984",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 389550733,
            "range": "± 2238860",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 809844123,
            "range": "± 2784337",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1699335174,
            "range": "± 4906288",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3533224836,
            "range": "± 8315013",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7383298525,
            "range": "± 28752722",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 12,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 29,
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
            "value": 26,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 83,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 52,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 149490,
            "range": "± 1160",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1578047,
            "range": "± 5186",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 834743,
            "range": "± 1080",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1682992,
            "range": "± 12914",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 273,
            "range": "± 24",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 273,
            "range": "± 8",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 10,
            "range": "± 36",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 13,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 74,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 86,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 9956,
            "range": "± 679",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 46,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
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
          "id": "9fe3ac2853329121594705a22b4cdb15d51226d7",
          "message": "refactor: Extract common serialization helpers to traits.rs (#1070)\n\nMove duplicated serialization code from Groth16 and PLONK provers\nto shared helper functions in lambdaworks-math:\n\n- serialize_with_length: Serialize with u32 big-endian length prefix\n- deserialize_with_length: Deserialize Deserializable types with length prefix\n- deserialize_field_element_with_length: Deserialize FieldElements with length prefix\n\nThis removes ~50 lines of duplicated code between provers and provides\na consistent serialization format for proof components.\n\nCo-authored-by: jotabulacios <jbulacios@fi.uba.ar>",
          "timestamp": "2026-01-16T13:23:06Z",
          "tree_id": "7272d56cd5d6e0253c429d55433fa502538c8fbd",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/9fe3ac2853329121594705a22b4cdb15d51226d7"
        },
        "date": 1768571548654,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 326085088,
            "range": "± 5076574",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 373700258,
            "range": "± 1007473",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 279260680,
            "range": "± 261639",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 685308544,
            "range": "± 537760",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 787891039,
            "range": "± 1644226",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1434449316,
            "range": "± 1648985",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1666167172,
            "range": "± 7019988",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1225662608,
            "range": "± 823900",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2995602174,
            "range": "± 4813067",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3466007475,
            "range": "± 22346535",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6270591210,
            "range": "± 12024689",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7317650004,
            "range": "± 22133592",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5376563679,
            "range": "± 12212512",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 8085459,
            "range": "± 51419",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 8079962,
            "range": "± 5070",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 14302184,
            "range": "± 2051171",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 11867528,
            "range": "± 1170903",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 18934063,
            "range": "± 74188",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 18899586,
            "range": "± 79808",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 33900003,
            "range": "± 1799930",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 31064259,
            "range": "± 667127",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 37665053,
            "range": "± 113260",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 37504547,
            "range": "± 85421",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 72779723,
            "range": "± 2098147",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 73596607,
            "range": "± 2431850",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 75918900,
            "range": "± 268137",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 75807800,
            "range": "± 117571",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 147949749,
            "range": "± 1328031",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 147242613,
            "range": "± 575077",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 153322042,
            "range": "± 769935",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 152469835,
            "range": "± 245727",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 297137014,
            "range": "± 2008368",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 296580328,
            "range": "± 7914811",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 17018234,
            "range": "± 231040",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 36474079,
            "range": "± 360661",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 71946908,
            "range": "± 496858",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 146799652,
            "range": "± 1128372",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 369885843,
            "range": "± 2192296",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 365440762,
            "range": "± 2684923",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 772412288,
            "range": "± 3784243",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1611265291,
            "range": "± 4566217",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3367046281,
            "range": "± 21788562",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7122501319,
            "range": "± 19595248",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 396928501,
            "range": "± 2585118",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 820717531,
            "range": "± 1336090",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1715249648,
            "range": "± 1921187",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3537571282,
            "range": "± 6425114",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7449623168,
            "range": "± 17869434",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 247,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 7931,
            "range": "± 95",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 287,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 167,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 495,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 7108,
            "range": "± 26",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 148991,
            "range": "± 815",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1568265,
            "range": "± 5476",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 833969,
            "range": "± 1489",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1707391,
            "range": "± 14220",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 1204,
            "range": "± 1194",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 22498,
            "range": "± 481",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 253,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 14,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 10388,
            "range": "± 38",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 90,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 10182,
            "range": "± 245",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 69,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "jbulacios@fi.uba.ar",
            "name": "jotabulacios",
            "username": "jotabulacios"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f0b082b7217978bdc839fd8208a4bcee6e84ad3b",
          "message": "Curve optimizations (#1076)\n\n* split old PR. Take optimization for curves\n\n* fix clippy\n\n* add fuzzing tests for karabina and g2 prepared\n\n* add g2_prepared in toml\n\n---------\n\nCo-authored-by: Nicole Graus <nicole.graus@lambdaclass.com>",
          "timestamp": "2026-01-16T20:23:10Z",
          "tree_id": "29dec1aa08b76f029de52604111e5d97dec24c7a",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/f0b082b7217978bdc839fd8208a4bcee6e84ad3b"
        },
        "date": 1768596798491,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 324882668,
            "range": "± 1696842",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 369554968,
            "range": "± 379288",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 278475696,
            "range": "± 5038688",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 682956091,
            "range": "± 2412728",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 778277088,
            "range": "± 8727165",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1429418350,
            "range": "± 916593",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1640821433,
            "range": "± 1981052",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1223261152,
            "range": "± 610489",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2988008038,
            "range": "± 7778982",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3444361367,
            "range": "± 11220418",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6254071763,
            "range": "± 9542742",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7273788103,
            "range": "± 38751755",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5369818983,
            "range": "± 23136228",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 8065007,
            "range": "± 10145",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 8077874,
            "range": "± 9356",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 10529449,
            "range": "± 282831",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 10783136,
            "range": "± 401755",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 18796726,
            "range": "± 389138",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 18789168,
            "range": "± 95151",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 29877701,
            "range": "± 1175592",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 29853891,
            "range": "± 2070385",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 37540722,
            "range": "± 90411",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 37521998,
            "range": "± 303029",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 71361028,
            "range": "± 1581314",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 70470813,
            "range": "± 821429",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 75754478,
            "range": "± 147296",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 75437380,
            "range": "± 274004",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 145796958,
            "range": "± 796294",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 145650652,
            "range": "± 664484",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 152208327,
            "range": "± 2157135",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 151987736,
            "range": "± 214818",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 296671441,
            "range": "± 10491509",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 301359277,
            "range": "± 2114985",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 16797978,
            "range": "± 176094",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 35335105,
            "range": "± 586541",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 69918315,
            "range": "± 1384942",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 143012294,
            "range": "± 1705950",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 368252631,
            "range": "± 1715179",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 363066717,
            "range": "± 2920152",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 768942174,
            "range": "± 1653123",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1607200473,
            "range": "± 3600154",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3353980948,
            "range": "± 14752080",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7072877795,
            "range": "± 29476632",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 389650477,
            "range": "± 5881145",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 807704014,
            "range": "± 2570723",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1690188210,
            "range": "± 16643830",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3500412335,
            "range": "± 11811955",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7358141535,
            "range": "± 14612704",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 103,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 1683,
            "range": "± 46",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 180,
            "range": "± 13",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 115,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 300,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 1623,
            "range": "± 116",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 149486,
            "range": "± 252",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1568101,
            "range": "± 8317",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 839985,
            "range": "± 20802",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1729594,
            "range": "± 36171",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 749,
            "range": "± 302",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 7107,
            "range": "± 652",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 110,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 13,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 13,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 49,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 60,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 29,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 10671,
            "range": "± 685",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "jbulacios@fi.uba.ar",
            "name": "jotabulacios",
            "username": "jotabulacios"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "987e7f5243ad00b7c14cffa079aa2a72a4d0098b",
          "message": "Field optimizations (#1078)\n\n* take field optimizations from PR-1066\n\n* add missing lines\n\n* restore goldilocks version\n\n* add fuzzing tests\n\n---------\n\nCo-authored-by: Diego K <43053772+diegokingston@users.noreply.github.com>\nCo-authored-by: Nicole <nicole.graus@lambdaclass.com>",
          "timestamp": "2026-01-19T14:28:44Z",
          "tree_id": "d621b918ea3648ea6d2fe8569f52e2db3933b2a7",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/987e7f5243ad00b7c14cffa079aa2a72a4d0098b"
        },
        "date": 1768834694420,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 323787094,
            "range": "± 314499",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 365122512,
            "range": "± 830255",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 277562110,
            "range": "± 3712501",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 681860013,
            "range": "± 225524",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 778796462,
            "range": "± 3809771",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1431241580,
            "range": "± 1545898",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1643241199,
            "range": "± 7579821",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1224346111,
            "range": "± 906482",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2993419192,
            "range": "± 2275558",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3438092515,
            "range": "± 11777479",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6266388249,
            "range": "± 2860807",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7256015590,
            "range": "± 17189791",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5363577404,
            "range": "± 5780819",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 8220448,
            "range": "± 1892",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 8291672,
            "range": "± 5729",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 15589753,
            "range": "± 354454",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 11962498,
            "range": "± 1226987",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 19054372,
            "range": "± 54713",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 19202596,
            "range": "± 99008",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 34605935,
            "range": "± 709970",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 30898944,
            "range": "± 450765",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 38362065,
            "range": "± 46919",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 38953783,
            "range": "± 75656",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 74859764,
            "range": "± 209435",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 75573642,
            "range": "± 628844",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 78325658,
            "range": "± 112091",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 77762196,
            "range": "± 707793",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 149916883,
            "range": "± 1584696",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 150536607,
            "range": "± 664118",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 153065580,
            "range": "± 117536",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 153045314,
            "range": "± 274763",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 294558487,
            "range": "± 1317238",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 295583812,
            "range": "± 1269007",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 18520131,
            "range": "± 333479",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 37008083,
            "range": "± 367955",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 72902092,
            "range": "± 1352123",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 143153426,
            "range": "± 4750304",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 346721132,
            "range": "± 2139262",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 358729342,
            "range": "± 1975018",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 759614788,
            "range": "± 4814787",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1600586798,
            "range": "± 26913832",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3338736149,
            "range": "± 9058997",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7025434426,
            "range": "± 28091736",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 378138495,
            "range": "± 387255",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 799483960,
            "range": "± 981203",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1683935179,
            "range": "± 2758565",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3493490007,
            "range": "± 2836102",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7329764662,
            "range": "± 4339619",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 20,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 88,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 63,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 28,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 91,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 143,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 149660,
            "range": "± 199",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1569463,
            "range": "± 11856",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 839051,
            "range": "± 29258",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1721754,
            "range": "± 6330",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 316,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 698,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 20,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 13,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 16,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 79,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 9542,
            "range": "± 403",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 36,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "jbulacios@fi.uba.ar",
            "name": "jotabulacios",
            "username": "jotabulacios"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "7fc2b01c3658de4409e358c14171d2804a887d2f",
          "message": "Fix slow sqrt (#1079)\n\n* fix slow square root for bn 254\n\n* fix length check\n\n* rename function\n\n* add more tests and fuzzers\n\n---------\n\nCo-authored-by: Diego K <43053772+diegokingston@users.noreply.github.com>",
          "timestamp": "2026-01-20T14:34:56Z",
          "tree_id": "46f98dcf39812f669131aaa7d187ae4a849fd3f1",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/7fc2b01c3658de4409e358c14171d2804a887d2f"
        },
        "date": 1768921473919,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 328646206,
            "range": "± 1385644",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 397408259,
            "range": "± 10548887",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 277394799,
            "range": "± 2994393",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 687687084,
            "range": "± 515425",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 813960299,
            "range": "± 8333346",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1441373210,
            "range": "± 1099871",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1727189983,
            "range": "± 14616239",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1220314586,
            "range": "± 903986",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 3012198090,
            "range": "± 3085427",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3593507693,
            "range": "± 33646531",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6296341177,
            "range": "± 2547001",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7686086861,
            "range": "± 163037262",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5352498595,
            "range": "± 5886747",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7932040,
            "range": "± 9758",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7975294,
            "range": "± 6206",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 10303351,
            "range": "± 181703",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 10390860,
            "range": "± 265837",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 18470932,
            "range": "± 110481",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 18483423,
            "range": "± 85635",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 31172400,
            "range": "± 927157",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 30834244,
            "range": "± 2117503",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 37477200,
            "range": "± 118324",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 37681302,
            "range": "± 339559",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 79466232,
            "range": "± 1302327",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 78591501,
            "range": "± 1183184",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 76211254,
            "range": "± 159169",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 75930577,
            "range": "± 215222",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 162372433,
            "range": "± 1851571",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 159997891,
            "range": "± 2265084",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 150863620,
            "range": "± 216175",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 151107354,
            "range": "± 451631",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 297301987,
            "range": "± 2993979",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 302902587,
            "range": "± 5903610",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 18177515,
            "range": "± 773971",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 38306664,
            "range": "± 1286589",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 76290803,
            "range": "± 2102312",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 154798716,
            "range": "± 3606525",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 394825874,
            "range": "± 6259389",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 362353925,
            "range": "± 1798905",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 777407213,
            "range": "± 5051559",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1634294884,
            "range": "± 5224032",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3393269009,
            "range": "± 10278943",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7155787741,
            "range": "± 19459655",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 392266058,
            "range": "± 3612192",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 824380095,
            "range": "± 7151160",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1731910216,
            "range": "± 16782272",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3573021066,
            "range": "± 25238934",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7496253853,
            "range": "± 27968563",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 258,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 8426,
            "range": "± 66",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 299,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 178,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 496,
            "range": "± 10",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 6800,
            "range": "± 490",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 149375,
            "range": "± 422",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1576024,
            "range": "± 5356",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 834985,
            "range": "± 3469",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1710216,
            "range": "± 22282",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 1339,
            "range": "± 1217",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 23033,
            "range": "± 289",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 267,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 22,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 6,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 164,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 12519,
            "range": "± 510",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 425,
            "range": "± 25",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
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
          "id": "1d8d182d9df572110d61f30cc2ed2d974b138875",
          "message": "Fix/doc warnings (#1074)\n\n* test: Add edge case tests for curves and field elements\n\nAdd comprehensive edge case tests for secp256k1, secq256k1, and secp256r1\ncurves covering:\n- Neutral element operations (addition, doubling)\n- Point negation and inverse\n- Scalar multiplication edge cases (0, 1)\n- Group axioms (associativity, commutativity, distributivity)\n\nAdd field element edge case tests covering:\n- Inverse of zero returns error\n- Arithmetic identities (zero, one)\n- Power operations (0, 1, 2)\n\nEach curve now has 17 tests (up from 9), and field element tests\nincreased from 28 to 40.\n\n* docs: Fix rustdoc warnings for URLs and math notation\n\n- Wrap URLs in angle brackets for proper hyperlink rendering\n- Escape math notation with backticks to prevent link parsing\n- Fix typos in comments (Godilocks -> Goldilocks, implemenation -> implementation)\n- Standardize URL reference format with \"See <url>\" pattern\n\nFiles updated:\n- polynomial/mod.rs: Fix bracket notation in doc comments\n- fft/cpu/fft.rs: Fix twiddle factor notation\n- field_extension files: Fix extension field definitions\n- curve.rs files: Fix eprint.iacr.org references\n- pairing.rs files: Fix hackmd.io and github.com references\n- compression.rs files: Fix gnark and zcash references\n- unsigned_integer files: Fix HAC algorithm references\n- p448_goldilocks: Fix sourceforge.net references\n\n---------\n\nCo-authored-by: jotabulacios <jbulacios@fi.uba.ar>",
          "timestamp": "2026-01-26T18:08:57Z",
          "tree_id": "786eedb9df6006d5465c2392685b8ef5895d9042",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/1d8d182d9df572110d61f30cc2ed2d974b138875"
        },
        "date": 1769452695534,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 330087180,
            "range": "± 244221",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 373742240,
            "range": "± 8194827",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 280021843,
            "range": "± 3348347",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 692492514,
            "range": "± 1521625",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 783767528,
            "range": "± 11527255",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1443216151,
            "range": "± 656890",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1648924216,
            "range": "± 13419036",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1228909873,
            "range": "± 8073429",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 3016511429,
            "range": "± 7233670",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3474824458,
            "range": "± 22089852",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6329168640,
            "range": "± 20537395",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7319814084,
            "range": "± 41367040",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5385799187,
            "range": "± 12543830",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7684633,
            "range": "± 18443",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7728725,
            "range": "± 96440",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 10433108,
            "range": "± 336967",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 10780887,
            "range": "± 470706",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 17962666,
            "range": "± 113489",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 17828470,
            "range": "± 119628",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 26366049,
            "range": "± 839984",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 30269603,
            "range": "± 1358368",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 36378752,
            "range": "± 499312",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 36630602,
            "range": "± 142764",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 73316702,
            "range": "± 2882320",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 69833927,
            "range": "± 3764029",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 73352963,
            "range": "± 229242",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 72546688,
            "range": "± 168523",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 145457772,
            "range": "± 3591505",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 151222137,
            "range": "± 2028027",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 145254678,
            "range": "± 329673",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 145465449,
            "range": "± 235791",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 300669360,
            "range": "± 3935217",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 304886470,
            "range": "± 15433671",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 14951303,
            "range": "± 1439794",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 38298948,
            "range": "± 1157587",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 77870343,
            "range": "± 2595774",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 154787535,
            "range": "± 9520019",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 390737220,
            "range": "± 13414786",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 367443258,
            "range": "± 3640490",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 780989097,
            "range": "± 5686658",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1639558327,
            "range": "± 28816030",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3401238278,
            "range": "± 17085881",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7185062390,
            "range": "± 41082966",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 396940339,
            "range": "± 5041007",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 828489771,
            "range": "± 9540594",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1714911384,
            "range": "± 5966905",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3557324116,
            "range": "± 13008686",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7501336068,
            "range": "± 26192641",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 43,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 382,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 104,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 60,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 172,
            "range": "± 5",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 423,
            "range": "± 5",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 142509,
            "range": "± 5968",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1578226,
            "range": "± 32596",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 797671,
            "range": "± 19299",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1742683,
            "range": "± 32267",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 539,
            "range": "± 37",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 2012,
            "range": "± 33",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 46,
            "range": "± 60",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 26,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 6,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 175,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 12705,
            "range": "± 1867",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 347,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
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
          "id": "6fea37754362b9de7ee6277bf6b9837fce33a11c",
          "message": "docs: Add Getting Started guide and Architecture overview (P0) (#1075)\n\n- Add comprehensive getting-started.md with:\n  - Prerequisites and quick installation instructions\n  - Four Hello World examples (fields, curves, Merkle trees, STARKs)\n  - Feature flags documentation\n  - Links to further resources\n\n- Add architecture.md with:\n  - Crate structure and dependency graph\n  - Module organization for math, crypto, and prover crates\n  - Design principles (generic over fields, no-std, modular backends)\n  - Guide for extending lambdaworks\n\n- Update SUMMARY.md to include new pages",
          "timestamp": "2026-01-26T18:10:36Z",
          "tree_id": "0e803ea203e19e491200fd42bcf8bda0192d2318",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/6fea37754362b9de7ee6277bf6b9837fce33a11c"
        },
        "date": 1769452810118,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 327234646,
            "range": "± 1987683",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 370268497,
            "range": "± 3872767",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 279682018,
            "range": "± 521143",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 687769280,
            "range": "± 969974",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 784915689,
            "range": "± 2660308",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1441495874,
            "range": "± 1294033",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1671228698,
            "range": "± 9720443",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1232343404,
            "range": "± 723312",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 3013172820,
            "range": "± 2730487",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3468439379,
            "range": "± 27614057",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6293664866,
            "range": "± 7483932",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7273908895,
            "range": "± 16450185",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5398191030,
            "range": "± 1114071",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7669686,
            "range": "± 4032",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7713886,
            "range": "± 3215",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 9848566,
            "range": "± 10748",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 9911646,
            "range": "± 38662",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 17486515,
            "range": "± 148686",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 17444367,
            "range": "± 48651",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 24642230,
            "range": "± 518281",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 24336594,
            "range": "± 453357",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 35562460,
            "range": "± 67254",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 35783901,
            "range": "± 73552",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 68566850,
            "range": "± 787013",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 69500870,
            "range": "± 901065",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 73271670,
            "range": "± 195654",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 72926276,
            "range": "± 127673",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 151411130,
            "range": "± 1433043",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 150629843,
            "range": "± 2008603",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 145562083,
            "range": "± 303895",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 145503314,
            "range": "± 207782",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 300360879,
            "range": "± 4132494",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 298651786,
            "range": "± 4073304",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 9357581,
            "range": "± 114706",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 33563707,
            "range": "± 1113222",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 79082091,
            "range": "± 2034992",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 160464586,
            "range": "± 3755613",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 395265887,
            "range": "± 5067199",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 359206077,
            "range": "± 2261250",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 770152914,
            "range": "± 6534191",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1626146531,
            "range": "± 7962512",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3392524512,
            "range": "± 4869724",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7150760386,
            "range": "± 10206960",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 390381523,
            "range": "± 1667353",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 817536724,
            "range": "± 4000908",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1716420822,
            "range": "± 4210833",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3562303170,
            "range": "± 9132502",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7478238629,
            "range": "± 12101094",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 525,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 33686,
            "range": "± 68",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 460,
            "range": "± 31",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 230,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 730,
            "range": "± 26",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 20407,
            "range": "± 522",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 144729,
            "range": "± 278",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1554984,
            "range": "± 16264",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 813404,
            "range": "± 22856",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1663288,
            "range": "± 10581",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 1675,
            "range": "± 1553",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 63983,
            "range": "± 1467",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 539,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 14,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 10363,
            "range": "± 48",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 93,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 10189,
            "range": "± 165",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 65,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
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
          "id": "8a956f5c9244430e28eedaa503a38b59de46b5c7",
          "message": "docs: improve README documentation with quick start guides (#1081)\n\n- Add Quick Start section to main README with field arithmetic,\n  elliptic curve, and polynomial examples\n- Enhance math crate README with overview, examples, supported\n  fields/curves tables, and feature documentation\n- Enhance crypto crate README with Poseidon hash and Merkle tree\n  examples, component tables, and usage guides\n\nCo-authored-by: Nicole Graus <nicole.graus@lambdaclass.com>",
          "timestamp": "2026-01-26T18:52:36Z",
          "tree_id": "a73b3eea31a36d697a62b57997c4c07897e54a2d",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/8a956f5c9244430e28eedaa503a38b59de46b5c7"
        },
        "date": 1769455255812,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 313815011,
            "range": "± 6340565",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 357483174,
            "range": "± 6482078",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 272295029,
            "range": "± 5620376",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 679609243,
            "range": "± 10590352",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 759245137,
            "range": "± 15475015",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1399122023,
            "range": "± 24130111",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1575969967,
            "range": "± 24057230",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1196222690,
            "range": "± 16603079",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2955792691,
            "range": "± 45795864",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3322233046,
            "range": "± 37647242",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6270329417,
            "range": "± 59797027",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7136066980,
            "range": "± 10806667",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5361424076,
            "range": "± 11705648",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7653425,
            "range": "± 47767",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7700908,
            "range": "± 29446",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 9696912,
            "range": "± 24976",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 9735989,
            "range": "± 46199",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 17286620,
            "range": "± 130546",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 17330222,
            "range": "± 67511",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 23951460,
            "range": "± 548461",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 23844356,
            "range": "± 358831",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 35238980,
            "range": "± 272628",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 35582600,
            "range": "± 188646",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 64822869,
            "range": "± 314298",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 65046300,
            "range": "± 389892",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 71936312,
            "range": "± 131329",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 71561152,
            "range": "± 449929",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 139888380,
            "range": "± 464130",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 137992661,
            "range": "± 374825",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 143489041,
            "range": "± 315136",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 144015334,
            "range": "± 78552",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 281375186,
            "range": "± 551148",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 281774124,
            "range": "± 413267",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 8798988,
            "range": "± 65980",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 29318295,
            "range": "± 261082",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 66335980,
            "range": "± 216691",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 135874129,
            "range": "± 677648",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 359605734,
            "range": "± 1353038",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 354262295,
            "range": "± 2417273",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 758365445,
            "range": "± 4449230",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1599608737,
            "range": "± 11632763",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3335255222,
            "range": "± 22383935",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7048876707,
            "range": "± 22666607",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 380206706,
            "range": "± 3438751",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 796790510,
            "range": "± 6513182",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1687175275,
            "range": "± 8722659",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3490267464,
            "range": "± 8106329",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7335270911,
            "range": "± 16803814",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 255,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 8171,
            "range": "± 94",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 273,
            "range": "± 8",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 174,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 473,
            "range": "± 9",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 6172,
            "range": "± 111",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 142195,
            "range": "± 3999",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1578297,
            "range": "± 6058",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 797231,
            "range": "± 2196",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1677811,
            "range": "± 15488",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 1263,
            "range": "± 1113",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 22182,
            "range": "± 513",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 265,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 13,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 75,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 88,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 10800,
            "range": "± 805",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 46,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
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
          "id": "eabdb5b24c60d42a867e5b9c51079647dfc96f38",
          "message": "Add FROST 2-of-2 threshold signature example (RFC 9591) (#1080)\n\n* Add FROST 2-of-2 threshold signature example (RFC 9591)\n\nImplements a threshold Schnorr signature scheme where two parties\ncan jointly sign messages without revealing their secret key shares.\n\nFeatures:\n- Shamir secret sharing with Lagrange interpolation\n- Two nonces per party (hiding + binding) for security\n- Binding factors to prevent nonce manipulation attacks\n- Domain-separated hashing for cryptographic hygiene\n\n* erro handling and posibble bugs\n\n---------\n\nCo-authored-by: jotabulacios <jbulacios@fi.uba.ar>\nCo-authored-by: Joaquin Carletti <56092489+ColoCarletti@users.noreply.github.com>",
          "timestamp": "2026-01-26T19:35:56Z",
          "tree_id": "62cdb379244e64027d8e64e99a5e25af27624cb3",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/eabdb5b24c60d42a867e5b9c51079647dfc96f38"
        },
        "date": 1769457873465,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 329493073,
            "range": "± 246676",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 373753951,
            "range": "± 1920549",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 279644480,
            "range": "± 213786",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 690854876,
            "range": "± 749342",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 788420842,
            "range": "± 7765004",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1446903343,
            "range": "± 2401623",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1644403721,
            "range": "± 14011686",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1229523381,
            "range": "± 1174386",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 3020107663,
            "range": "± 3713735",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3474248752,
            "range": "± 14101773",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6305370809,
            "range": "± 2922047",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7281428255,
            "range": "± 32692210",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5389987769,
            "range": "± 2732775",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7672590,
            "range": "± 7122",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7724025,
            "range": "± 5448",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 11368594,
            "range": "± 213313",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 11456231,
            "range": "± 1565158",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 18005347,
            "range": "± 138918",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 18011368,
            "range": "± 32009",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 32863921,
            "range": "± 1282963",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 31489510,
            "range": "± 545974",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 36282211,
            "range": "± 94400",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 36419889,
            "range": "± 65845",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 70763344,
            "range": "± 492773",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 71011576,
            "range": "± 1111008",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 73491216,
            "range": "± 333358",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 72452658,
            "range": "± 287826",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 142963291,
            "range": "± 9137923",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 143035388,
            "range": "± 1870483",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 144949442,
            "range": "± 343759",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 144543209,
            "range": "± 107414",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 285735337,
            "range": "± 2137622",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 285378528,
            "range": "± 967516",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 10068709,
            "range": "± 852002",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 36603099,
            "range": "± 1418319",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 73965674,
            "range": "± 2799870",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 149487214,
            "range": "± 4487071",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 387694092,
            "range": "± 4854085",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 368939302,
            "range": "± 1917263",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 777582128,
            "range": "± 1723832",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1624990549,
            "range": "± 3654980",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3385263961,
            "range": "± 6089882",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7112387378,
            "range": "± 10682100",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 399666932,
            "range": "± 825641",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 825382584,
            "range": "± 2463868",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1718724981,
            "range": "± 2045151",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3563385921,
            "range": "± 5769504",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7451843133,
            "range": "± 14891544",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 109,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 1768,
            "range": "± 5",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 178,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 107,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 300,
            "range": "± 17",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 1635,
            "range": "± 73",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 142202,
            "range": "± 1167",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1576483,
            "range": "± 7858",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 797515,
            "range": "± 2758",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1733085,
            "range": "± 13909",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 850,
            "range": "± 522",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 6923,
            "range": "± 209",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 117,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 14,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 10450,
            "range": "± 48",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 101,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 10283,
            "range": "± 389",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 65,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
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
          "id": "a37ec42c29f8057b4df506bd20d6f233503cb324",
          "message": "feat(examples): Add Reed-Solomon codes educational example (#1082)\n\n* feat(examples): Add Reed-Solomon codes educational example\n\nComprehensive educational example demonstrating Reed-Solomon codes:\n\n- RS code fundamentals: encoding, parameters, MDS property\n- Distance properties: Hamming distance, Singleton bound\n- Berlekamp-Welch unique decoding (up to (n-k)/2 errors)\n- Sudan's list decoding (n - sqrt(2nk) errors)\n- Guruswami-Sudan list decoding (n - sqrt(nk) errors)\n- Bivariate polynomial utilities and Roth-Ruckenstein root finding\n- Interactive demo showing all algorithms in action\n\n* fix(reed-solomon): Improve Roth-Ruckenstein root finding for zero polynomials\n\n- Handle zero polynomial case in find_univariate_roots_with_hints\n- Add BivariatePolynomial::coeffs() accessor for debugging\n- Add substitute_and_divide_debug and find_univariate_roots_debug helpers\n- Add Sage comparison test for validating against reference implementation\n- Expand search range for univariate roots to 2000\n\nKnown limitation: When Q'(0,y) is the zero polynomial (all field elements\nare roots), the algorithm returns a focused set of likely candidates.\nSome valid roots with large coefficients may be missed in this edge case.\n\n* feat(reed-solomon): Implement sophisticated Roth-Ruckenstein root finding\n\nAdd improved root-finding algorithm for Guruswami-Sudan decoder:\n\n- Implement domain-aware hint transformation through recursion\n- Add Lagrange interpolation to directly find candidate polynomials\n- Track domain points alongside hints for correct interpolation\n- Add lagrange_interpolate_polynomial for full polynomial construction\n- Add find_polynomial_roots_with_domain for domain-aware root finding\n\nKey improvements:\n1. Transform hints at each RR recursion level: new_hint[i] = (hint[i] - c) / domain[i]\n2. Use direct interpolation on received values to find candidate polynomials\n3. Filter domain alongside hints to maintain alignment\n\nThis enables finding candidates with large coefficients (like [1001, 1002, 3, 4])\nthat the original algorithm would miss when Q'(0, y) is the zero polynomial.\n\nAll 46 tests pass, including Sage comparison test.\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\n* fix clippy\n\n* fix clippy\n\n* add rsc to the list of examples in the readme\n\n---------\n\nCo-authored-by: Nicole <nicole.graus@lambdaclass.com>",
          "timestamp": "2026-01-27T12:32:35Z",
          "tree_id": "16e1a6bf7825a1f12561dd8b60dfb607b9dc0a6d",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/a37ec42c29f8057b4df506bd20d6f233503cb324"
        },
        "date": 1769519057443,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 327538843,
            "range": "± 352594",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 367734384,
            "range": "± 965981",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 278696497,
            "range": "± 696296",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 687520942,
            "range": "± 803695",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 772986917,
            "range": "± 1866873",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1439660242,
            "range": "± 7245089",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1620386711,
            "range": "± 6016741",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1225704546,
            "range": "± 964980",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 3007164697,
            "range": "± 1270273",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3385970729,
            "range": "± 3052990",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6288852963,
            "range": "± 13548649",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7175624318,
            "range": "± 19853294",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5377492807,
            "range": "± 1988511",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7657435,
            "range": "± 111821",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7711063,
            "range": "± 7424",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 11915151,
            "range": "± 346617",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 11394224,
            "range": "± 285550",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 17983203,
            "range": "± 197272",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 17949113,
            "range": "± 92215",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 29763063,
            "range": "± 1130224",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 32340198,
            "range": "± 1075457",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 35630918,
            "range": "± 132855",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 35957311,
            "range": "± 35166",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 68112641,
            "range": "± 330739",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 70050989,
            "range": "± 572010",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 72964294,
            "range": "± 131334",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 72059464,
            "range": "± 142874",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 139452612,
            "range": "± 1338806",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 138836049,
            "range": "± 1057610",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 144416668,
            "range": "± 104758",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 144241863,
            "range": "± 453664",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 277611729,
            "range": "± 975830",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 277562772,
            "range": "± 845478",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 10754520,
            "range": "± 394607",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 31406742,
            "range": "± 972488",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 65673053,
            "range": "± 588704",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 134866808,
            "range": "± 335535",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 352345698,
            "range": "± 1343188",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 360255144,
            "range": "± 1005192",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 764145258,
            "range": "± 1911298",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1604246270,
            "range": "± 1655714",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3346544420,
            "range": "± 12537803",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7045281893,
            "range": "± 13281670",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 387133214,
            "range": "± 484592",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 807329128,
            "range": "± 1260352",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1694750413,
            "range": "± 1644200",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3509278748,
            "range": "± 1434178",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7373373049,
            "range": "± 5589949",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 1082,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 138602,
            "range": "± 195",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 826,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 400,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 1153,
            "range": "± 19",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 80476,
            "range": "± 1561",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 144248,
            "range": "± 399",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1553739,
            "range": "± 12233",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 812620,
            "range": "± 1143",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1708025,
            "range": "± 7636",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 2621,
            "range": "± 4284",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 193641,
            "range": "± 5538",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 1101,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 11,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 14,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 46,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 57,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 26,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 9930,
            "range": "± 665",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
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
          "id": "73b96dc4780e3265b43bd2305073311e2a0cfb8b",
          "message": "feat: Add XMSS signature scheme example (#1083)\n\n* feat: Add XMSS signature scheme example\n\nEducational implementation of XMSS (eXtended Merkle Signature Scheme)\nfollowing RFC 8391, inspired by Ethereum's LeanSig proposal for\npost-quantum consensus signatures.\n\nFeatures:\n- Complete XMSS implementation with SHA-256 hasher\n- WOTS+ one-time signatures with Winternitz parameter w=16\n- Merkle tree of height h=10 (1024 signatures per key)\n- ~2.5 KB signatures, 64-byte public keys\n\nIncludes two example binaries:\n- basic_usage: demonstrates keygen -> sign -> verify flow\n- attack_demo: demonstrates WOTS+ forgery attack when index is reused\n\nReferences:\n- RFC 8391: https://datatracker.ietf.org/doc/html/rfc8391\n- LeanSig: https://eprint.iacr.org/2025/055\n\n* fix: prevent integer underflow in remaining_signatures\n\n* solve conflics\n\n* fmt\n\n* add signature serialization\n\n* emove unused with_params, add safety warning to idx field\n\n---------\n\nCo-authored-by: jotabulacios <jbulacios@fi.uba.ar>",
          "timestamp": "2026-01-28T14:53:00Z",
          "tree_id": "cd1a1125eaf3841c4da4cd5f1c595fd6feee2db1",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/73b96dc4780e3265b43bd2305073311e2a0cfb8b"
        },
        "date": 1769614450884,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 330770966,
            "range": "± 481935",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 670016302,
            "range": "± 7864355",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 294701559,
            "range": "± 2576841",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 694946558,
            "range": "± 624995",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 1614637669,
            "range": "± 18737717",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1450487758,
            "range": "± 1109310",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 3784817091,
            "range": "± 16151009",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1293948629,
            "range": "± 762522",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 3029247305,
            "range": "± 2023730",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 8427574583,
            "range": "± 46875788",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6323074837,
            "range": "± 4283530",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 18660539826,
            "range": "± 40212467",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5653058879,
            "range": "± 5061500",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 8079561,
            "range": "± 24250",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 8070853,
            "range": "± 33136",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 14042513,
            "range": "± 823797",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 14615419,
            "range": "± 802709",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 21523508,
            "range": "± 159870",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 21462842,
            "range": "± 151069",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 40164807,
            "range": "± 530858",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 40236538,
            "range": "± 827681",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 43557816,
            "range": "± 138098",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 43977439,
            "range": "± 427472",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 84380731,
            "range": "± 445412",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 85090794,
            "range": "± 258218",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 87977618,
            "range": "± 249287",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 88090176,
            "range": "± 257293",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 172897105,
            "range": "± 751561",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 169459414,
            "range": "± 737688",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 175000145,
            "range": "± 632227",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 175776672,
            "range": "± 347958",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 336843862,
            "range": "± 981007",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 337135551,
            "range": "± 2032453",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 18050338,
            "range": "± 341692",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 41719394,
            "range": "± 214761",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 82622970,
            "range": "± 1179037",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 160229081,
            "range": "± 471430",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 326311478,
            "range": "± 710357",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 387269333,
            "range": "± 2193520",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 815994490,
            "range": "± 1700227",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1700164669,
            "range": "± 1815632",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3532824525,
            "range": "± 7842954",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7312366877,
            "range": "± 10692739",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 418177937,
            "range": "± 2316223",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 867583211,
            "range": "± 2224595",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1803112737,
            "range": "± 4391941",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3728012053,
            "range": "± 6713833",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7700478659,
            "range": "± 9199873",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 5,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 13,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 45,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 21,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 65,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 29,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 149826,
            "range": "± 307",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1622711,
            "range": "± 18440",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 838480,
            "range": "± 2338",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1823813,
            "range": "± 31219",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 292,
            "range": "± 11",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 47,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 6,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 11,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 71,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 87,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 7626,
            "range": "± 472",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 43,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 0,
            "range": "± 0",
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
          "id": "56081c7c20a40b0a6329dcd205f7f06a7888ec78",
          "message": "fix(math): Guard polynomial division and xgcd against zero divisors (#1085)\n\n* fix(math): Guard polynomial division and xgcd against zero divisors\n\n- long_division_with_remainder now asserts the divisor is non-zero\n- xgcd now panics with clear message when both inputs are zero\n- Added documentation about panic conditions\n\n* fix(polynomial): add is_zero helper and correct zero-division checks\n\n* fix(math): Return errors instead of panicking on polynomial division by zero\n\n- Add PolynomialError enum with DivisionByZero and XgcdBothZero variants\n- Change long_division_with_remainder to return Result<(Self, Self), PolynomialError>\n- Change xgcd to return Result<(Self, Self, Self), PolynomialError>\n- Change div_with_ref to return Result<Self, PolynomialError>\n- Update Div trait impl to unwrap with clear panic message\n- Update all call sites to handle the Result appropriately\n- Convert #[should_panic] tests to Result-based assertions\n\n* fix(math): Replace unwrap with expect and improve comments\n\n- Replace unwrap() with expect() with descriptive messages\n- Add comments explaining why expect() is safe in each context\n- Update FFT tests to document non-zero divisor guarantee\n- Update benchmark to use expect() with clear message\n\n* fix compiling issue in rsc files: division by zero\n\n---------\n\nCo-authored-by: jotabulacios <jbulacios@fi.uba.ar>\nCo-authored-by: Nicole Graus <nicole.graus@lambdaclass.com>",
          "timestamp": "2026-01-28T17:53:07Z",
          "tree_id": "882261183b78a907b1a6687b1dc69c8560f1a4b2",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/56081c7c20a40b0a6329dcd205f7f06a7888ec78"
        },
        "date": 1769624886178,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 326975255,
            "range": "± 465799",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 503170967,
            "range": "± 24365842",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 292000011,
            "range": "± 462579",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 690352964,
            "range": "± 393733",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 1540996942,
            "range": "± 8888736",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1447501773,
            "range": "± 4021683",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 3669220540,
            "range": "± 14808265",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1294273673,
            "range": "± 701480",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 3027675201,
            "range": "± 1395765",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 8209177065,
            "range": "± 23704453",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6315686433,
            "range": "± 4609288",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 18184570342,
            "range": "± 194477006",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5644074895,
            "range": "± 1692644",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7918190,
            "range": "± 13380",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7964355,
            "range": "± 11090",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 11405837,
            "range": "± 574968",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 11228744,
            "range": "± 36699",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 20303889,
            "range": "± 337951",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 20289113,
            "range": "± 160525",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 29784166,
            "range": "± 529503",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 30781344,
            "range": "± 1145388",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 41931356,
            "range": "± 425280",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 42317873,
            "range": "± 113746",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 76195236,
            "range": "± 1186423",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 76557029,
            "range": "± 1125606",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 85284053,
            "range": "± 281943",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 85104874,
            "range": "± 185598",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 162219210,
            "range": "± 866896",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 165431547,
            "range": "± 543422",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 171203187,
            "range": "± 539610",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 171343379,
            "range": "± 229911",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 331099421,
            "range": "± 1019748",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 331715108,
            "range": "± 661613",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 10303894,
            "range": "± 760027",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 33760939,
            "range": "± 841387",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 79345820,
            "range": "± 1143031",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 159828362,
            "range": "± 1534803",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 320606459,
            "range": "± 856112",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 370467116,
            "range": "± 2453013",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 797219869,
            "range": "± 2368926",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1684895099,
            "range": "± 2881817",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3508360200,
            "range": "± 3264443",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7273037971,
            "range": "± 3246499",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 403764159,
            "range": "± 2957313",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 848852036,
            "range": "± 2113065",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1781583729,
            "range": "± 3132244",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3697557694,
            "range": "± 7271747",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7681996305,
            "range": "± 6421995",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 5,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 13,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 45,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 21,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 65,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 29,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 149607,
            "range": "± 1909",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1527201,
            "range": "± 15259",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 834845,
            "range": "± 2854",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1797733,
            "range": "± 21387",
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
          "id": "1c7c0ac2b9b4f20094e5691e2b20c6ff6e21d1a7",
          "message": "fix(crypto): Guard sample_u64 against zero upper_bound (#1088)\n\nAdd assertion to prevent division by zero when upper_bound is 0\nin both DefaultTranscript and StoneProverTranscript implementations.\n\nCo-authored-by: Nicole Graus <nicole.graus@lambdaclass.com>",
          "timestamp": "2026-01-28T19:04:29Z",
          "tree_id": "6858bf707246329f69d6eb9a8826ce901f91dc9e",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/1c7c0ac2b9b4f20094e5691e2b20c6ff6e21d1a7"
        },
        "date": 1769629030301,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 328165736,
            "range": "± 251943",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 370080198,
            "range": "± 2227481",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 282447111,
            "range": "± 380734",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 689298184,
            "range": "± 4026793",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 776792474,
            "range": "± 2158941",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1442033432,
            "range": "± 877032",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1621023256,
            "range": "± 6108114",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1241742581,
            "range": "± 1503636",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 3013237495,
            "range": "± 1270713",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3413543312,
            "range": "± 8833693",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6301459747,
            "range": "± 3870057",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7172176118,
            "range": "± 20221501",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5440159608,
            "range": "± 854493",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7666680,
            "range": "± 319091",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7721144,
            "range": "± 7966",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 12195396,
            "range": "± 599245",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 11345794,
            "range": "± 309505",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 17935667,
            "range": "± 72825",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 17820594,
            "range": "± 72098",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 32059145,
            "range": "± 1075837",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 30845552,
            "range": "± 858523",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 35833975,
            "range": "± 113557",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 36001956,
            "range": "± 125014",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 68664555,
            "range": "± 350187",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 69730342,
            "range": "± 535810",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 72523388,
            "range": "± 126371",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 72225555,
            "range": "± 263648",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 141059092,
            "range": "± 633694",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 140260158,
            "range": "± 327450",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 143951837,
            "range": "± 338073",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 144035988,
            "range": "± 366923",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 280083812,
            "range": "± 2291172",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 280583140,
            "range": "± 1682157",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 14129255,
            "range": "± 1204103",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 34086374,
            "range": "± 318498",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 69433209,
            "range": "± 845900",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 141001658,
            "range": "± 1899327",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 364523389,
            "range": "± 3888011",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 366753495,
            "range": "± 918959",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 772330928,
            "range": "± 2380114",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1620691706,
            "range": "± 2592426",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3367316609,
            "range": "± 2869796",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7119562909,
            "range": "± 14864407",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 394854275,
            "range": "± 2050919",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 822433353,
            "range": "± 2883756",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1719101910,
            "range": "± 5639724",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3546651591,
            "range": "± 7971790",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7426776359,
            "range": "± 12412748",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 21,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 85,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 62,
            "range": "± 5",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 28,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 90,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 158,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 144874,
            "range": "± 288",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1554507,
            "range": "± 6291",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 813711,
            "range": "± 2011",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1688548,
            "range": "± 10445",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 241,
            "range": "± 23",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 615,
            "range": "± 148",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 20,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 16,
            "range": "± 0",
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
          "id": "836799142207f300aa2a61e1bb527d22812014db",
          "message": "perf(stark): Optimize Table::columns with pre-allocation (#1090)\n\n- Pre-allocate all column vectors upfront with known capacity\n- Use single pass through data instead of nested iteration\n- Reduces allocation overhead for large traces\n\nCo-authored-by: Nicole Graus <nicole.graus@lambdaclass.com>",
          "timestamp": "2026-01-28T19:35:43Z",
          "tree_id": "55f3aa75e67d444803b9904fa8b86ff9432334a9",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/836799142207f300aa2a61e1bb527d22812014db"
        },
        "date": 1769630979747,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 329099056,
            "range": "± 220112",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 371172905,
            "range": "± 4581461",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 281896994,
            "range": "± 438164",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 687758932,
            "range": "± 4852852",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 780605461,
            "range": "± 3861499",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1440404604,
            "range": "± 1668285",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1627531892,
            "range": "± 10907369",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1240045631,
            "range": "± 1276182",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 3010542876,
            "range": "± 1395118",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3407921355,
            "range": "± 30803436",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6293653178,
            "range": "± 18193570",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7198453878,
            "range": "± 23098027",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5436233629,
            "range": "± 16462467",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7648132,
            "range": "± 1989",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7703904,
            "range": "± 52410",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 9962540,
            "range": "± 231428",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 9687395,
            "range": "± 141848",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 17526040,
            "range": "± 150492",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 18016421,
            "range": "± 286870",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 29764319,
            "range": "± 1254126",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 30624800,
            "range": "± 921848",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 36116482,
            "range": "± 1853056",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 36068064,
            "range": "± 83871",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 65585210,
            "range": "± 959592",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 68179118,
            "range": "± 1657652",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 72557662,
            "range": "± 267540",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 72646491,
            "range": "± 164829",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 141110503,
            "range": "± 735394",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 140387224,
            "range": "± 637056",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 144408595,
            "range": "± 1361467",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 144535845,
            "range": "± 357241",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 293485422,
            "range": "± 3095052",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 285032670,
            "range": "± 4372208",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 10456481,
            "range": "± 574220",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 30695419,
            "range": "± 2153131",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 71664865,
            "range": "± 1633576",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 143211216,
            "range": "± 2018933",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 360609084,
            "range": "± 12364172",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 355848940,
            "range": "± 851944",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 765178903,
            "range": "± 6544910",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1612227008,
            "range": "± 2813248",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3377086532,
            "range": "± 15291875",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7123360260,
            "range": "± 42689073",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 399537093,
            "range": "± 2605756",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 813662182,
            "range": "± 11776327",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1718076267,
            "range": "± 16262505",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3524784092,
            "range": "± 8357266",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7396039012,
            "range": "± 14071001",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 20,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 94,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 61,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 28,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 89,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 164,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 142257,
            "range": "± 242",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1576102,
            "range": "± 4763",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 798546,
            "range": "± 1713",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1690544,
            "range": "± 33656",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 268,
            "range": "± 11",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 638,
            "range": "± 141",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 21,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 19,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 6,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 73,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 11365,
            "range": "± 300",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 201,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
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
          "id": "d14cdb4e850a4922c2c0c3bf2efcc52a8bdfd006",
          "message": "feat(math): Add IsFFTField and optimized extensions for Goldilocks (#1101)\n\n* feat(math): Add IsFFTField and optimized extensions for Goldilocks\n\nRewrites the Goldilocks field implementation with:\n\n- Optimized base field arithmetic using EPSILON = 2^32 - 1\n- Addition chain inversion for efficient a^(p-2) computation\n- IsFFTField implementation with TWO_ADICITY = 32\n\nAdds optimized quadratic extension (Fp2):\n- Uses w^2 = 7 as quadratic non-residue\n- Karatsuba-style multiplication\n- mul_by_7 helper using 7 = 1 + 2 + 4 optimization\n- Optimized squaring formula\n\nAdds optimized cubic extension (Fp3):\n- Uses w^3 = 2 as cubic non-residue\n- Karatsuba-style multiplication with doubling\n- Optimized squaring and inversion\n\nAll implementations use direct u64 representation (no Montgomery form)\nfor maximum performance on the Goldilocks prime structure.\n\n* refactor(math): Simplify Goldilocks field implementation\n\n- Use strip_prefix instead of manual char iteration in from_hex\n- Simplify Display impl by returning write! directly\n- Clarify cubic inv norm calculation with named intermediate\n- Simplify IsSubFieldOf impls with cleaner type annotations\n\n* fix clippy and fmt\n\n* fix comment\n\n---------\n\nCo-authored-by: jotabulacios <jbulacios@fi.uba.ar>",
          "timestamp": "2026-01-28T17:35:06-03:00",
          "tree_id": "4529c4563edfc072d0ef32c3116e28b9a916b7cb",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/d14cdb4e850a4922c2c0c3bf2efcc52a8bdfd006"
        },
        "date": 1769633814337,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 327669094,
            "range": "± 442223",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 369213676,
            "range": "± 542330",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 282510805,
            "range": "± 288358",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 688328458,
            "range": "± 8563346",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 779349975,
            "range": "± 1378093",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1440203616,
            "range": "± 3851090",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1625626901,
            "range": "± 2175913",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1240172826,
            "range": "± 810439",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 3007854091,
            "range": "± 2298880",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3391065203,
            "range": "± 8210951",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6295227853,
            "range": "± 7382460",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7157684036,
            "range": "± 12338529",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5441225996,
            "range": "± 11938328",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7649464,
            "range": "± 4028",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7706702,
            "range": "± 15684",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 9989513,
            "range": "± 124445",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 9973727,
            "range": "± 102329",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 17576069,
            "range": "± 212831",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 17584718,
            "range": "± 158473",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 26099446,
            "range": "± 1097443",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 26886857,
            "range": "± 1440549",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 35601643,
            "range": "± 139779",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 35651008,
            "range": "± 63899",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 65562259,
            "range": "± 406887",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 67181073,
            "range": "± 883390",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 72794030,
            "range": "± 147152",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 72251296,
            "range": "± 45185",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 139637340,
            "range": "± 1785889",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 139821032,
            "range": "± 2858498",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 144366987,
            "range": "± 2257191",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 144310207,
            "range": "± 240292",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 281664403,
            "range": "± 5587249",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 279460105,
            "range": "± 1566726",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 9387163,
            "range": "± 431417",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 30192148,
            "range": "± 442874",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 66892622,
            "range": "± 849907",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 136239432,
            "range": "± 969547",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 356848625,
            "range": "± 1507247",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 359694192,
            "range": "± 1012177",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 764967232,
            "range": "± 1064330",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1608461107,
            "range": "± 2522778",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3353867672,
            "range": "± 9738429",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7078946966,
            "range": "± 12526401",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 386952813,
            "range": "± 2931235",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 810993146,
            "range": "± 2490342",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1701551325,
            "range": "± 7045844",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3523570480,
            "range": "± 5895576",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7386738721,
            "range": "± 9652237",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 4,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 15,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 50,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 24,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 73,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 32,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 144652,
            "range": "± 2318",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1568104,
            "range": "± 7337",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 812423,
            "range": "± 2635",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1725448,
            "range": "± 20311",
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
          "id": "aacd9ff44c0b89eec151a8d23e0728b4a0f51430",
          "message": "Optimize MSM signed digit recoding with flat allocation (#1098)\n\n* Optimize MSM signed digit recoding with flat allocation\n\nReplace Vec<Vec<i64>> with flat Vec<i64> in recode_scalars_signed() to\navoid per-scalar heap allocations.\n\nChanges:\n- Changed recode_scalars_signed() to return Vec<i64> instead of Vec<Vec<i64>>\n- Added total_windows parameter for proper indexing\n- Added inline get_digit() helper for accessing flat storage\n- Updated msm_with_signed() and parallel_msm_with_signed() to use new format\n\nBenefits:\n- Single heap allocation instead of N allocations (N = number of scalars)\n- Better cache locality with contiguous memory layout\n- Reduced memory overhead (no Vec metadata per scalar)\n- ~10-30% speedup for the recoding phase\n\n* Fix out-of-bounds panic in msm_with_signed when points.len() >cs.len()\n\n* fix no-std\n\n---------\n\nCo-authored-by: jotabulacios <jbulacios@fi.uba.ar>",
          "timestamp": "2026-01-28T17:35:50-03:00",
          "tree_id": "f16e358c8d23bb5ca62213412e1a34c6bec6eb37",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/aacd9ff44c0b89eec151a8d23e0728b4a0f51430"
        },
        "date": 1769634015301,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 326299041,
            "range": "± 701048",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 365058493,
            "range": "± 808767",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 281536845,
            "range": "± 2250448",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 687215897,
            "range": "± 708433",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 774264029,
            "range": "± 998694",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1440504210,
            "range": "± 1670852",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1617164618,
            "range": "± 1622493",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1241259837,
            "range": "± 535196",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 3011821458,
            "range": "± 779230",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3385799407,
            "range": "± 12336302",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6297303627,
            "range": "± 2386894",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7152212106,
            "range": "± 6059358",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5447191551,
            "range": "± 7414558",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7679599,
            "range": "± 24585",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7721143,
            "range": "± 26339",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 9897914,
            "range": "± 204377",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 9925330,
            "range": "± 24040",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 17540927,
            "range": "± 44873",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 17495943,
            "range": "± 49548",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 24949529,
            "range": "± 279194",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 25304947,
            "range": "± 380053",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 35598320,
            "range": "± 26173",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 35727698,
            "range": "± 18545",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 66628095,
            "range": "± 214293",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 66833474,
            "range": "± 260867",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 72410618,
            "range": "± 66678",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 72137234,
            "range": "± 84560",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 141159298,
            "range": "± 243010",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 140624007,
            "range": "± 324240",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 144603174,
            "range": "± 67901",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 144566164,
            "range": "± 173013",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 285881920,
            "range": "± 555820",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 285381616,
            "range": "± 470524",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 9030621,
            "range": "± 124161",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 30798175,
            "range": "± 368204",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 70414529,
            "range": "± 228515",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 142388669,
            "range": "± 757543",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 373308088,
            "range": "± 532422",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 357443276,
            "range": "± 2969978",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 763478023,
            "range": "± 1398684",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1616757474,
            "range": "± 1079098",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3365115856,
            "range": "± 2291174",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7090372813,
            "range": "± 2879913",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 383217837,
            "range": "± 695844",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 807878811,
            "range": "± 2166452",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1699263062,
            "range": "± 1445520",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3520796986,
            "range": "± 2117395",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7405172392,
            "range": "± 5657599",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 11,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 27,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 55,
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
            "value": 81,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 50,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 142008,
            "range": "± 2681",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1568354,
            "range": "± 9821",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 799708,
            "range": "± 3213",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1807291,
            "range": "± 11345",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 258,
            "range": "± 29",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 259,
            "range": "± 18",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 9,
            "range": "± 63",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 12,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 15,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 45,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 8043,
            "range": "± 319",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 30,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
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
          "id": "84f2ae0e347b45a0a3870f8e6a2adad86334520c",
          "message": "fix(crypto): Correct hash_to_field conversion issues (#1091)\n\n* fix(crypto): Correct hash_to_field conversion issues\n\n- Fix os2ip to zero-pad hex values (02x format instead of x)\n- Fix os2ip to flush partial chunks that don't fill a complete block\n- Fix build_two_to_the_nth to correctly compute 2^(N*64)\n- Fix compute_length to use actual modulus bit length, not limb count\n- Add bounds check for pseudo_random_bytes to prevent slice panics\n\n* fix os2ip enddianness and oveflow fix\n\n* change panic for error result\n\n---------\n\nCo-authored-by: jotabulacios <jbulacios@fi.uba.ar>\nCo-authored-by: Nicole <nicole.graus@lambdaclass.com>",
          "timestamp": "2026-01-28T20:55:04Z",
          "tree_id": "6401b1cc351de4158c518ff3a34e45ae4f8f976f",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/84f2ae0e347b45a0a3870f8e6a2adad86334520c"
        },
        "date": 1769636042600,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 343439706,
            "range": "± 2552308",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 688614090,
            "range": "± 13263360",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 296849206,
            "range": "± 1748959",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 700580325,
            "range": "± 7389469",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 1833934869,
            "range": "± 121257874",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1497491027,
            "range": "± 6338615",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 4261447852,
            "range": "± 178950059",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1318682059,
            "range": "± 8054003",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 3065343243,
            "range": "± 17320720",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 9064475723,
            "range": "± 431530281",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6378896887,
            "range": "± 63481842",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 19294020034,
            "range": "± 657858444",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5691349244,
            "range": "± 19805119",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 8318581,
            "range": "± 73810",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 8318559,
            "range": "± 67615",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 22046188,
            "range": "± 528383",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 21904703,
            "range": "± 807523",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 23847041,
            "range": "± 527304",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 23367851,
            "range": "± 211199",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 48129580,
            "range": "± 3276310",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 47343519,
            "range": "± 2408127",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 44737109,
            "range": "± 1413415",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 45026663,
            "range": "± 1483301",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 89203946,
            "range": "± 979955",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 89031766,
            "range": "± 1340500",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 89598076,
            "range": "± 593359",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 91256848,
            "range": "± 1251865",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 178928811,
            "range": "± 1828984",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 176213890,
            "range": "± 1148661",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 178509410,
            "range": "± 1840181",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 174562726,
            "range": "± 1224258",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 361624985,
            "range": "± 8443509",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 340089408,
            "range": "± 3932880",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 20354706,
            "range": "± 472959",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 46276800,
            "range": "± 1075996",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 90212217,
            "range": "± 1315636",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 174618159,
            "range": "± 4723539",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 340001432,
            "range": "± 6000579",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 383552682,
            "range": "± 1900221",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 821842818,
            "range": "± 7207738",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1712265286,
            "range": "± 5069169",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3586660197,
            "range": "± 21070706",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7413385886,
            "range": "± 25119792",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 425917391,
            "range": "± 5152490",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 867082548,
            "range": "± 8795256",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1810286504,
            "range": "± 9029375",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3740623340,
            "range": "± 14022032",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7676309753,
            "range": "± 10962937",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 58,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 475,
            "range": "± 9",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 113,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 70,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 196,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 484,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 149711,
            "range": "± 492",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1515901,
            "range": "± 27258",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 837735,
            "range": "± 22250",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1809727,
            "range": "± 23852",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 526,
            "range": "± 95",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 2089,
            "range": "± 85",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 59,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 10,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 12,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 36,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 50,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 23,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 7379,
            "range": "± 95",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 0,
            "range": "± 0",
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
          "id": "800dd293199a564dae09b6fd1ca66a4836787f33",
          "message": "fix(bls12-381): Fix critical bugs in G1/G2 decompression (#1093)\n\n* fix(bls12-381): Fix critical bugs in G1/G2 decompression\n\n- Fix operator precedence bug in length checks: `!x == 48` was always\n  false since `!` binds tighter than `==`. Changed to `x != 48/96`.\n- Add subgroup membership check for G2 decompression (matching G1).\n- Replace `.unwrap()` with `?` for proper error propagation.\n- Update tests to use valid subgroup points (generator multiples).\n\n* validate canonical encoding for infinity points\n\n---------\n\nCo-authored-by: jotabulacios <jbulacios@fi.uba.ar>\nCo-authored-by: Nicole Graus <nicole.graus@lambdaclass.com>",
          "timestamp": "2026-01-28T21:09:52Z",
          "tree_id": "6ac53206ccf636b750bada204090d5401c2050f5",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/800dd293199a564dae09b6fd1ca66a4836787f33"
        },
        "date": 1769636621917,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 327275221,
            "range": "± 1098997",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 367930646,
            "range": "± 2990986",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 282864130,
            "range": "± 3295917",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 687231848,
            "range": "± 668122",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 770724083,
            "range": "± 1389466",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1438883676,
            "range": "± 4535356",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1622818450,
            "range": "± 2859628",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1243216417,
            "range": "± 1525488",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 3008458382,
            "range": "± 1248104",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3392924682,
            "range": "± 12981653",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6301114480,
            "range": "± 17092015",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7161888641,
            "range": "± 25272134",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5447847876,
            "range": "± 8518812",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7686061,
            "range": "± 131932",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7728384,
            "range": "± 4348",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 10963153,
            "range": "± 565006",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 10635984,
            "range": "± 441646",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 17945013,
            "range": "± 71634",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 17967227,
            "range": "± 140827",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 30958985,
            "range": "± 767062",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 30762282,
            "range": "± 266728",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 35902220,
            "range": "± 58603",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 36137268,
            "range": "± 119588",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 69370848,
            "range": "± 750526",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 69240801,
            "range": "± 402933",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 72631265,
            "range": "± 412591",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 72483339,
            "range": "± 170850",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 140521242,
            "range": "± 5696536",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 142459713,
            "range": "± 1120568",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 144909867,
            "range": "± 842257",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 144865820,
            "range": "± 2601653",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 282782182,
            "range": "± 2764719",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 278281359,
            "range": "± 1394198",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 14449625,
            "range": "± 410229",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 33312536,
            "range": "± 771908",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 66201926,
            "range": "± 935597",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 137265375,
            "range": "± 1055120",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 358780113,
            "range": "± 4024177",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 366253151,
            "range": "± 908165",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 770843688,
            "range": "± 848537",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1616451297,
            "range": "± 2367453",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3364888504,
            "range": "± 13518512",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7065089232,
            "range": "± 13437073",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 393187520,
            "range": "± 639952",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 813786275,
            "range": "± 1067467",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1699098925,
            "range": "± 1607159",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3521408777,
            "range": "± 7751911",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7386633913,
            "range": "± 10946167",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 552,
            "range": "± 5",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 35375,
            "range": "± 133",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 428,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 227,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 724,
            "range": "± 7",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 24870,
            "range": "± 338",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 144710,
            "range": "± 1303",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1576632,
            "range": "± 13763",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 797213,
            "range": "± 3235",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1657894,
            "range": "± 64394",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 1732,
            "range": "± 1332",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 66172,
            "range": "± 1130",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 569,
            "range": "± 7",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 12,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 15,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 44,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 7400,
            "range": "± 634",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 30,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
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
          "id": "fe4c91d0ed6d62a553ac281a235ef50219697b5e",
          "message": "fix(bn254): Fix error handling and add G2 subgroup check (#1094)\n\n- Replace `.unwrap()` with `?` for proper error propagation in G2\n  decompression when parsing field elements.\n- Add subgroup membership check for G2 decompression (matching G1\n  security expectations and BLS12-381 behavior).\n\nCo-authored-by: Nicole Graus <nicole.graus@lambdaclass.com>",
          "timestamp": "2026-01-28T21:10:17Z",
          "tree_id": "ed3eeb70fb4fe0add70a4c225b9f521d30b3c927",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/fe4c91d0ed6d62a553ac281a235ef50219697b5e"
        },
        "date": 1769636626370,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 328317161,
            "range": "± 891298",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 367907980,
            "range": "± 2185375",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 282361458,
            "range": "± 4449499",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 687973198,
            "range": "± 696778",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 780111242,
            "range": "± 17171000",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1440775176,
            "range": "± 567218",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1627369617,
            "range": "± 3158403",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1242039909,
            "range": "± 9957019",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 3012835528,
            "range": "± 20280674",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3439880598,
            "range": "± 22311345",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6291345254,
            "range": "± 13764392",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7192952956,
            "range": "± 25747644",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5440735580,
            "range": "± 11315807",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7666555,
            "range": "± 6141",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7709921,
            "range": "± 5261",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 10228043,
            "range": "± 253602",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 10222604,
            "range": "± 116432",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 17709905,
            "range": "± 34421",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 17890533,
            "range": "± 112634",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 27379480,
            "range": "± 396021",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 27929740,
            "range": "± 686049",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 35836907,
            "range": "± 69997",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 35943235,
            "range": "± 65772",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 67633908,
            "range": "± 1377511",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 67472309,
            "range": "± 576605",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 72753053,
            "range": "± 1930615",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 72441560,
            "range": "± 56366",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 140437336,
            "range": "± 865767",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 140489594,
            "range": "± 771211",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 145469226,
            "range": "± 2014337",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 144973820,
            "range": "± 223661",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 282502961,
            "range": "± 660642",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 282889593,
            "range": "± 756217",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 10759579,
            "range": "± 526140",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 33203998,
            "range": "± 1568828",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 73636304,
            "range": "± 2617339",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 145696932,
            "range": "± 2435094",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 370960787,
            "range": "± 10898395",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 364597305,
            "range": "± 14854821",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 773352602,
            "range": "± 5210238",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1613508212,
            "range": "± 2878396",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3374212890,
            "range": "± 12537331",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7094059747,
            "range": "± 8732945",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 394577180,
            "range": "± 2059235",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 819654030,
            "range": "± 2935960",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1704950841,
            "range": "± 1542676",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3535033101,
            "range": "± 25544265",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7410898633,
            "range": "± 9826209",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 4,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 14,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 50,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 24,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 74,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 32,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 141898,
            "range": "± 1094",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1568978,
            "range": "± 31420",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 809952,
            "range": "± 3643",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1824225,
            "range": "± 17797",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 220,
            "range": "± 17",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 47,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 5,
            "range": "± 54",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 11,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 14,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 43,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 57,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 26,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 9684,
            "range": "± 519",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
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
          "id": "9a2446d60ad213621120600a5b5e193408031d11",
          "message": "fix(crypto): Address merkle tree and related issues (#1092)\n\n* fix(crypto): Address merkle tree and related issues\n\nMerkle tree improvements:\n- Add domain separation to FieldElementVectorBackend hash functions\n  (prefix leaves with 'leaf' + count, nodes with 'node')\n- Include element length in hash to prevent collision with variable-width bytes\n- Document padding behavior in complete_until_power_of_two\n- Fix proof deserialization to auto-detect node size instead of hardcoding 8 bytes\n\nAdditional fixes:\n- Add Serializable trait to lambdaworks_math::traits (was missing)\n- Add AsBytes impl for U64PrimeField elements\n- Fix test_transcript.rs broken import (removed non-existent Transcript trait)\n\n* improve proof deserialization and fix\n  clippy\n\n---------\n\nCo-authored-by: jotabulacios <jbulacios@fi.uba.ar>\nCo-authored-by: Nicole Graus <nicole.graus@lambdaclass.com>",
          "timestamp": "2026-01-28T21:13:29Z",
          "tree_id": "e0aae0cc64362258f9750a4d6575ccd49f27f028",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/9a2446d60ad213621120600a5b5e193408031d11"
        },
        "date": 1769636860523,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 328255429,
            "range": "± 540074",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 370160371,
            "range": "± 2015446",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 282258486,
            "range": "± 220579",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 688925812,
            "range": "± 1734376",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 778046051,
            "range": "± 2450656",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1443683907,
            "range": "± 12568329",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1634405382,
            "range": "± 4590802",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1241939874,
            "range": "± 991777",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 3009829421,
            "range": "± 18152461",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3415660364,
            "range": "± 13253865",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6292789925,
            "range": "± 18710657",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7229267801,
            "range": "± 31330693",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5444589317,
            "range": "± 13670409",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7682760,
            "range": "± 23928",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7726406,
            "range": "± 3726",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 11193017,
            "range": "± 584062",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 10746453,
            "range": "± 396203",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 17848392,
            "range": "± 42394",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 17849608,
            "range": "± 30280",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 30054182,
            "range": "± 773379",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 29090213,
            "range": "± 1001624",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 36072202,
            "range": "± 161543",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 36083510,
            "range": "± 78184",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 68633630,
            "range": "± 834650",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 69293896,
            "range": "± 1437998",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 72963255,
            "range": "± 786143",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 72651455,
            "range": "± 144961",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 143065893,
            "range": "± 1104628",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 143891468,
            "range": "± 566504",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 145140857,
            "range": "± 242238",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 145240637,
            "range": "± 1374733",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 285580704,
            "range": "± 1711712",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 284786637,
            "range": "± 1589749",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 14286589,
            "range": "± 1052959",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 33687311,
            "range": "± 509399",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 69726751,
            "range": "± 1327524",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 142997272,
            "range": "± 2392696",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 368119237,
            "range": "± 5274301",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 365083540,
            "range": "± 1479974",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 775788894,
            "range": "± 2572370",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1617898356,
            "range": "± 2063172",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3370226565,
            "range": "± 13053168",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7084086159,
            "range": "± 11854210",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 388743886,
            "range": "± 860872",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 815430144,
            "range": "± 2860390",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1705300995,
            "range": "± 8311948",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3534264462,
            "range": "± 5765167",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7390222157,
            "range": "± 26669016",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 109,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 1776,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 191,
            "range": "± 9",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 120,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 310,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 1693,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 142520,
            "range": "± 433",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1575240,
            "range": "± 15077",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 801562,
            "range": "± 4438",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1734768,
            "range": "± 14607",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 835,
            "range": "± 461",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 7154,
            "range": "± 126",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 117,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 11,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 13,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 42,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 58,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 26,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
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
          "id": "69dc2ea9fb79ea7a0e741cc567a7fe4551bdccf2",
          "message": "feat: Add AI code reviewers for PRs (#1103)\n\n* feat: Add AI code reviewers for PRs\n\nAdd automated code review workflows using Claude, ChatGPT, and Kimi AI.\nReviewers focus on cryptographic correctness, security (timing side-channels,\nconstant-time operations), performance, and code quality specific to\nlambdaworks' finite field and ZK proof implementations.\n\n* fix: Skip AI review workflows for fork PRs\n\nAdd condition to only run reviews when PR originates from the same\nrepository, preventing failed checks on external contributor PRs.\n\n* fix: Address greptile security review comments\n\n- Claude: Use stable @v1 tag instead of @beta\n- ChatGPT: Pin to specific commit SHA, pass params via env vars\n- Kimi: Fix command/script injection by using jq for JSON construction\n- Kimi: Read review from file instead of output interpolation\n- Kimi: Add error feedback comment when API fails\n\n* fix: Remove unused env var and fix error message interpolation\n\n- Remove ineffective REVIEW_CONTENT env var\n- Pass error message via env var instead of direct interpolation",
          "timestamp": "2026-01-29T15:08:59Z",
          "tree_id": "bc1c78cac7737c914c07837c285a440dc3890c9c",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/69dc2ea9fb79ea7a0e741cc567a7fe4551bdccf2"
        },
        "date": 1769701377247,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 328853999,
            "range": "± 402714",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 372149054,
            "range": "± 1203713",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 283248289,
            "range": "± 1093324",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 689674023,
            "range": "± 11280872",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 781085325,
            "range": "± 1718135",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1441406667,
            "range": "± 8930684",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1639500118,
            "range": "± 6381856",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1245030174,
            "range": "± 11188426",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 3008969683,
            "range": "± 13653622",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3415474535,
            "range": "± 25858724",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6289321359,
            "range": "± 20444668",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7238710974,
            "range": "± 17628338",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5449934021,
            "range": "± 19046768",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7661099,
            "range": "± 100322",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7709189,
            "range": "± 5823",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 13326756,
            "range": "± 570564",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 13047017,
            "range": "± 616269",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 18146852,
            "range": "± 31377",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 18095164,
            "range": "± 144509",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 34838701,
            "range": "± 750711",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 34329559,
            "range": "± 377554",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 36111868,
            "range": "± 94604",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 36349476,
            "range": "± 94387",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 71375558,
            "range": "± 352802",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 72137531,
            "range": "± 958981",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 73263753,
            "range": "± 89271",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 72955281,
            "range": "± 453857",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 145056879,
            "range": "± 1959006",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 144069377,
            "range": "± 454280",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 145283153,
            "range": "± 330082",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 145144653,
            "range": "± 2019969",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 292108190,
            "range": "± 1964960",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 290766299,
            "range": "± 2219855",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 15098660,
            "range": "± 801027",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 34620962,
            "range": "± 589322",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 70872723,
            "range": "± 1218299",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 142899720,
            "range": "± 1862655",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 375229721,
            "range": "± 14158642",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 365194267,
            "range": "± 1973487",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 773443379,
            "range": "± 3956497",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1618200441,
            "range": "± 5416914",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3368357858,
            "range": "± 22103368",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7082799395,
            "range": "± 7345149",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 393622178,
            "range": "± 7779236",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 822931723,
            "range": "± 3942093",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1706721808,
            "range": "± 8397545",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3525536879,
            "range": "± 23226210",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7402620305,
            "range": "± 20581498",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 45,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 383,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 116,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 73,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 189,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 509,
            "range": "± 11",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 142603,
            "range": "± 340",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1577832,
            "range": "± 15719",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 801285,
            "range": "± 3554",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1674638,
            "range": "± 15588",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 582,
            "range": "± 181",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 2324,
            "range": "± 27",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 45,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 17,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 18,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 56,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 10022,
            "range": "± 261",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 42,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
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
          "id": "93ecf9b1a7d75288f02951e82c48fb1f05bb68e2",
          "message": "Optimize polynomial operations to reduce cloning and allocations (#1102)\n\n* Optimize polynomial operations to reduce cloning and allocations\n\n- Add in-place operations: add_assign, sub_assign, scale_coeffs_mut, neg_mut\n- Implement AddAssign, SubAssign, MulAssign traits for ergonomic usage\n- Optimize Add trait to avoid pad_with_zero_coefficients which cloned both polynomials\n- Optimize Sub trait to avoid creating intermediate negated polynomial\n- Optimize Mul<&FieldElement> to avoid cloning self and multiplicand\n- Refactor xgcd to use in-place operations, reducing allocations in the loop\n- Add comprehensive tests for all new in-place operations\n\n* resolve infinite recursion\n\n* fix clippy\n\n* addres PR comments\n\n---------\n\nCo-authored-by: jotabulacios <jbulacios@fi.uba.ar>",
          "timestamp": "2026-01-29T13:46:40-03:00",
          "tree_id": "21dd15c732bf105208692626b97ef294cb9854b8",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/93ecf9b1a7d75288f02951e82c48fb1f05bb68e2"
        },
        "date": 1769706949677,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 334993676,
            "range": "± 204485",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 657998556,
            "range": "± 4246339",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 294363038,
            "range": "± 2032234",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 702169353,
            "range": "± 679149",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 1582920634,
            "range": "± 12506680",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1470118022,
            "range": "± 1363182",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 3790599535,
            "range": "± 30823696",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1294762412,
            "range": "± 1088819",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 3068443525,
            "range": "± 1269533",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 8316813885,
            "range": "± 47450081",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6399351806,
            "range": "± 11067263",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 18542448112,
            "range": "± 123328802",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5647055950,
            "range": "± 1190440",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 8078082,
            "range": "± 13060",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 8115071,
            "range": "± 8416",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 16059461,
            "range": "± 1424714",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 14217669,
            "range": "± 1006188",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 21525604,
            "range": "± 101700",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 21443591,
            "range": "± 102681",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 41732048,
            "range": "± 702057",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 40870484,
            "range": "± 598685",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 43270068,
            "range": "± 218838",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 43460741,
            "range": "± 161470",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 86357981,
            "range": "± 779818",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 85202472,
            "range": "± 260146",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 87360828,
            "range": "± 478596",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 87505544,
            "range": "± 544844",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 170570899,
            "range": "± 502817",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 173445455,
            "range": "± 1237637",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 176552205,
            "range": "± 806006",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 179053532,
            "range": "± 1474451",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 345711538,
            "range": "± 3112030",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 342130683,
            "range": "± 1373891",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 17605958,
            "range": "± 428396",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 42368636,
            "range": "± 239341",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 85291894,
            "range": "± 1209949",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 166890369,
            "range": "± 496500",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 334722389,
            "range": "± 3705031",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 392483590,
            "range": "± 1035324",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 826896148,
            "range": "± 2354528",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1726280519,
            "range": "± 3004349",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3580847093,
            "range": "± 3266851",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7421814615,
            "range": "± 8561095",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 423876080,
            "range": "± 1497109",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 880923107,
            "range": "± 3509297",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1818083901,
            "range": "± 2685932",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3774533482,
            "range": "± 4716404",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7809068216,
            "range": "± 88068240",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 359,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 11529,
            "range": "± 9",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 256,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 165,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 250,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 6189,
            "range": "± 45",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 149612,
            "range": "± 159",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1493667,
            "range": "± 10888",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 831787,
            "range": "± 3981",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1136407,
            "range": "± 37084",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 821,
            "range": "± 478",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 13696,
            "range": "± 261",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 349,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 20,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 6,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 77,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 9795,
            "range": "± 1351",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 215,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 6871,
            "range": "± 276",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 0,
            "range": "± 0",
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
          "id": "1a52040bed529db78cba4c22374ace55fef7f085",
          "message": "Stark prover profiling (#1100)\n\n* Add profiling instrumentation to STARK prover\n\n- Add dhat-heap feature for memory profiling with dhat\n- Create prover_profile benchmark binary for profiling\n- Support samply and flamegraph for CPU profiling\n- Add PROFILING.md with detailed usage instructions\n\nThe profiling binary supports:\n- Configurable trace length (--trace-length N for 2^N rows)\n- Multiple iterations (--iterations N)\n- Parallel mode (--parallel with parallel feature)\n- dhat heap profiling (--features dhat-heap)\n\nExample memory profile with 2^14 rows:\n- Total allocations: 200 MB in 952K blocks\n- Peak memory: 33 MB\n\n* Optimize constraint evaluator to reduce allocations in hot loop\n\nBased on profiling with dhat, identified that the constraint evaluation\nloop was allocating a new Vec for transition evaluations on every\niteration (65K+ times for a 16K trace with 4x blowup).\n\nChanges:\n- Add compute_transition_into() to AIR trait for buffer reuse\n- Optimize non-parallel path to reuse buffers for both periodic values\n  and transition evaluations\n- Add OPTIMIZATION_FINDINGS.md documenting profiling results\n\nBenchmark results (2^14 trace):\n- Before: 952,106 allocation blocks\n- After: 886,571 allocation blocks\n- Improvement: 65,535 fewer blocks (6.9% reduction)\n\nThe number of eliminated allocations matches exactly the number of\nloop iterations, confirming the optimization is effective.\n\n* Optimize peak memory in commit_composition_polynomial with pre-allocation\n\nPre-allocate vectors with known capacity in commit_composition_polynomial\nto avoid incremental growth via push(). This was the largest peak memory\ncontributor (32 MB via grow_one pattern).\n\nResults:\n- 2^14 trace: Peak 33→27 MB (18.9% reduction)\n- 2^16 trace: Peak 133→108 MB (18.9% reduction)\n\n* Add fused bit-reverse transpose to eliminate LDE trace clone\n\n- Add columns2rows_bit_reversed() that combines bit-reverse permutation\n  with column-to-row transpose in a single pass\n- Update interpolate_and_commit_main/aux to use fused function\n- Eliminates clone of lde_trace_evaluations (~8 MB per trace)\n- Fix conditional import of Polynomial to match usage context\n\nResults:\n- Total allocations: 757 → 749 MB (1% reduction)\n- Code is cleaner with single-pass operation\n\n* Optimize zerofier computation with split base/exemptions caching\n\nMost transition constraints share the same base zerofier 1/(x^n - 1) with\nperiod=1, offset=0. Previously, the full zerofier (including end exemptions\npolynomial) was cached per unique combination of all parameters.\n\nNow we cache separately:\n1. Base zerofier by (period, offset, exemptions_period, periodic_exemptions_offset)\n2. End exemptions polynomial evaluations by (end_exemptions, period)\n\nThis benefits AIRs with multiple constraints sharing the same base zerofier\nbut different end_exemptions values (common pattern). The expensive batch\ninverse for the base zerofier is computed only once.\n\nExample: fibonacci_2_cols_shifted has 2 constraints both with period=1,\noffset=0, end_exemptions=1 - they now share the same zerofier computation.\n\n* Parallelize zerofier computation for multi-constraint AIRs\n\nRefactored transition_zerofier_evaluations to support parallel computation:\n\n1. Extract helper functions (compute_base_zerofier, compute_end_exemptions_evals,\n   compute_end_exemptions_poly) to module level for object-safety\n\n2. Pre-collect unique keys before computation to enable parallelism\n\n3. Use par_iter when parallel feature is enabled to compute:\n   - Base zerofiers in parallel\n   - End exemptions evaluations in parallel\n\n4. Combine results sequentially (cheap operation)\n\nPerformance with parallel feature (2^16 trace):\n- Boundary evaluation: 27ms → 8.9ms (67% faster)\n- Transitions evaluation: 81ms → 31ms (65% faster)\n\nNote: For simple AIRs with 1 constraint (like Fibonacci), the parallel\noverhead exceeds benefits. Optimization primarily benefits complex AIRs\nwith many constraints.\n\n* Update OPTIMIZATION_FINDINGS.md with parallelization results\n\n* Optimize h_terms accumulation with in-place coefficient addition\n\nReplace intermediate polynomial allocations in compute_deep_composition_poly\nwith direct coefficient accumulation. Instead of creating new polynomials\non each iteration with h_terms = h_terms + h_i_term, we now:\n\n1. Pre-allocate coefficient buffer with max degree\n2. Accumulate gamma * (part - h_i_eval) directly into coefficients\n3. Create final polynomial once from accumulated coefficients\n\nResults (2^16 trace):\n- Total allocations: 748 MB → 729 MB (2.5% reduction, ~19 MB savings)\n- Peak memory: unchanged (108 MB)\n\n* Update OPTIMIZATION_FINDINGS.md with in-place polynomial results\n\n* Optimize boundary zerofier caching and z_shifted pre-computation\n\n- Cache boundary zerofiers by step: multiple constraints at the same\n  step now share the same zerofier computation\n- Pre-build Vec of references to avoid HashMap lookup in hot loop\n- Pre-compute z_shifted values in compute_deep_composition_poly to\n  avoid redundant primitive_root.pow() calculations per trace column\n\nThese optimizations benefit complex AIRs with multiple boundary\nconstraints at the same step and reduce redundant computation in\nthe deep composition polynomial.\n\n* Update OPTIMIZATION_FINDINGS.md with completed optimizations\n\nMark boundary zerofier caching and z_shifted pre-computation as done.\nUpdate phase sections to reflect current progress.\n\n* Add support for multiple AIRs in prover profiling benchmark\n\nAdd --air option to select between:\n- fibonacci: Simple AIR with 1 transition, 2 boundary constraints\n- read_only_memory (rom): Complex AIR with 3 transition, 6 boundary constraints\n\nThe ReadOnlyMemory AIR better demonstrates the boundary zerofier caching\noptimization since it has 5 constraints at step 0 that share a zerofier.\n\n* Address PR review comments for stark prover profiling\n\n- Make dhat dependency optional (only compiled with dhat-heap feature)\n- Add empty input safety check in columns2rows_bit_reversed\n- Remove unnecessary Vec allocation by using iter().cloned() instead of clone().into_iter()\n- Add documentation for zerofier_cache optimization explaining the deduplication strategy\n- Fix bug where h_terms was used before definition (incomplete optimization)\n\n* fix: Restore .take(cs.len()) bounds check in signed MSM functions\n\nThe previous changes accidentally removed .take(cs.len()) from the point\niteration loops in msm_with_signed and parallel_msm_with_signed. This could\ncause out-of-bounds access to signed_digits when points.len() > cs.len(),\nsince signed_digits is sized for cs.len() scalars.\n\n* fix: Remove unnecessary clone() on Copy types in prover_profile\n\n* fix: Move num_periodic_cols into non-parallel block where it's used",
          "timestamp": "2026-01-29T16:19:18-03:00",
          "tree_id": "1a253eff8453e985bb679e2f319da7186494733b",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/1a52040bed529db78cba4c22374ace55fef7f085"
        },
        "date": 1769715858950,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 328088435,
            "range": "± 666660",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 364504556,
            "range": "± 810661",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 277423547,
            "range": "± 634367",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 691230993,
            "range": "± 468773",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 776374437,
            "range": "± 2749369",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1447237529,
            "range": "± 915813",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1624882622,
            "range": "± 5749585",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1223714841,
            "range": "± 296191",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 3029725172,
            "range": "± 10148366",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3404820654,
            "range": "± 7384724",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6333035733,
            "range": "± 9111944",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7204975363,
            "range": "± 62435286",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5366526235,
            "range": "± 8726927",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7645765,
            "range": "± 54187",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7671395,
            "range": "± 3799",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 9647972,
            "range": "± 134125",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 9638484,
            "range": "± 34493",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 17492411,
            "range": "± 28573",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 17460144,
            "range": "± 39329",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 25395060,
            "range": "± 918762",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 25832344,
            "range": "± 290064",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 35687713,
            "range": "± 153589",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 35578373,
            "range": "± 68492",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 68244576,
            "range": "± 940221",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 66283736,
            "range": "± 1216597",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 72239348,
            "range": "± 159511",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 71907512,
            "range": "± 310100",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 141790059,
            "range": "± 1391913",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 141809280,
            "range": "± 738854",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 144268503,
            "range": "± 176752",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 144399541,
            "range": "± 202298",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 285925278,
            "range": "± 4073161",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 283946355,
            "range": "± 3795311",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 10300032,
            "range": "± 528609",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 30918117,
            "range": "± 321568",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 69552415,
            "range": "± 1081671",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 144843589,
            "range": "± 2845032",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 382074072,
            "range": "± 3136641",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 360044572,
            "range": "± 2084630",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 771217914,
            "range": "± 2520567",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1624219157,
            "range": "± 10070448",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3387095399,
            "range": "± 4275789",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7133492557,
            "range": "± 30544927",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 389366329,
            "range": "± 832852",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 809754223,
            "range": "± 2171026",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1707769122,
            "range": "± 3865012",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3543615577,
            "range": "± 6969332",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7466660788,
            "range": "± 13315319",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 103,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 1684,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 169,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 104,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 155,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 1719,
            "range": "± 77",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 142226,
            "range": "± 392",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1577837,
            "range": "± 88597",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 796824,
            "range": "± 3056",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1082819,
            "range": "± 9326",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 425,
            "range": "± 137",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 4039,
            "range": "± 124",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 110,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 14,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 82,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 47,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 10896,
            "range": "± 728",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 46,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
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
          "id": "6cd19a4b85e3fc34bdf74963f33d722ce24da18d",
          "message": "fix(stark): Validate grinding_factor to prevent overflow (#1086)\n\n* fix(stark): Validate grinding_factor to prevent overflow\n\n- Add assertion that grinding_factor must be <= 64\n- Handle grinding_factor == 0 case explicitly (any nonce is valid)\n- Prevents undefined behavior from 1 << 64 shift\n- Prevents underflow when grinding_factor > 64\n\n* add tests\n\n---------\n\nCo-authored-by: jotabulacios <jbulacios@fi.uba.ar>",
          "timestamp": "2026-01-29T16:22:40-03:00",
          "tree_id": "bf39e4728958e594a9a4d0d739dbe91d85b3d49e",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/6cd19a4b85e3fc34bdf74963f33d722ce24da18d"
        },
        "date": 1769716013481,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 327983450,
            "range": "± 263304",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 362901050,
            "range": "± 471113",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 277619757,
            "range": "± 404758",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 690484795,
            "range": "± 375959",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 769649543,
            "range": "± 1873888",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1449337482,
            "range": "± 330815",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1617365534,
            "range": "± 2082166",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1224608987,
            "range": "± 672020",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 3025022676,
            "range": "± 1707683",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3372806834,
            "range": "± 6102393",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6327180964,
            "range": "± 2368528",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7117146620,
            "range": "± 4145487",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5370048070,
            "range": "± 5118102",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7625187,
            "range": "± 4535",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7667183,
            "range": "± 5379",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 9649432,
            "range": "± 5650",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 9687154,
            "range": "± 7418",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 17280104,
            "range": "± 26280",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 17197188,
            "range": "± 15384",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 23058317,
            "range": "± 112027",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 23059746,
            "range": "± 116131",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 35225747,
            "range": "± 22427",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 35238988,
            "range": "± 32433",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 62663385,
            "range": "± 144741",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 62743937,
            "range": "± 156773",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 71339211,
            "range": "± 67527",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 71181227,
            "range": "± 42898",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 133692075,
            "range": "± 296145",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 134737540,
            "range": "± 449330",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 143815046,
            "range": "± 277912",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 143755197,
            "range": "± 100479",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 271356103,
            "range": "± 454133",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 270951144,
            "range": "± 670173",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 8293572,
            "range": "± 61846",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 27294809,
            "range": "± 96705",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 63301551,
            "range": "± 288272",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 130207327,
            "range": "± 282839",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 340249947,
            "range": "± 564294",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 356118045,
            "range": "± 1294204",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 760682914,
            "range": "± 1048144",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1607895035,
            "range": "± 4372024",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3359024810,
            "range": "± 1340284",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7086185639,
            "range": "± 1902793",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 384255451,
            "range": "± 540868",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 803266708,
            "range": "± 590119",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1690996529,
            "range": "± 765784",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3516298662,
            "range": "± 2146595",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7394118799,
            "range": "± 1996843",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 13,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 65,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 34,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 28,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 35,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 110,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 142777,
            "range": "± 819",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1576931,
            "range": "± 13685",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 796840,
            "range": "± 1212",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1082852,
            "range": "± 10454",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 134,
            "range": "± 46",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 340,
            "range": "± 40",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 14,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 13,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 16,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 46,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 9267,
            "range": "± 341",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 34,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 9588,
            "range": "± 435",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "jbulacios@fi.uba.ar",
            "name": "jotabulacios",
            "username": "jotabulacios"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "41708478aa58f08632ee8f737c9ae97339ca7e73",
          "message": "Fix/transcript sampling bias state (#1096)\n\n* fix(crypto): Guard sample_u64 against zero upper_bound\n\nAdd assertion to prevent division by zero when upper_bound is 0\nin both DefaultTranscript and StoneProverTranscript implementations.\n\n* fix(transcript): advance sampling state and remove modulo bias\n\n---------\n\nCo-authored-by: diegokingston <dkingston@fi.uba.ar>\nCo-authored-by: Diego K <43053772+diegokingston@users.noreply.github.com>",
          "timestamp": "2026-01-29T16:40:05-03:00",
          "tree_id": "fd1ba581f800bde452fa544cb5e719adc78fb98f",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/41708478aa58f08632ee8f737c9ae97339ca7e73"
        },
        "date": 1769717339001,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 334710251,
            "range": "± 457573",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 613609418,
            "range": "± 9408860",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 293227437,
            "range": "± 2954677",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 702164337,
            "range": "± 434049",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 1558030968,
            "range": "± 9860884",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1469289442,
            "range": "± 555548",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 3692222541,
            "range": "± 18725079",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1290273879,
            "range": "± 921633",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 3076206945,
            "range": "± 14327840",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 8487573779,
            "range": "± 169303879",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6414337326,
            "range": "± 5321136",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 18617521359,
            "range": "± 154154127",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5627698872,
            "range": "± 9718073",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 8046673,
            "range": "± 36490",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 8086344,
            "range": "± 12131",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 12915378,
            "range": "± 352529",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 12670579,
            "range": "± 267193",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 21320406,
            "range": "± 196327",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 21278718,
            "range": "± 63148",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 39897906,
            "range": "± 823190",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 39104248,
            "range": "± 845342",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 42909108,
            "range": "± 130266",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 43475208,
            "range": "± 110486",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 83701859,
            "range": "± 514144",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 85728898,
            "range": "± 1612279",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 87016750,
            "range": "± 477244",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 86524231,
            "range": "± 348816",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 167451291,
            "range": "± 868606",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 166662849,
            "range": "± 382932",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 172881413,
            "range": "± 590168",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 172482096,
            "range": "± 664826",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 335174794,
            "range": "± 1022274",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 332013691,
            "range": "± 782169",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 14933171,
            "range": "± 900946",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 39587020,
            "range": "± 657409",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 81476202,
            "range": "± 312347",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 163282208,
            "range": "± 1877528",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 333766701,
            "range": "± 906634",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 387749886,
            "range": "± 1322348",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 819776205,
            "range": "± 1181609",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1719454954,
            "range": "± 25067191",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3566315466,
            "range": "± 7244436",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7390280579,
            "range": "± 27564700",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 419934804,
            "range": "± 2314439",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 877367433,
            "range": "± 3256064",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1819902266,
            "range": "± 6717748",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3756789520,
            "range": "± 3414325",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7765193930,
            "range": "± 21809645",
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
            "value": 107,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 35,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 25,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 35,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 155,
            "range": "± 15",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 149441,
            "range": "± 438",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1523615,
            "range": "± 12899",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 834470,
            "range": "± 1649",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1135627,
            "range": "± 10472",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 188,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 429,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 24,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 10,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 13,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 38,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 7293,
            "range": "± 546",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 27,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 0,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 0,
            "range": "± 0",
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
          "id": "052e117d12a35d0f000abbd8e6dfa04816aee3fa",
          "message": "fix: Implement ByteConversion for P448 Goldilocks U56x8 (#1106)\n\n* fix: Implement ByteConversion for P448 Goldilocks U56x8\n\nImplement the four ByteConversion trait methods that were previously\nstubbed with unimplemented!():\n- to_bytes_be: Convert to 56-byte big-endian representation\n- to_bytes_le: Convert to 56-byte little-endian representation\n- from_bytes_be: Parse from big-endian bytes\n- from_bytes_le: Parse from little-endian bytes\n\nThe U56x8 type uses 8 limbs of 56 bits each (7 bytes per limb),\nstored in little-endian order internally.\n\nAdded tests for roundtrip conversion, known values, and invalid input.\n\n* style: Use iterators instead of index-based loops (clippy fix) (#1110)",
          "timestamp": "2026-01-29T18:31:17-03:00",
          "tree_id": "18ca4bbab06ef0385476688362b804671d659580",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/052e117d12a35d0f000abbd8e6dfa04816aee3fa"
        },
        "date": 1769723757879,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 331278444,
            "range": "± 403955",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 375249826,
            "range": "± 1099943",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 279278465,
            "range": "± 202979",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 694291868,
            "range": "± 596418",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 782224482,
            "range": "± 2218001",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1453790484,
            "range": "± 471459",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1653962671,
            "range": "± 6532680",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1226572142,
            "range": "± 759264",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 3037348585,
            "range": "± 1217216",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3456971126,
            "range": "± 14272636",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6349898985,
            "range": "± 5636386",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7296071044,
            "range": "± 42924117",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5372313161,
            "range": "± 1918956",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7643462,
            "range": "± 6153",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7671687,
            "range": "± 13282",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 10236332,
            "range": "± 64204",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 10309577,
            "range": "± 104583",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 17647481,
            "range": "± 88524",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 17635961,
            "range": "± 65137",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 27132616,
            "range": "± 567089",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 28886419,
            "range": "± 1014562",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 35782076,
            "range": "± 215143",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 35881102,
            "range": "± 79320",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 67666399,
            "range": "± 555940",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 68298740,
            "range": "± 833193",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 72236912,
            "range": "± 602809",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 72636921,
            "range": "± 178093",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 144314778,
            "range": "± 1031321",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 145207574,
            "range": "± 827372",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 145158884,
            "range": "± 464618",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 145238598,
            "range": "± 395897",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 284858417,
            "range": "± 1793531",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 283631501,
            "range": "± 1545508",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 13318675,
            "range": "± 1075570",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 35434657,
            "range": "± 954298",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 71135707,
            "range": "± 954596",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 142369309,
            "range": "± 2741923",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 372991927,
            "range": "± 2709860",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 369373599,
            "range": "± 3859932",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 783973805,
            "range": "± 3291516",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1634246563,
            "range": "± 2536033",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3416837104,
            "range": "± 5360212",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7206092460,
            "range": "± 25020969",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 396610506,
            "range": "± 1157276",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 832592396,
            "range": "± 3437082",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1737872188,
            "range": "± 6517377",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3584570609,
            "range": "± 10119580",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7525436617,
            "range": "± 15611045",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 105,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 1777,
            "range": "± 14",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 147,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 102,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 154,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 1808,
            "range": "± 8",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 142186,
            "range": "± 250",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1555361,
            "range": "± 7472",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 795210,
            "range": "± 2930",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1079107,
            "range": "± 21501",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 448,
            "range": "± 210",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 4075,
            "range": "± 86",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 112,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 13,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 16,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 46,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 9850,
            "range": "± 177",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 37,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
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
          "id": "ae30b80a356c1bbd2eaa40a01a79ebd04954bcd9",
          "message": "feat: Add ECDSA signature scheme for secp256k1 (#1072)\n\n* feat: Add ECDSA signature scheme for secp256k1\n\nImplement ECDSA signing and verification for the secp256k1 curve:\n\n- sign(): Create signatures with (message_hash, private_key, nonce)\n- verify(): Verify signatures against public keys\n- derive_public_key(): Compute public key from private key\n- Signature serialization/deserialization (64 bytes)\n\nIncludes comprehensive test suite:\n- Sign and verify roundtrip\n- Invalid signature detection\n- Wrong message/public key rejection\n- Zero nonce rejection\n- Serialization roundtrip\n\nNote: This implementation is NOT constant-time and should only be\nused for verification or testing. Production signing requires\nconstant-time implementations.\n\n* refactor: Move ECDSA from math crate to examples (#1105)\n\n* refactor: Move ECDSA from math crate to examples\n\nECDSA is a signature scheme that uses elliptic curves, not an intrinsic\nproperty of the secp256k1 curve itself. Moving it to examples:\n\n- Aligns with existing pattern (schnorr-signature, frost-signature, rsa)\n- Keeps math library focused on mathematical primitives\n- Provides educational reference implementation\n- Shows users how to build crypto protocols with lambdaworks\n\nThe example includes:\n- Full ECDSA sign/verify implementation\n- Comprehensive README explaining the protocol\n- Security warnings about non-constant-time operations\n- References to RFC 6979 for production use\n\n* Update README.md\n\n* Add security improvements for ECDSA implementation (#1111)\n\n- Add module-level security warning about educational use only\n- Add InvalidPublicKey error variant\n- Add half_order() helper for low-S comparison\n- Add is_point_on_curve() for public key validation\n- Normalize signatures to low-S form in sign() to prevent malleability\n- Reject high-S signatures in verify()\n- Validate public key is on curve and not point at infinity\n- Add tests for low-S normalization, high-S rejection, and invalid public key\n\n---------\n\nCo-authored-by: jotabulacios <jbulacios@fi.uba.ar>",
          "timestamp": "2026-01-29T18:34:45-03:00",
          "tree_id": "b58a683445266454124bce5e416d6bbc5b93d8d2",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/ae30b80a356c1bbd2eaa40a01a79ebd04954bcd9"
        },
        "date": 1769723956487,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 329493646,
            "range": "± 1068285",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 367809043,
            "range": "± 955854",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 278289080,
            "range": "± 352467",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 693185777,
            "range": "± 1221701",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 774576465,
            "range": "± 3366950",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1450158790,
            "range": "± 651776",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1632314699,
            "range": "± 3154447",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1224010835,
            "range": "± 721230",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 3030870955,
            "range": "± 2188706",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3406441339,
            "range": "± 10948608",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6337487789,
            "range": "± 7181277",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7190053194,
            "range": "± 14964577",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5361903296,
            "range": "± 3052446",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7639944,
            "range": "± 9164",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7663702,
            "range": "± 3996",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 9732159,
            "range": "± 175673",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 9786673,
            "range": "± 146272",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 17644382,
            "range": "± 75839",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 17788199,
            "range": "± 59150",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 27342267,
            "range": "± 936098",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 27558580,
            "range": "± 1010019",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 35571662,
            "range": "± 38312",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 35769777,
            "range": "± 100738",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 66306486,
            "range": "± 526710",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 66850496,
            "range": "± 930983",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 72043336,
            "range": "± 153287",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 71803174,
            "range": "± 214378",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 139794374,
            "range": "± 1816709",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 138855396,
            "range": "± 1223998",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 144222904,
            "range": "± 388516",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 144108903,
            "range": "± 240754",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 280227115,
            "range": "± 2167063",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 276550003,
            "range": "± 1309291",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 10636177,
            "range": "± 741783",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 31978501,
            "range": "± 1109932",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 67244944,
            "range": "± 797447",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 138817467,
            "range": "± 1692018",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 355797048,
            "range": "± 5220551",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 361833924,
            "range": "± 913388",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 769419805,
            "range": "± 2110004",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1615705969,
            "range": "± 2233195",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3376448696,
            "range": "± 4478761",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 7102729920,
            "range": "± 5669786",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 389028128,
            "range": "± 1711904",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 813775264,
            "range": "± 2627676",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1705316773,
            "range": "± 2435565",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3528566979,
            "range": "± 7213087",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7416549500,
            "range": "± 4707601",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 250,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 8079,
            "range": "± 9",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 265,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 171,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 258,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 7072,
            "range": "± 388",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 144118,
            "range": "± 425",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1576955,
            "range": "± 7332",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 793600,
            "range": "± 1430",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1102348,
            "range": "± 10514",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 791,
            "range": "± 717",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 12281,
            "range": "± 221",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 255,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 22,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 6,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 79,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 11633,
            "range": "± 972",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 213,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 10659,
            "range": "± 765",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "jbulacios@fi.uba.ar",
            "name": "jotabulacios",
            "username": "jotabulacios"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "4950f8ee800827c3b76e6d64763b9dc7b8f85418",
          "message": "pin miden-vm to lambdaworks-backup branch (#1115)\n\n* pin miden-vm to lambdaworks-backup branch\n\n* use tag",
          "timestamp": "2026-01-30T19:19:45Z",
          "tree_id": "3fa41ecd51b81eabb3fc16616a855d5aedec2b5c",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/4950f8ee800827c3b76e6d64763b9dc7b8f85418"
        },
        "date": 1769802880826,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 325084788,
            "range": "± 277447",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 368797735,
            "range": "± 696419",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4",
            "value": 280742065,
            "range": "± 3658298",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 683295982,
            "range": "± 614049",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 771400527,
            "range": "± 1737623",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 1431668808,
            "range": "± 2315528",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 1624627258,
            "range": "± 6846232",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #2",
            "value": 1233890686,
            "range": "± 663185",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 2989115120,
            "range": "± 3168837",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 3396300027,
            "range": "± 9431595",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #5",
            "value": 6245159513,
            "range": "± 5855528",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #5",
            "value": 7152771213,
            "range": "± 17413416",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix4 #3",
            "value": 5404166937,
            "range": "± 2877771",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 7744638,
            "range": "± 19713",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 7878205,
            "range": "± 5213",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 10235071,
            "range": "± 191062",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 10148268,
            "range": "± 210115",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 17972135,
            "range": "± 33383",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 18043285,
            "range": "± 128576",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 29747274,
            "range": "± 802328",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 29463464,
            "range": "± 809903",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 35682034,
            "range": "± 118287",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 36637020,
            "range": "± 68686",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 68210208,
            "range": "± 367732",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 67987316,
            "range": "± 692956",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 73795450,
            "range": "± 208574",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 73044862,
            "range": "± 120055",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 139352294,
            "range": "± 753826",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 139692669,
            "range": "± 753164",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #5",
            "value": 144351061,
            "range": "± 146069",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #5",
            "value": 143854911,
            "range": "± 308292",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #5",
            "value": 279046762,
            "range": "± 1891539",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #5",
            "value": 279226708,
            "range": "± 1613124",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 10319773,
            "range": "± 374789",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 31566048,
            "range": "± 351985",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 65765115,
            "range": "± 982940",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 135204927,
            "range": "± 1212326",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #5",
            "value": 354686212,
            "range": "± 1930226",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 358770565,
            "range": "± 1038149",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 761757187,
            "range": "± 1612984",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 1599366777,
            "range": "± 2483608",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 3335842836,
            "range": "± 2389452",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #5",
            "value": 6995206637,
            "range": "± 9262419",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 386192031,
            "range": "± 863181",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 802746703,
            "range": "± 2846377",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 1682077483,
            "range": "± 1751648",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 3483490493,
            "range": "± 4536047",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #5",
            "value": 7303635379,
            "range": "± 6758902",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 12,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 29,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 32,
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
            "value": 32,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 52,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast_mul big poly",
            "value": 144724,
            "range": "± 474",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow mul big poly",
            "value": 1554447,
            "range": "± 11021",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/fast div big poly",
            "value": 808605,
            "range": "± 1629",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/slow div big poly",
            "value": 1069987,
            "range": "± 12276",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 276,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with generic div",
            "value": 265,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div by 'x - b' with Ruffini",
            "value": 10,
            "range": "± 37",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate #2",
            "value": 22,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_with",
            "value": 6,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/merge",
            "value": 77,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add #2",
            "value": 11494,
            "range": "± 918",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul #2",
            "value": 201,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 3",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 4",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 5",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 6",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 7",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 8",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 9",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate 10",
            "value": 1,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}