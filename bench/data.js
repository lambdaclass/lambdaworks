window.BENCHMARK_DATA = {
  "lastUpdate": 1685988800814,
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
          "id": "943963cdfb977c2f54bf6dca60e5475985c30606",
          "message": "Add cairo memory and range check constraints (#337)\n\n* add trace commitments to transcript\n\n* move sampling of z to round 3\n\n* add batch_commit function\n\n* add commitments of the composition polynomial to the transcript\n\n* refactor sampling of boundary and transition coefficients\n\n* add ood evaluations to transcript\n\n* minor refactor\n\n* move sample batch to lib\n\n* extract deep composition randomness to round 4\n\n* refactor fri commitment phase\n\n* refactor next_fri_layer\n\n* remove last iteration of fri commit phase\n\n* refactor fri_commit_phase\n\n* move sampling of q_0 to query phase. Rename\n\n* refactor fri decommitment\n\n* add fri last value to proof and remove it from decommitments\n\n* remove layers commitments from decommitment\n\n* remove unused FriQuery struct\n\n* leave only symmetric points in the proof\n\n* remove unnecesary last fri layer\n\n* reuse composition poly evaluations from round_1 in the consistency check\n\n* minor refactor\n\n* fix trace ood commitments and small refactor\n\n* move fri_query_phase to fri mod\n\n* minor refactor in build deeep composition poly function\n\n* refactor deep composition poly related code\n\n* minor modifications to comments\n\n* clippy\n\n* add comments\n\n* move iota sampling to step 0 and rename challenges\n\n* minor refactor and add missing opening checks\n\n* minor refactor\n\n* move transition divisor computation outside of the main loop of the computation of the composition polynomial\n\n* add protocol to docs\n\n* clippy and fmt\n\n* remove old test\n\n* fix typo\n\n* fmt\n\n* remove unnecessary public attribute\n\n* move build_execution trace to round 1\n\n* add rap support to air\n\n* remove commit_original_trace function\n\n* Add auxiliary trace polynomials and commitments in the first round\n\n* Test simple fibonacci\n\n* fix format in starks protocol docs\n\n* fmt, clippy\n\n* remove comments, add sampling of cairo rap\n\n* add commented test\n\n* remove old test\n\n* First step of memory constraints\n\n* Add public input to AIR. Build auxiliary columns for Cairo.\n\n* Fix bug in build_aux\n\n* Cargo fmt\n\n* fix aux table bug\n\n* fix number_auxiliary_rap_columns\n\n* Remove program_size parameter for CairoAIR\n\n* Refactor merge PublicInputs\n\n* Add transition degree in fibonacci rap\n\n* Tests for build auxiliary trace\n\n* Add auxiliary transition constraints for memory\n\n* Add final permutation argumentation boundary auxiliary constraint\n\n* WIP: boundary memory constraints\n\n* fix memory indexing\n\n* make public input mutable argument of build_main_trace\n\n* add range check column build function\n\n* Add missing values in offsets columns. Concatenate sorted offsets to original trace\n\n* add test to concatenate. Fix offset variables\n\n* add permutation argument column for range checks\n\n* add range_check permutation argument transition constraints\n\n* add range checks continuity transition constraints\n\n* Add debug assertion for verifying divisibility of transition polynomials by their zerofiers\n\n* Add boundary constraint for range checks\n\n* Add boundary constraints range check min and max\n\n* Add dummy AIR and test to debug\n\n* fix evaluate_polynomial_on_lde_domain\n\n* Remove default implementation for number of auxiliary RAP columns\n\n* fix composition poly bound degree\n\n* remove unused public input attribute\n\n* clippy\n\n* fix dummy test\n\n* Remove check polynomial divisibility\n\n* Add test of malicious program\n\n* Add simple_program.json for testing purposes\n\n* Remove some useless commented code\n\n* Fix: malicious test uses also malicious memory\n\n* Cargo\n\n* Add comments\n\n* comment transition exemptions vector\n\n* update stark protocol to include range checks\n\n* use markdown auto numbered lists\n\n* Ignore test_prove_cairo_simple_program for Metal\n\n* move boundary term degree adjustment power out of the closure\n\n* Add test for range checks. Rename malicious test.\n\n* Helper function to load cairo programs and memory\n\n* Return Error when trace length is not enough for the Cairo trace to fit in\n\n* Return an error when trace length is not large enough. Add larger\nfibonacci tests.\n\n* clippy\n\n* ignore large test cases\n\n---------\n\nCo-authored-by: ajgarassino <ajgarassino@gmail.com>\nCo-authored-by: Mariano Nicolini <mariano.nicolini.91@gmail.com>\nCo-authored-by: Sergio Chouhy <schouhy@eryxsoluciones.com.ar>",
          "timestamp": "2023-05-25T09:40:44Z",
          "tree_id": "cd9853079a2588775eb36934c8d10b13ec7c0046",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/943963cdfb977c2f54bf6dca60e5475985c30606"
        },
        "date": 1685007989914,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 131503467,
            "range": "± 4314482",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 248141614,
            "range": "± 19962184",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 481096083,
            "range": "± 2121285",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 975324958,
            "range": "± 8794766",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 34464143,
            "range": "± 250819",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 68014087,
            "range": "± 438897",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 132877031,
            "range": "± 631354",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 278139822,
            "range": "± 3846727",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 31187599,
            "range": "± 345414",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 58582782,
            "range": "± 988400",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 122859854,
            "range": "± 3210673",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 263854364,
            "range": "± 20041801",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 164858428,
            "range": "± 751532",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 331561989,
            "range": "± 1197276",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 653177895,
            "range": "± 4825007",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1302930104,
            "range": "± 10709339",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 456422531,
            "range": "± 1391724",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 915871750,
            "range": "± 5743565",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1800456646,
            "range": "± 3952873",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3535342542,
            "range": "± 16118794",
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
          "id": "943963cdfb977c2f54bf6dca60e5475985c30606",
          "message": "Add cairo memory and range check constraints (#337)\n\n* add trace commitments to transcript\n\n* move sampling of z to round 3\n\n* add batch_commit function\n\n* add commitments of the composition polynomial to the transcript\n\n* refactor sampling of boundary and transition coefficients\n\n* add ood evaluations to transcript\n\n* minor refactor\n\n* move sample batch to lib\n\n* extract deep composition randomness to round 4\n\n* refactor fri commitment phase\n\n* refactor next_fri_layer\n\n* remove last iteration of fri commit phase\n\n* refactor fri_commit_phase\n\n* move sampling of q_0 to query phase. Rename\n\n* refactor fri decommitment\n\n* add fri last value to proof and remove it from decommitments\n\n* remove layers commitments from decommitment\n\n* remove unused FriQuery struct\n\n* leave only symmetric points in the proof\n\n* remove unnecesary last fri layer\n\n* reuse composition poly evaluations from round_1 in the consistency check\n\n* minor refactor\n\n* fix trace ood commitments and small refactor\n\n* move fri_query_phase to fri mod\n\n* minor refactor in build deeep composition poly function\n\n* refactor deep composition poly related code\n\n* minor modifications to comments\n\n* clippy\n\n* add comments\n\n* move iota sampling to step 0 and rename challenges\n\n* minor refactor and add missing opening checks\n\n* minor refactor\n\n* move transition divisor computation outside of the main loop of the computation of the composition polynomial\n\n* add protocol to docs\n\n* clippy and fmt\n\n* remove old test\n\n* fix typo\n\n* fmt\n\n* remove unnecessary public attribute\n\n* move build_execution trace to round 1\n\n* add rap support to air\n\n* remove commit_original_trace function\n\n* Add auxiliary trace polynomials and commitments in the first round\n\n* Test simple fibonacci\n\n* fix format in starks protocol docs\n\n* fmt, clippy\n\n* remove comments, add sampling of cairo rap\n\n* add commented test\n\n* remove old test\n\n* First step of memory constraints\n\n* Add public input to AIR. Build auxiliary columns for Cairo.\n\n* Fix bug in build_aux\n\n* Cargo fmt\n\n* fix aux table bug\n\n* fix number_auxiliary_rap_columns\n\n* Remove program_size parameter for CairoAIR\n\n* Refactor merge PublicInputs\n\n* Add transition degree in fibonacci rap\n\n* Tests for build auxiliary trace\n\n* Add auxiliary transition constraints for memory\n\n* Add final permutation argumentation boundary auxiliary constraint\n\n* WIP: boundary memory constraints\n\n* fix memory indexing\n\n* make public input mutable argument of build_main_trace\n\n* add range check column build function\n\n* Add missing values in offsets columns. Concatenate sorted offsets to original trace\n\n* add test to concatenate. Fix offset variables\n\n* add permutation argument column for range checks\n\n* add range_check permutation argument transition constraints\n\n* add range checks continuity transition constraints\n\n* Add debug assertion for verifying divisibility of transition polynomials by their zerofiers\n\n* Add boundary constraint for range checks\n\n* Add boundary constraints range check min and max\n\n* Add dummy AIR and test to debug\n\n* fix evaluate_polynomial_on_lde_domain\n\n* Remove default implementation for number of auxiliary RAP columns\n\n* fix composition poly bound degree\n\n* remove unused public input attribute\n\n* clippy\n\n* fix dummy test\n\n* Remove check polynomial divisibility\n\n* Add test of malicious program\n\n* Add simple_program.json for testing purposes\n\n* Remove some useless commented code\n\n* Fix: malicious test uses also malicious memory\n\n* Cargo\n\n* Add comments\n\n* comment transition exemptions vector\n\n* update stark protocol to include range checks\n\n* use markdown auto numbered lists\n\n* Ignore test_prove_cairo_simple_program for Metal\n\n* move boundary term degree adjustment power out of the closure\n\n* Add test for range checks. Rename malicious test.\n\n* Helper function to load cairo programs and memory\n\n* Return Error when trace length is not enough for the Cairo trace to fit in\n\n* Return an error when trace length is not large enough. Add larger\nfibonacci tests.\n\n* clippy\n\n* ignore large test cases\n\n---------\n\nCo-authored-by: ajgarassino <ajgarassino@gmail.com>\nCo-authored-by: Mariano Nicolini <mariano.nicolini.91@gmail.com>\nCo-authored-by: Sergio Chouhy <schouhy@eryxsoluciones.com.ar>",
          "timestamp": "2023-05-25T09:40:44Z",
          "tree_id": "cd9853079a2588775eb36934c8d10b13ec7c0046",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/943963cdfb977c2f54bf6dca60e5475985c30606"
        },
        "date": 1685010429220,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 1627068009,
            "range": "± 10868121",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 3892612129,
            "range": "± 77041785",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 3395123876,
            "range": "± 5676327",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 8552195085,
            "range": "± 61548463",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 7067305799,
            "range": "± 6287837",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 18341367412,
            "range": "± 89841504",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 14729824848,
            "range": "± 15221688",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 39242362801,
            "range": "± 529624859",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential",
            "value": 1154489496,
            "range": "± 1383378",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #2",
            "value": 2419840374,
            "range": "± 3255042",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #3",
            "value": 5072167223,
            "range": "± 4063114",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #4",
            "value": 10599014256,
            "range": "± 8163458",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 82600361,
            "range": "± 3500586",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 164438756,
            "range": "± 2049624",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 336847889,
            "range": "± 6851403",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 668649937,
            "range": "± 10956530",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 2801689561,
            "range": "± 6932607",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 5859908944,
            "range": "± 6062540",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 12263922694,
            "range": "± 13167590",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 25435395707,
            "range": "± 17499229",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 3012304017,
            "range": "± 3008576",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 6302840024,
            "range": "± 2862855",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 13107479122,
            "range": "± 7515206",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 27251055831,
            "range": "± 51458368",
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
            "value": 35,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 84,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 17,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 156,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 119,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 119,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/8",
            "value": 1383979,
            "range": "± 396",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/128",
            "value": 26872750,
            "range": "± 6048",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/256",
            "value": 78151890,
            "range": "± 155968",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/512",
            "value": 255266107,
            "range": "± 346243",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/1024",
            "value": 910254037,
            "range": "± 2644000",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/2 column Fibonacci",
            "value": 949639,
            "range": "± 1163",
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
          "id": "bf93accc0ea0401499c514634dac064dea4f12eb",
          "message": "Remove plonk (#379)\n\n* Remove plonk\r\n\r\n* Update gitignore\r\n\r\n* Remove DS Store",
          "timestamp": "2023-05-25T08:22:13-03:00",
          "tree_id": "87b52a97812cc51b857b130adf551b025d5d176d",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/bf93accc0ea0401499c514634dac064dea4f12eb"
        },
        "date": 1685014089841,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 128996345,
            "range": "± 2161765",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 256197906,
            "range": "± 1387115",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 487212520,
            "range": "± 5320694",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 981927812,
            "range": "± 10392032",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 34196275,
            "range": "± 174178",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 67784818,
            "range": "± 507683",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 144628746,
            "range": "± 5204995",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 297874958,
            "range": "± 6323514",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 36963347,
            "range": "± 1943273",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 60371793,
            "range": "± 2779711",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 118825035,
            "range": "± 4304820",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 237915229,
            "range": "± 3152582",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 233118020,
            "range": "± 24756147",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 447166635,
            "range": "± 40835955",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 1136933771,
            "range": "± 20024922",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 2514252854,
            "range": "± 108715456",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 467387260,
            "range": "± 2950213",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 926398229,
            "range": "± 4028138",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1925434333,
            "range": "± 118453923",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 4272485833,
            "range": "± 130077728",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mail@fcarrone.com",
            "name": "Federico Carrone",
            "username": "unbalancedparentheses"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "61b26ab9e89e5dcd1c7081fa0beebcde1d58445f",
          "message": "Update README.md (#380)",
          "timestamp": "2023-05-25T13:24:01+02:00",
          "tree_id": "b4ce60a8101c097ba371ae32e56154241e06f554",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/61b26ab9e89e5dcd1c7081fa0beebcde1d58445f"
        },
        "date": 1685014188512,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 138548295,
            "range": "± 4142052",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 267261552,
            "range": "± 9165221",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 648007521,
            "range": "± 79751754",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 1394477104,
            "range": "± 30737531",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 39696783,
            "range": "± 1737617",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 78650968,
            "range": "± 5463998",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 149374437,
            "range": "± 9139138",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 306033593,
            "range": "± 64436788",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 31653540,
            "range": "± 426199",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 59464131,
            "range": "± 1326287",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 116437808,
            "range": "± 2621343",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 230319027,
            "range": "± 5191552",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 235418187,
            "range": "± 226151387",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 463869875,
            "range": "± 699461639",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 1030519854,
            "range": "± 846841051",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1323081250,
            "range": "± 788981557",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 456829187,
            "range": "± 2019273",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 923789750,
            "range": "± 8540651",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1798817520,
            "range": "± 6020321",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3529748000,
            "range": "± 20291106",
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
          "id": "141a46c422856c12aaae7774ef87bc3d0750c6d9",
          "message": "Revert \"Fix STARK prover benchmarks (#354)\" (#377)\n\nThis reverts commit 3d33d9457b8279f65c959924a39ce029c0386af2.",
          "timestamp": "2023-05-25T06:33:46-03:00",
          "tree_id": "38db3caa9e582c9f2ce243eec16952744cb2b086",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/141a46c422856c12aaae7774ef87bc3d0750c6d9"
        },
        "date": 1685016459610,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 1933959696,
            "range": "± 10201072",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 4326542461,
            "range": "± 36507459",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 3982139426,
            "range": "± 11716186",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 9501294073,
            "range": "± 35363840",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 8350178596,
            "range": "± 44327496",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 20582578076,
            "range": "± 75701123",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 17254674438,
            "range": "± 106727089",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 44100180856,
            "range": "± 346088082",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential",
            "value": 1384578516,
            "range": "± 1612630",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #2",
            "value": 2889094412,
            "range": "± 9397640",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #3",
            "value": 6056187985,
            "range": "± 24883948",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #4",
            "value": 12706537350,
            "range": "± 2507022",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 88149995,
            "range": "± 237229",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 178323837,
            "range": "± 761750",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 354394736,
            "range": "± 688306",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 723500737,
            "range": "± 7487269",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 3346004959,
            "range": "± 1278987",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 6997164029,
            "range": "± 76454461",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 14614508169,
            "range": "± 7706202",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 30379950246,
            "range": "± 177683519",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 3601798455,
            "range": "± 12974185",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 7441315207,
            "range": "± 18054877",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 15433403239,
            "range": "± 246212417",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 31740618399,
            "range": "± 356354543",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 454,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 7273,
            "range": "± 37",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 457,
            "range": "± 7",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 23,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 692,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 479,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 625,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/8",
            "value": 2063329,
            "range": "± 35347",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/128",
            "value": 1022250154,
            "range": "± 5439745",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/256",
            "value": 7844194863,
            "range": "± 39503878",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/512",
            "value": 61461112202,
            "range": "± 337878038",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/Simple Fibonacci/1024",
            "value": 482421826918,
            "range": "± 3758753594",
            "unit": "ns/iter"
          },
          {
            "name": "STARK/2 column Fibonacci",
            "value": 1287608,
            "range": "± 15094",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "estefano.bargas@fing.edu.uy",
            "name": "Estéfano Bargas",
            "username": "xqft"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "422e57513d8d9b86b398e92d6ad7ee5d38d9fdbb",
          "message": "Remove stark criterion benches from CI (#381)",
          "timestamp": "2023-05-26T09:43:05Z",
          "tree_id": "ca086bf41de7a431cada702e211811f6dd2d5aa0",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/422e57513d8d9b86b398e92d6ad7ee5d38d9fdbb"
        },
        "date": 1685094540574,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 128862239,
            "range": "± 2839819",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 255122687,
            "range": "± 1532030",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 489138916,
            "range": "± 3858098",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 990766604,
            "range": "± 13359028",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 34274338,
            "range": "± 456718",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 68168964,
            "range": "± 454163",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 133158623,
            "range": "± 1856548",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 276797260,
            "range": "± 2332844",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 31158982,
            "range": "± 163348",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 55511352,
            "range": "± 3172042",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 121245016,
            "range": "± 3315653",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 244752444,
            "range": "± 18522935",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 164661518,
            "range": "± 896110",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 330912125,
            "range": "± 2079146",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 658372666,
            "range": "± 5263557",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1325714291,
            "range": "± 32731242",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 456346614,
            "range": "± 5027036",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 916959583,
            "range": "± 7885323",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1798091479,
            "range": "± 8205384",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3500912021,
            "range": "± 35355525",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "estefano.bargas@fing.edu.uy",
            "name": "Estéfano Bargas",
            "username": "xqft"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "422e57513d8d9b86b398e92d6ad7ee5d38d9fdbb",
          "message": "Remove stark criterion benches from CI (#381)",
          "timestamp": "2023-05-26T09:43:05Z",
          "tree_id": "ca086bf41de7a431cada702e211811f6dd2d5aa0",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/422e57513d8d9b86b398e92d6ad7ee5d38d9fdbb"
        },
        "date": 1685096880467,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 1634616142,
            "range": "± 2383837",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 4211436569,
            "range": "± 34184913",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 3407962209,
            "range": "± 3314629",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 9184112146,
            "range": "± 31731647",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 7110200009,
            "range": "± 8491203",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 19752949141,
            "range": "± 39798555",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 14807727014,
            "range": "± 7382567",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 41758891522,
            "range": "± 280311277",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential",
            "value": 1153713185,
            "range": "± 193935",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #2",
            "value": 2420380570,
            "range": "± 1234613",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #3",
            "value": 5068030527,
            "range": "± 1324635",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #4",
            "value": 10590095795,
            "range": "± 2647691",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 90929473,
            "range": "± 678481",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 186140236,
            "range": "± 957974",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 369233463,
            "range": "± 2074842",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 747496636,
            "range": "± 4758684",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 2817214873,
            "range": "± 3865601",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 5885693918,
            "range": "± 3473009",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 12282137835,
            "range": "± 4838269",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 25565152920,
            "range": "± 22685614",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 3047168630,
            "range": "± 1821828",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 6343230987,
            "range": "± 3142128",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 13187689925,
            "range": "± 5445118",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 27420739663,
            "range": "± 13250606",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 378,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 6063,
            "range": "± 5",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 390,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 19,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 594,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 429,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 429,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mariano.nicolini.91@gmail.com",
            "name": "Mariano A. Nicolini",
            "username": "entropidelic"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "599b356aaa37f3971e507eb00d3f6860140c7a51",
          "message": "Change field in math benchmarks to Stark field (#383)",
          "timestamp": "2023-05-26T08:41:24-03:00",
          "tree_id": "4bc60338d977e43fec8596b93e7bbe35e9ce81e3",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/599b356aaa37f3971e507eb00d3f6860140c7a51"
        },
        "date": 1685101615493,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 128448479,
            "range": "± 3339125",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 254473562,
            "range": "± 2625581",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 483488677,
            "range": "± 4896354",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 984364250,
            "range": "± 14278988",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 33974463,
            "range": "± 242793",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 67609252,
            "range": "± 624949",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 133146864,
            "range": "± 806196",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 275412312,
            "range": "± 2999974",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 30793910,
            "range": "± 288014",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 59052981,
            "range": "± 479872",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 121317833,
            "range": "± 4584346",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 248518187,
            "range": "± 20718994",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 163162751,
            "range": "± 757268",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 329741489,
            "range": "± 3501862",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 660018354,
            "range": "± 2889649",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1304457479,
            "range": "± 22631424",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 459235854,
            "range": "± 1445626",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 909413749,
            "range": "± 6058018",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1796377875,
            "range": "± 8899692",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3537617812,
            "range": "± 24256173",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mariano.nicolini.91@gmail.com",
            "name": "Mariano A. Nicolini",
            "username": "entropidelic"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "599b356aaa37f3971e507eb00d3f6860140c7a51",
          "message": "Change field in math benchmarks to Stark field (#383)",
          "timestamp": "2023-05-26T08:41:24-03:00",
          "tree_id": "4bc60338d977e43fec8596b93e7bbe35e9ce81e3",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/599b356aaa37f3971e507eb00d3f6860140c7a51"
        },
        "date": 1685103902746,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 1620400245,
            "range": "± 2109808",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 3901033234,
            "range": "± 21351887",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 3383611397,
            "range": "± 13880794",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 8500882439,
            "range": "± 10655422",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 7055064460,
            "range": "± 6694192",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 18326411754,
            "range": "± 22871303",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 14707618495,
            "range": "± 12413474",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 39031886940,
            "range": "± 64380359",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential",
            "value": 1154286793,
            "range": "± 591655",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #2",
            "value": 2420862095,
            "range": "± 1127819",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #3",
            "value": 5070668836,
            "range": "± 770846",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #4",
            "value": 10592744685,
            "range": "± 1688423",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 80226064,
            "range": "± 254664",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 160893718,
            "range": "± 1039781",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 323701124,
            "range": "± 986893",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 665185068,
            "range": "± 2977145",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 2800154306,
            "range": "± 2400760",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 5856080738,
            "range": "± 4977462",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 12224620705,
            "range": "± 9050254",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 25491883778,
            "range": "± 11588209",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 3024642325,
            "range": "± 3317181",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 6305694321,
            "range": "± 5635069",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 13116540894,
            "range": "± 7809667",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 27275264044,
            "range": "± 7337233",
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
            "value": 34,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 85,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 17,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 157,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 119,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 118,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "juanbono94@gmail.com",
            "name": "Juan Bono",
            "username": "juanbono"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b5b938d27bc2436738acfa70efd1888769f50417",
          "message": "inline cios (#385)",
          "timestamp": "2023-05-26T12:50:44-03:00",
          "tree_id": "7a28a497c70e7403855ca44a4a3f91dbab111421",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/b5b938d27bc2436738acfa70efd1888769f50417"
        },
        "date": 1685116636228,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 139023706,
            "range": "± 3798061",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 274355448,
            "range": "± 40377570",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 679639166,
            "range": "± 13961238",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 1411421666,
            "range": "± 9222080",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 37506242,
            "range": "± 1493951",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 75255769,
            "range": "± 1622221",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 197671125,
            "range": "± 4893877",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 408291781,
            "range": "± 18822115",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 31915074,
            "range": "± 483034",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 58831920,
            "range": "± 3883532",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 123617891,
            "range": "± 1908210",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 262964861,
            "range": "± 14086123",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 180152794,
            "range": "± 8438447",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 439557198,
            "range": "± 14850367",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 829112458,
            "range": "± 46207746",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1798563812,
            "range": "± 118789330",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 646727771,
            "range": "± 104040964",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 1389940687,
            "range": "± 170147403",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 2788698937,
            "range": "± 245589989",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 5652175020,
            "range": "± 212068806",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mariano.nicolini.91@gmail.com",
            "name": "Mariano A. Nicolini",
            "username": "entropidelic"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "75423a1b88fbcb2bf0308db445e268f416c377b7",
          "message": "Add div2 method for unsigned integer (#384)\n\n* Add div2 method for unsigned integer\r\n\r\n* Add allow unused lint to div2",
          "timestamp": "2023-05-26T12:50:55-03:00",
          "tree_id": "eb808a4e18c044afc8ed78fccb234b573b9e15ff",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/75423a1b88fbcb2bf0308db445e268f416c377b7"
        },
        "date": 1685116647015,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 177425354,
            "range": "± 19715929",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 356192635,
            "range": "± 51595596",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 723433375,
            "range": "± 103802590",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 1891749437,
            "range": "± 90264516",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 49592187,
            "range": "± 2149559",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 102263721,
            "range": "± 4768830",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 144363889,
            "range": "± 4866306",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 289644416,
            "range": "± 1116331",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 33289374,
            "range": "± 1026064",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 61763675,
            "range": "± 1088559",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 123997493,
            "range": "± 9251694",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 243264541,
            "range": "± 2981588",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 219911382,
            "range": "± 2113325",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 442038781,
            "range": "± 3915094",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 939331187,
            "range": "± 63262423",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1824562396,
            "range": "± 231752047",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 945180708,
            "range": "± 255024695",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 1444856771,
            "range": "± 507967617",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 2780987146,
            "range": "± 799474611",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 5509832187,
            "range": "± 1103182473",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "juanbono94@gmail.com",
            "name": "Juan Bono",
            "username": "juanbono"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b5b938d27bc2436738acfa70efd1888769f50417",
          "message": "inline cios (#385)",
          "timestamp": "2023-05-26T12:50:44-03:00",
          "tree_id": "7a28a497c70e7403855ca44a4a3f91dbab111421",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/b5b938d27bc2436738acfa70efd1888769f50417"
        },
        "date": 1685118496300,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 1445614152,
            "range": "± 1136047",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 3448184691,
            "range": "± 16050906",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 3019672325,
            "range": "± 4050511",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 7590172531,
            "range": "± 27894992",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 6292952189,
            "range": "± 5445270",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 16299191638,
            "range": "± 55996357",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 13110274562,
            "range": "± 5142054",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 34646588858,
            "range": "± 308440990",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential",
            "value": 850170750,
            "range": "± 1021564",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #2",
            "value": 1786644564,
            "range": "± 1374322",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #3",
            "value": 3742125702,
            "range": "± 1730124",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #4",
            "value": 7834755149,
            "range": "± 2365538",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 74638583,
            "range": "± 2456884",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 160451887,
            "range": "± 2167788",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 319943870,
            "range": "± 2911558",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 644542042,
            "range": "± 1479996",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 2320571599,
            "range": "± 8355280",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 4858260753,
            "range": "± 4434250",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 10140975131,
            "range": "± 5498498",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 21128565537,
            "range": "± 16430723",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 2505804719,
            "range": "± 2922835",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 5215635040,
            "range": "± 3834271",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 10862655565,
            "range": "± 25016345",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 22566989543,
            "range": "± 27860902",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 379,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 6067,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 390,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 20,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 599,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 411,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 527,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mariano.nicolini.91@gmail.com",
            "name": "Mariano A. Nicolini",
            "username": "entropidelic"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "75423a1b88fbcb2bf0308db445e268f416c377b7",
          "message": "Add div2 method for unsigned integer (#384)\n\n* Add div2 method for unsigned integer\r\n\r\n* Add allow unused lint to div2",
          "timestamp": "2023-05-26T12:50:55-03:00",
          "tree_id": "eb808a4e18c044afc8ed78fccb234b573b9e15ff",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/75423a1b88fbcb2bf0308db445e268f416c377b7"
        },
        "date": 1685118966163,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 1464190653,
            "range": "± 24212917",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 4400543919,
            "range": "± 106188162",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 3224313447,
            "range": "± 226649788",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 9251506270,
            "range": "± 86360203",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 7115663339,
            "range": "± 512993840",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 21213003976,
            "range": "± 60714149",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 15537765566,
            "range": "± 58788765",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 44793845231,
            "range": "± 127397248",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential",
            "value": 1027712395,
            "range": "± 9794195",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #2",
            "value": 2162366590,
            "range": "± 19311961",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #3",
            "value": 4523169564,
            "range": "± 20721607",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #4",
            "value": 9509305056,
            "range": "± 42444077",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 102029428,
            "range": "± 2341254",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 203506397,
            "range": "± 3609861",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 413303962,
            "range": "± 6378264",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 834003113,
            "range": "± 11822526",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 2774734779,
            "range": "± 16699167",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 5812065798,
            "range": "± 17473414",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 12153190424,
            "range": "± 45455780",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 25317365738,
            "range": "± 76107074",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 3003548849,
            "range": "± 22236738",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 6267089099,
            "range": "± 28727457",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 13024884967,
            "range": "± 43747888",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 26999689778,
            "range": "± 126216471",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 98,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 416,
            "range": "± 32",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 141,
            "range": "± 32",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 19,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 234,
            "range": "± 34",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 185,
            "range": "± 41",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 186,
            "range": "± 37",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "maigonzalez@fi.uba.ar",
            "name": "Matías Ignacio González",
            "username": "matias-gonz"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f6ffed48b1b96d55ff6d994195822c0e5b36acf9",
          "message": "Add gen_twiddles implementation in CUDA (#310)\n\n* Added cuda mod and poc\n\n* Finished poc\n\n* Working CUDA FFT POC\n\n* Added cuda ptx compilation make rule\n\n* Added CUDA u256 prime field\n\n* Added CI job for testing with CUDA\n\n* Add CUDAFieldElement\n\n* Added support for u256 montgomery field\n\n* Remove unwrap()s\n\n* Add evaluate_fft_cuda\n\n* Remove default feature cuda\n\n* Remove default feature cuda\n\n* Remove unnecessary reference\n\n* Fix clippy errors\n\nNOTE: we currently don't have a linting job in the CI for the _metal_ and _cuda_ features\n\n* Fix benches error\n\n* Rename `IsTwoAdicField` -> `IsFFTField`\n\n* Fix cannot find function error\n\n* Add TODO\n\n* Interpolate fft cuda (#300)\n\n* Add interpolate_fft_cuda\r\n\r\n* Fix RootsConfig\r\n\r\n* Remove unnecessary coefficients\r\n\r\n* Add not(feature = \"cuda\")\r\n\r\n* Add unwrap for interpolate_fft\n\n* Add error handling for CUDA's fft implementation (#301)\n\n* Move cuda/field to cuda/abstractions\r\n\r\nThis is to more closely mimic the metal dir structure\r\n\r\n* Move helpers from metal to crate root\r\n\r\n* Add `CudaError`\r\n\r\n* Move functions, remove errors\r\n\r\n* Add CudaError variants for fft\r\n\r\n* Add TODO\r\n\r\n* Remove default metal feature\r\n\r\n* Fix compile errors\r\n\r\n* Fix missing imports errors\r\n\r\n* Fix compile errors\r\n\r\n* Allow dead_code in helpers module\n\n* Remove unwrap from interpolate_fft\n\n* Add calc_twiddles* CUDA functions\n\n* Fix CUDA compile errors\n\n* Fix more CUDA compile errors\n\n* Recompile metallib\n\n* Separate twiddle functions into different files\n\n* Update makefile\n\n* Compile all `.cu` files in `CUDAPATH`\n\n* Compile all `.cu` files in `CUDAPATH`\n\n* Recombine all twiddles functions in the same file\n\nSomething's up with `Fp256::inverse`.\nUsing the function inside a kernel causes compilation to freeze.\n\n* Comment out `calc_twiddles*_inv` functions\n\nThese functions slow down compilation to a halt, and are equivalent to calling `calc_twiddles*` with an inverted `_omega`\n\n* Refactor get_twiddles so that it uses cuda\n\n* Fix PTX\n\n* Fix compile errors\n\n* Fix CUDAFelt from\n\n* Fix comparision chain\n\n* Add missing slice\n\n* Add missing type annotation\n\n* Change ? to unwrap\n\n* Remove reference\n\n* Change input param type\n\n* Update param types\n\n* Update calc_twiddles_bitrev\n\n* Change root\n\n* Load all functions\n\n* Change types to unsigned\n\n* Refactor reference\n\n* Update LaunchConfig\n\n* Update Block size\n\n* Use vec!\n\n* Update blockdim\n\n* Refactor twiddles\n\n* Add twiddles.ptx\n\n* Fix imports\n\n* Remove unwraps\n\n* Fix errors\n\n* Add map_err\n\n* Add missing map_err\n\n* Remove unused import\n\n* Update interpolate_fft\n\n* Integrate _CUDA_ implementation with _fft_ crate (#298)\n\n* Add evaluate_fft_cuda\r\n\r\n* Remove default feature cuda\r\n\r\n* Remove default feature cuda\r\n\r\n* Remove unnecessary reference\r\n\r\n* Fix clippy errors\r\n\r\nNOTE: we currently don't have a linting job in the CI for the _metal_ and _cuda_ features\r\n\r\n* Fix benches error\r\n\r\n* Fix cannot find function error\r\n\r\n* Add TODO\r\n\r\n* Interpolate fft cuda (#300)\r\n\r\n* Add interpolate_fft_cuda\r\n\r\n* Fix RootsConfig\r\n\r\n* Remove unnecessary coefficients\r\n\r\n* Add not(feature = \"cuda\")\r\n\r\n* Add unwrap for interpolate_fft\r\n\r\n* Add error handling for CUDA's fft implementation (#301)\r\n\r\n* Move cuda/field to cuda/abstractions\r\n\r\nThis is to more closely mimic the metal dir structure\r\n\r\n* Move helpers from metal to crate root\r\n\r\n* Add `CudaError`\r\n\r\n* Move functions, remove errors\r\n\r\n* Add CudaError variants for fft\r\n\r\n* Add TODO\r\n\r\n* Remove default metal feature\r\n\r\n* Fix compile errors\r\n\r\n* Fix missing imports errors\r\n\r\n* Fix compile errors\r\n\r\n* Allow dead_code in helpers module\r\n\r\n* Remove unwrap from interpolate_fft\r\n\r\n* Add `CudaState` as a wrapper around cudarc primitives (#311)\r\n\r\n* Add CudaState\r\n\r\n* Use CudaState in `fft` function\r\n\r\n* Remove old attributes\r\n\r\n* Remove `unwrap`s in Metal and Cuda state init\r\n\r\n* Extract library loading to helper function\r\n\r\n* Fix compilation errors and move LaunchConfig use\r\n\r\n* Remove unnecesary modulo operation\r\n\r\nThe `threadIdx.x` builtin variable goes from 0 to `blockDim.x` (non-inclusive) so we don't need the modulo.\r\n\r\n* Add bounds checking to launch\r\n\r\n* Fix compilation errors\r\n\r\n* Fix all compile errors\r\n\r\n* Recompile fft.ptx\r\n\r\n---------\r\n\r\nCo-authored-by: matias-gonz <maigonzalez@fi.uba.ar>\n\n* Fix compile error\n\n* Fix compilation errors\n\n* Use prop_assert_eq instead of assert_eq\n\n* Remove unused fp.cuh\n\n* Don't use `prop_filter` for `field_vec`\n\nThe use of `prop_filter` slows tests down, and can cause nondeterministic test failures when the filter function true/false ratio is too low.\nIn this case, using it would cause tests with a too high max exponent to fail.\n\n* Refactor index calculation in CUDA\n\n* Fix compile errors\n\n* Update evaluate fft cuda call\n\n* Update evaluate_fft_cpu call\n\n* Remove repeated code\n\n* Remove unused constant\n\n* Add missing cfg not cuda\n\n* Fix cfg\n\n* Refactor gen_twiddles using CudaState\n\n* Fix compile errors\n\n* Add state to function call\n\n* Update GenTwiddles.launch\n\n* Fix warnings\n\n* Remove cfg\n\n* Fix cfg\n\n* Update felt condition\n\n* Remove mut\n\n* Update error\n\n* Change library import\n\n* Add namespace\n\n* Update block dim\n\n* Add casts\n\n* Fix unsigned\n\n* Update makefile\n\n* Add turbofish to gen_twiddles\n\n* Add turbofish to get_calc_twiddles\n\n* Comment code\n\n* Update omega param\n\n* Update block dim\n\n* Add assert\n\n* Add casts\n\n* Update twiddles\n\n* Fix map\n\n* Add type annotation\n\n* Add turbo fish\n\n* Add type annotation\n\n* Fix turbofish\n\n* Fix turbofish type annotation\n\n* Add assert\n\n* Add guard clause\n\n* Implement gen_twiddles in stark256 file\n\n* Template twiddles\n\n* Fix Fp\n\n* Delete unused files\n\n* Update stark256.ptx\n\n---------\n\nCo-authored-by: Estéfano Bargas <estefano.bargas@fing.edu.uy>\nCo-authored-by: Tomás <tomas.gruner@lambdaclass.com>\nCo-authored-by: Tomás <47506558+MegaRedHand@users.noreply.github.com>",
          "timestamp": "2023-05-30T17:41:04Z",
          "tree_id": "ab6fbb42e4c0110bdeea2208f947ab886c6858db",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/f6ffed48b1b96d55ff6d994195822c0e5b36acf9"
        },
        "date": 1685468802616,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 130098540,
            "range": "± 3309219",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 251694083,
            "range": "± 2968183",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 483093718,
            "range": "± 8068407",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 970608937,
            "range": "± 6512765",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 34261828,
            "range": "± 957463",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 66812522,
            "range": "± 164271",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 132622764,
            "range": "± 1145791",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 269387677,
            "range": "± 903740",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 28263531,
            "range": "± 579054",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 54719935,
            "range": "± 2751650",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 115147860,
            "range": "± 4311363",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 266345874,
            "range": "± 22053390",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 164079142,
            "range": "± 1596455",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 329493322,
            "range": "± 1480319",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 650504000,
            "range": "± 4874076",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1305697687,
            "range": "± 16203418",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 447179135,
            "range": "± 1542227",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 901440021,
            "range": "± 4317688",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1749579875,
            "range": "± 14298011",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3473994895,
            "range": "± 12999047",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "maigonzalez@fi.uba.ar",
            "name": "Matías Ignacio González",
            "username": "matias-gonz"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f6ffed48b1b96d55ff6d994195822c0e5b36acf9",
          "message": "Add gen_twiddles implementation in CUDA (#310)\n\n* Added cuda mod and poc\n\n* Finished poc\n\n* Working CUDA FFT POC\n\n* Added cuda ptx compilation make rule\n\n* Added CUDA u256 prime field\n\n* Added CI job for testing with CUDA\n\n* Add CUDAFieldElement\n\n* Added support for u256 montgomery field\n\n* Remove unwrap()s\n\n* Add evaluate_fft_cuda\n\n* Remove default feature cuda\n\n* Remove default feature cuda\n\n* Remove unnecessary reference\n\n* Fix clippy errors\n\nNOTE: we currently don't have a linting job in the CI for the _metal_ and _cuda_ features\n\n* Fix benches error\n\n* Rename `IsTwoAdicField` -> `IsFFTField`\n\n* Fix cannot find function error\n\n* Add TODO\n\n* Interpolate fft cuda (#300)\n\n* Add interpolate_fft_cuda\r\n\r\n* Fix RootsConfig\r\n\r\n* Remove unnecessary coefficients\r\n\r\n* Add not(feature = \"cuda\")\r\n\r\n* Add unwrap for interpolate_fft\n\n* Add error handling for CUDA's fft implementation (#301)\n\n* Move cuda/field to cuda/abstractions\r\n\r\nThis is to more closely mimic the metal dir structure\r\n\r\n* Move helpers from metal to crate root\r\n\r\n* Add `CudaError`\r\n\r\n* Move functions, remove errors\r\n\r\n* Add CudaError variants for fft\r\n\r\n* Add TODO\r\n\r\n* Remove default metal feature\r\n\r\n* Fix compile errors\r\n\r\n* Fix missing imports errors\r\n\r\n* Fix compile errors\r\n\r\n* Allow dead_code in helpers module\n\n* Remove unwrap from interpolate_fft\n\n* Add calc_twiddles* CUDA functions\n\n* Fix CUDA compile errors\n\n* Fix more CUDA compile errors\n\n* Recompile metallib\n\n* Separate twiddle functions into different files\n\n* Update makefile\n\n* Compile all `.cu` files in `CUDAPATH`\n\n* Compile all `.cu` files in `CUDAPATH`\n\n* Recombine all twiddles functions in the same file\n\nSomething's up with `Fp256::inverse`.\nUsing the function inside a kernel causes compilation to freeze.\n\n* Comment out `calc_twiddles*_inv` functions\n\nThese functions slow down compilation to a halt, and are equivalent to calling `calc_twiddles*` with an inverted `_omega`\n\n* Refactor get_twiddles so that it uses cuda\n\n* Fix PTX\n\n* Fix compile errors\n\n* Fix CUDAFelt from\n\n* Fix comparision chain\n\n* Add missing slice\n\n* Add missing type annotation\n\n* Change ? to unwrap\n\n* Remove reference\n\n* Change input param type\n\n* Update param types\n\n* Update calc_twiddles_bitrev\n\n* Change root\n\n* Load all functions\n\n* Change types to unsigned\n\n* Refactor reference\n\n* Update LaunchConfig\n\n* Update Block size\n\n* Use vec!\n\n* Update blockdim\n\n* Refactor twiddles\n\n* Add twiddles.ptx\n\n* Fix imports\n\n* Remove unwraps\n\n* Fix errors\n\n* Add map_err\n\n* Add missing map_err\n\n* Remove unused import\n\n* Update interpolate_fft\n\n* Integrate _CUDA_ implementation with _fft_ crate (#298)\n\n* Add evaluate_fft_cuda\r\n\r\n* Remove default feature cuda\r\n\r\n* Remove default feature cuda\r\n\r\n* Remove unnecessary reference\r\n\r\n* Fix clippy errors\r\n\r\nNOTE: we currently don't have a linting job in the CI for the _metal_ and _cuda_ features\r\n\r\n* Fix benches error\r\n\r\n* Fix cannot find function error\r\n\r\n* Add TODO\r\n\r\n* Interpolate fft cuda (#300)\r\n\r\n* Add interpolate_fft_cuda\r\n\r\n* Fix RootsConfig\r\n\r\n* Remove unnecessary coefficients\r\n\r\n* Add not(feature = \"cuda\")\r\n\r\n* Add unwrap for interpolate_fft\r\n\r\n* Add error handling for CUDA's fft implementation (#301)\r\n\r\n* Move cuda/field to cuda/abstractions\r\n\r\nThis is to more closely mimic the metal dir structure\r\n\r\n* Move helpers from metal to crate root\r\n\r\n* Add `CudaError`\r\n\r\n* Move functions, remove errors\r\n\r\n* Add CudaError variants for fft\r\n\r\n* Add TODO\r\n\r\n* Remove default metal feature\r\n\r\n* Fix compile errors\r\n\r\n* Fix missing imports errors\r\n\r\n* Fix compile errors\r\n\r\n* Allow dead_code in helpers module\r\n\r\n* Remove unwrap from interpolate_fft\r\n\r\n* Add `CudaState` as a wrapper around cudarc primitives (#311)\r\n\r\n* Add CudaState\r\n\r\n* Use CudaState in `fft` function\r\n\r\n* Remove old attributes\r\n\r\n* Remove `unwrap`s in Metal and Cuda state init\r\n\r\n* Extract library loading to helper function\r\n\r\n* Fix compilation errors and move LaunchConfig use\r\n\r\n* Remove unnecesary modulo operation\r\n\r\nThe `threadIdx.x` builtin variable goes from 0 to `blockDim.x` (non-inclusive) so we don't need the modulo.\r\n\r\n* Add bounds checking to launch\r\n\r\n* Fix compilation errors\r\n\r\n* Fix all compile errors\r\n\r\n* Recompile fft.ptx\r\n\r\n---------\r\n\r\nCo-authored-by: matias-gonz <maigonzalez@fi.uba.ar>\n\n* Fix compile error\n\n* Fix compilation errors\n\n* Use prop_assert_eq instead of assert_eq\n\n* Remove unused fp.cuh\n\n* Don't use `prop_filter` for `field_vec`\n\nThe use of `prop_filter` slows tests down, and can cause nondeterministic test failures when the filter function true/false ratio is too low.\nIn this case, using it would cause tests with a too high max exponent to fail.\n\n* Refactor index calculation in CUDA\n\n* Fix compile errors\n\n* Update evaluate fft cuda call\n\n* Update evaluate_fft_cpu call\n\n* Remove repeated code\n\n* Remove unused constant\n\n* Add missing cfg not cuda\n\n* Fix cfg\n\n* Refactor gen_twiddles using CudaState\n\n* Fix compile errors\n\n* Add state to function call\n\n* Update GenTwiddles.launch\n\n* Fix warnings\n\n* Remove cfg\n\n* Fix cfg\n\n* Update felt condition\n\n* Remove mut\n\n* Update error\n\n* Change library import\n\n* Add namespace\n\n* Update block dim\n\n* Add casts\n\n* Fix unsigned\n\n* Update makefile\n\n* Add turbofish to gen_twiddles\n\n* Add turbofish to get_calc_twiddles\n\n* Comment code\n\n* Update omega param\n\n* Update block dim\n\n* Add assert\n\n* Add casts\n\n* Update twiddles\n\n* Fix map\n\n* Add type annotation\n\n* Add turbo fish\n\n* Add type annotation\n\n* Fix turbofish\n\n* Fix turbofish type annotation\n\n* Add assert\n\n* Add guard clause\n\n* Implement gen_twiddles in stark256 file\n\n* Template twiddles\n\n* Fix Fp\n\n* Delete unused files\n\n* Update stark256.ptx\n\n---------\n\nCo-authored-by: Estéfano Bargas <estefano.bargas@fing.edu.uy>\nCo-authored-by: Tomás <tomas.gruner@lambdaclass.com>\nCo-authored-by: Tomás <47506558+MegaRedHand@users.noreply.github.com>",
          "timestamp": "2023-05-30T17:41:04Z",
          "tree_id": "ab6fbb42e4c0110bdeea2208f947ab886c6858db",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/f6ffed48b1b96d55ff6d994195822c0e5b36acf9"
        },
        "date": 1685470638272,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 1499796356,
            "range": "± 8555870",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 2448672553,
            "range": "± 21101437",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 3129276361,
            "range": "± 1193422",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 5520142408,
            "range": "± 15305800",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 6524163066,
            "range": "± 3176275",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 11990290305,
            "range": "± 44997619",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 13608311880,
            "range": "± 12204431",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 25404183309,
            "range": "± 178723180",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential",
            "value": 939702399,
            "range": "± 220987",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #2",
            "value": 1974730975,
            "range": "± 514729",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #3",
            "value": 4142537559,
            "range": "± 7514118",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #4",
            "value": 8656844018,
            "range": "± 1850522",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 75889551,
            "range": "± 891475",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 151733878,
            "range": "± 979707",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 302940663,
            "range": "± 1377375",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 611703901,
            "range": "± 2228874",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 2474656132,
            "range": "± 523690",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 5181991238,
            "range": "± 1640416",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 10826851918,
            "range": "± 8217582",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 22613549187,
            "range": "± 9142262",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 2651667237,
            "range": "± 6782636",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 5524374021,
            "range": "± 3359061",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 11505257827,
            "range": "± 2875018",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 23930970595,
            "range": "± 16780250",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 8,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 21,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 71,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 15,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 139,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 101,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 102,
            "range": "± 3",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "maigonzalez@fi.uba.ar",
            "name": "Matías Ignacio González",
            "username": "matias-gonz"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "af9e400fb9371c4ee1bfd6e9264e67ce1e4d9cdc",
          "message": "Add Bitrev permutation in CUDA (#312)\n\n* Added cuda mod and poc\n\n* Finished poc\n\n* Working CUDA FFT POC\n\n* Added cuda ptx compilation make rule\n\n* Added CUDA u256 prime field\n\n* Added CI job for testing with CUDA\n\n* Add CUDAFieldElement\n\n* Added support for u256 montgomery field\n\n* Remove unwrap()s\n\n* Add evaluate_fft_cuda\n\n* Remove default feature cuda\n\n* Remove default feature cuda\n\n* Remove unnecessary reference\n\n* Fix clippy errors\n\nNOTE: we currently don't have a linting job in the CI for the _metal_ and _cuda_ features\n\n* Fix benches error\n\n* Rename `IsTwoAdicField` -> `IsFFTField`\n\n* Fix cannot find function error\n\n* Add TODO\n\n* Interpolate fft cuda (#300)\n\n* Add interpolate_fft_cuda\r\n\r\n* Fix RootsConfig\r\n\r\n* Remove unnecessary coefficients\r\n\r\n* Add not(feature = \"cuda\")\r\n\r\n* Add unwrap for interpolate_fft\n\n* Add error handling for CUDA's fft implementation (#301)\n\n* Move cuda/field to cuda/abstractions\r\n\r\nThis is to more closely mimic the metal dir structure\r\n\r\n* Move helpers from metal to crate root\r\n\r\n* Add `CudaError`\r\n\r\n* Move functions, remove errors\r\n\r\n* Add CudaError variants for fft\r\n\r\n* Add TODO\r\n\r\n* Remove default metal feature\r\n\r\n* Fix compile errors\r\n\r\n* Fix missing imports errors\r\n\r\n* Fix compile errors\r\n\r\n* Allow dead_code in helpers module\n\n* Remove unwrap from interpolate_fft\n\n* Add calc_twiddles* CUDA functions\n\n* Fix CUDA compile errors\n\n* Fix more CUDA compile errors\n\n* Recompile metallib\n\n* Separate twiddle functions into different files\n\n* Update makefile\n\n* Compile all `.cu` files in `CUDAPATH`\n\n* Compile all `.cu` files in `CUDAPATH`\n\n* Recombine all twiddles functions in the same file\n\nSomething's up with `Fp256::inverse`.\nUsing the function inside a kernel causes compilation to freeze.\n\n* Comment out `calc_twiddles*_inv` functions\n\nThese functions slow down compilation to a halt, and are equivalent to calling `calc_twiddles*` with an inverted `_omega`\n\n* Refactor get_twiddles so that it uses cuda\n\n* Fix PTX\n\n* Fix compile errors\n\n* Fix CUDAFelt from\n\n* Fix comparision chain\n\n* Add missing slice\n\n* Add missing type annotation\n\n* Change ? to unwrap\n\n* Remove reference\n\n* Change input param type\n\n* Update param types\n\n* Update calc_twiddles_bitrev\n\n* Change root\n\n* Load all functions\n\n* Change types to unsigned\n\n* Refactor reference\n\n* Update LaunchConfig\n\n* Update Block size\n\n* Use vec!\n\n* Update blockdim\n\n* Refactor twiddles\n\n* Add twiddles.ptx\n\n* Fix imports\n\n* Remove unwraps\n\n* Fix errors\n\n* Add map_err\n\n* Add missing map_err\n\n* Remove unused import\n\n* Update interpolate_fft\n\n* Add bitrev_permutation in CUDA\n\n* Fix function signature\n\n* Add bitrev_permutation in cuda interface\n\n* Fix compilation errors\n\n* Change usize into u64\n\n* Add bitrev_permutation.ptx\n\n* Integrate _CUDA_ implementation with _fft_ crate (#298)\n\n* Add evaluate_fft_cuda\r\n\r\n* Remove default feature cuda\r\n\r\n* Remove default feature cuda\r\n\r\n* Remove unnecessary reference\r\n\r\n* Fix clippy errors\r\n\r\nNOTE: we currently don't have a linting job in the CI for the _metal_ and _cuda_ features\r\n\r\n* Fix benches error\r\n\r\n* Fix cannot find function error\r\n\r\n* Add TODO\r\n\r\n* Interpolate fft cuda (#300)\r\n\r\n* Add interpolate_fft_cuda\r\n\r\n* Fix RootsConfig\r\n\r\n* Remove unnecessary coefficients\r\n\r\n* Add not(feature = \"cuda\")\r\n\r\n* Add unwrap for interpolate_fft\r\n\r\n* Add error handling for CUDA's fft implementation (#301)\r\n\r\n* Move cuda/field to cuda/abstractions\r\n\r\nThis is to more closely mimic the metal dir structure\r\n\r\n* Move helpers from metal to crate root\r\n\r\n* Add `CudaError`\r\n\r\n* Move functions, remove errors\r\n\r\n* Add CudaError variants for fft\r\n\r\n* Add TODO\r\n\r\n* Remove default metal feature\r\n\r\n* Fix compile errors\r\n\r\n* Fix missing imports errors\r\n\r\n* Fix compile errors\r\n\r\n* Allow dead_code in helpers module\r\n\r\n* Remove unwrap from interpolate_fft\r\n\r\n* Add `CudaState` as a wrapper around cudarc primitives (#311)\r\n\r\n* Add CudaState\r\n\r\n* Use CudaState in `fft` function\r\n\r\n* Remove old attributes\r\n\r\n* Remove `unwrap`s in Metal and Cuda state init\r\n\r\n* Extract library loading to helper function\r\n\r\n* Fix compilation errors and move LaunchConfig use\r\n\r\n* Remove unnecesary modulo operation\r\n\r\nThe `threadIdx.x` builtin variable goes from 0 to `blockDim.x` (non-inclusive) so we don't need the modulo.\r\n\r\n* Add bounds checking to launch\r\n\r\n* Fix compilation errors\r\n\r\n* Fix all compile errors\r\n\r\n* Recompile fft.ptx\r\n\r\n---------\r\n\r\nCo-authored-by: matias-gonz <maigonzalez@fi.uba.ar>\n\n* Fix compile error\n\n* Fix compilation errors\n\n* Use prop_assert_eq instead of assert_eq\n\n* Remove unused fp.cuh\n\n* Don't use `prop_filter` for `field_vec`\n\nThe use of `prop_filter` slows tests down, and can cause nondeterministic test failures when the filter function true/false ratio is too low.\nIn this case, using it would cause tests with a too high max exponent to fail.\n\n* Refactor index calculation in CUDA\n\n* Fix compile errors\n\n* Update evaluate fft cuda call\n\n* Update evaluate_fft_cpu call\n\n* Remove repeated code\n\n* Remove unused constant\n\n* Add missing cfg not cuda\n\n* Fix cfg\n\n* Refactor gen_twiddles using CudaState\n\n* Fix compile errors\n\n* Add state to function call\n\n* Update GenTwiddles.launch\n\n* Fix warnings\n\n* Remove cfg\n\n* Refactor bitrev_permutation using CudaState\n\n* Fix get_bitrev_permutation\n\n* Fix block_dim\n\n* Fix group size\n\n* Fix get_bitrev_permutation call\n\n* Remove unused state variable\n\n* Fix warnings\n\n* Add bitrev_permutation\n\n* Update makefile\n\n* Update make build-cuda\n\n* Upodate build-cuda\n\n* Update build-cuda\n\n* Template bitrev_permutation\n\n* Update stark256.ptx\n\n* Delete unused clone and add explaining comment\n\n---------\n\nCo-authored-by: Estéfano Bargas <estefano.bargas@fing.edu.uy>\nCo-authored-by: Tomás <tomas.gruner@lambdaclass.com>\nCo-authored-by: Tomás <47506558+MegaRedHand@users.noreply.github.com>",
          "timestamp": "2023-05-30T18:34:02Z",
          "tree_id": "d5d2dae9d2c8749bb6b6968f6bd64bd41b9e7a58",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/af9e400fb9371c4ee1bfd6e9264e67ce1e4d9cdc"
        },
        "date": 1685471983066,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 130904441,
            "range": "± 2376782",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 255556708,
            "range": "± 2353656",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 493369833,
            "range": "± 1167629",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 969144354,
            "range": "± 7066289",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 34239009,
            "range": "± 270801",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 68171980,
            "range": "± 755465",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 132849912,
            "range": "± 772739",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 276344656,
            "range": "± 3569298",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 31048995,
            "range": "± 320332",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 57916308,
            "range": "± 3691493",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 118081941,
            "range": "± 8888764",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 251505041,
            "range": "± 22174590",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 163692049,
            "range": "± 865290",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 330141041,
            "range": "± 2023644",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 650855000,
            "range": "± 6059685",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1296896729,
            "range": "± 17930924",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 452206760,
            "range": "± 1551733",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 896349812,
            "range": "± 4065737",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1757488104,
            "range": "± 12955481",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3473887708,
            "range": "± 30568313",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "maigonzalez@fi.uba.ar",
            "name": "Matías Ignacio González",
            "username": "matias-gonz"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "af9e400fb9371c4ee1bfd6e9264e67ce1e4d9cdc",
          "message": "Add Bitrev permutation in CUDA (#312)\n\n* Added cuda mod and poc\n\n* Finished poc\n\n* Working CUDA FFT POC\n\n* Added cuda ptx compilation make rule\n\n* Added CUDA u256 prime field\n\n* Added CI job for testing with CUDA\n\n* Add CUDAFieldElement\n\n* Added support for u256 montgomery field\n\n* Remove unwrap()s\n\n* Add evaluate_fft_cuda\n\n* Remove default feature cuda\n\n* Remove default feature cuda\n\n* Remove unnecessary reference\n\n* Fix clippy errors\n\nNOTE: we currently don't have a linting job in the CI for the _metal_ and _cuda_ features\n\n* Fix benches error\n\n* Rename `IsTwoAdicField` -> `IsFFTField`\n\n* Fix cannot find function error\n\n* Add TODO\n\n* Interpolate fft cuda (#300)\n\n* Add interpolate_fft_cuda\r\n\r\n* Fix RootsConfig\r\n\r\n* Remove unnecessary coefficients\r\n\r\n* Add not(feature = \"cuda\")\r\n\r\n* Add unwrap for interpolate_fft\n\n* Add error handling for CUDA's fft implementation (#301)\n\n* Move cuda/field to cuda/abstractions\r\n\r\nThis is to more closely mimic the metal dir structure\r\n\r\n* Move helpers from metal to crate root\r\n\r\n* Add `CudaError`\r\n\r\n* Move functions, remove errors\r\n\r\n* Add CudaError variants for fft\r\n\r\n* Add TODO\r\n\r\n* Remove default metal feature\r\n\r\n* Fix compile errors\r\n\r\n* Fix missing imports errors\r\n\r\n* Fix compile errors\r\n\r\n* Allow dead_code in helpers module\n\n* Remove unwrap from interpolate_fft\n\n* Add calc_twiddles* CUDA functions\n\n* Fix CUDA compile errors\n\n* Fix more CUDA compile errors\n\n* Recompile metallib\n\n* Separate twiddle functions into different files\n\n* Update makefile\n\n* Compile all `.cu` files in `CUDAPATH`\n\n* Compile all `.cu` files in `CUDAPATH`\n\n* Recombine all twiddles functions in the same file\n\nSomething's up with `Fp256::inverse`.\nUsing the function inside a kernel causes compilation to freeze.\n\n* Comment out `calc_twiddles*_inv` functions\n\nThese functions slow down compilation to a halt, and are equivalent to calling `calc_twiddles*` with an inverted `_omega`\n\n* Refactor get_twiddles so that it uses cuda\n\n* Fix PTX\n\n* Fix compile errors\n\n* Fix CUDAFelt from\n\n* Fix comparision chain\n\n* Add missing slice\n\n* Add missing type annotation\n\n* Change ? to unwrap\n\n* Remove reference\n\n* Change input param type\n\n* Update param types\n\n* Update calc_twiddles_bitrev\n\n* Change root\n\n* Load all functions\n\n* Change types to unsigned\n\n* Refactor reference\n\n* Update LaunchConfig\n\n* Update Block size\n\n* Use vec!\n\n* Update blockdim\n\n* Refactor twiddles\n\n* Add twiddles.ptx\n\n* Fix imports\n\n* Remove unwraps\n\n* Fix errors\n\n* Add map_err\n\n* Add missing map_err\n\n* Remove unused import\n\n* Update interpolate_fft\n\n* Add bitrev_permutation in CUDA\n\n* Fix function signature\n\n* Add bitrev_permutation in cuda interface\n\n* Fix compilation errors\n\n* Change usize into u64\n\n* Add bitrev_permutation.ptx\n\n* Integrate _CUDA_ implementation with _fft_ crate (#298)\n\n* Add evaluate_fft_cuda\r\n\r\n* Remove default feature cuda\r\n\r\n* Remove default feature cuda\r\n\r\n* Remove unnecessary reference\r\n\r\n* Fix clippy errors\r\n\r\nNOTE: we currently don't have a linting job in the CI for the _metal_ and _cuda_ features\r\n\r\n* Fix benches error\r\n\r\n* Fix cannot find function error\r\n\r\n* Add TODO\r\n\r\n* Interpolate fft cuda (#300)\r\n\r\n* Add interpolate_fft_cuda\r\n\r\n* Fix RootsConfig\r\n\r\n* Remove unnecessary coefficients\r\n\r\n* Add not(feature = \"cuda\")\r\n\r\n* Add unwrap for interpolate_fft\r\n\r\n* Add error handling for CUDA's fft implementation (#301)\r\n\r\n* Move cuda/field to cuda/abstractions\r\n\r\nThis is to more closely mimic the metal dir structure\r\n\r\n* Move helpers from metal to crate root\r\n\r\n* Add `CudaError`\r\n\r\n* Move functions, remove errors\r\n\r\n* Add CudaError variants for fft\r\n\r\n* Add TODO\r\n\r\n* Remove default metal feature\r\n\r\n* Fix compile errors\r\n\r\n* Fix missing imports errors\r\n\r\n* Fix compile errors\r\n\r\n* Allow dead_code in helpers module\r\n\r\n* Remove unwrap from interpolate_fft\r\n\r\n* Add `CudaState` as a wrapper around cudarc primitives (#311)\r\n\r\n* Add CudaState\r\n\r\n* Use CudaState in `fft` function\r\n\r\n* Remove old attributes\r\n\r\n* Remove `unwrap`s in Metal and Cuda state init\r\n\r\n* Extract library loading to helper function\r\n\r\n* Fix compilation errors and move LaunchConfig use\r\n\r\n* Remove unnecesary modulo operation\r\n\r\nThe `threadIdx.x` builtin variable goes from 0 to `blockDim.x` (non-inclusive) so we don't need the modulo.\r\n\r\n* Add bounds checking to launch\r\n\r\n* Fix compilation errors\r\n\r\n* Fix all compile errors\r\n\r\n* Recompile fft.ptx\r\n\r\n---------\r\n\r\nCo-authored-by: matias-gonz <maigonzalez@fi.uba.ar>\n\n* Fix compile error\n\n* Fix compilation errors\n\n* Use prop_assert_eq instead of assert_eq\n\n* Remove unused fp.cuh\n\n* Don't use `prop_filter` for `field_vec`\n\nThe use of `prop_filter` slows tests down, and can cause nondeterministic test failures when the filter function true/false ratio is too low.\nIn this case, using it would cause tests with a too high max exponent to fail.\n\n* Refactor index calculation in CUDA\n\n* Fix compile errors\n\n* Update evaluate fft cuda call\n\n* Update evaluate_fft_cpu call\n\n* Remove repeated code\n\n* Remove unused constant\n\n* Add missing cfg not cuda\n\n* Fix cfg\n\n* Refactor gen_twiddles using CudaState\n\n* Fix compile errors\n\n* Add state to function call\n\n* Update GenTwiddles.launch\n\n* Fix warnings\n\n* Remove cfg\n\n* Refactor bitrev_permutation using CudaState\n\n* Fix get_bitrev_permutation\n\n* Fix block_dim\n\n* Fix group size\n\n* Fix get_bitrev_permutation call\n\n* Remove unused state variable\n\n* Fix warnings\n\n* Add bitrev_permutation\n\n* Update makefile\n\n* Update make build-cuda\n\n* Upodate build-cuda\n\n* Update build-cuda\n\n* Template bitrev_permutation\n\n* Update stark256.ptx\n\n* Delete unused clone and add explaining comment\n\n---------\n\nCo-authored-by: Estéfano Bargas <estefano.bargas@fing.edu.uy>\nCo-authored-by: Tomás <tomas.gruner@lambdaclass.com>\nCo-authored-by: Tomás <47506558+MegaRedHand@users.noreply.github.com>",
          "timestamp": "2023-05-30T18:34:02Z",
          "tree_id": "d5d2dae9d2c8749bb6b6968f6bd64bd41b9e7a58",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/af9e400fb9371c4ee1bfd6e9264e67ce1e4d9cdc"
        },
        "date": 1685474364297,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 1757054123,
            "range": "± 15934451",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 4186012114,
            "range": "± 26274778",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 3701467969,
            "range": "± 49710699",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 8946788815,
            "range": "± 70886137",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 7608840521,
            "range": "± 57902546",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 19350450169,
            "range": "± 349721964",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 16127841431,
            "range": "± 125480447",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 41119879333,
            "range": "± 345696737",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential",
            "value": 1039376574,
            "range": "± 9914721",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #2",
            "value": 2184371180,
            "range": "± 11262465",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #3",
            "value": 4591960121,
            "range": "± 20808746",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #4",
            "value": 9637833495,
            "range": "± 80680488",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 98911888,
            "range": "± 3034566",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 196971872,
            "range": "± 2646822",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 386541674,
            "range": "± 6737143",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 789644322,
            "range": "± 20088458",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 2867958166,
            "range": "± 27840711",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 5954361087,
            "range": "± 70422978",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 12474418383,
            "range": "± 101012899",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 25907457229,
            "range": "± 149763675",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 3139462324,
            "range": "± 98282193",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 6353874027,
            "range": "± 108373902",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 13208002891,
            "range": "± 138678430",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 27730229196,
            "range": "± 421431058",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 868,
            "range": "± 30",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 27965,
            "range": "± 1681",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 786,
            "range": "± 40",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 22,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 1131,
            "range": "± 46",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 933,
            "range": "± 40",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 776,
            "range": "± 55",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mario.rugiero@nextroll.com",
            "name": "Mario Rugiero",
            "username": "Oppen"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "9378d38acc81c17d9c5a03e901bd8fbb1da21a3f",
          "message": "perf: weaken power in sqrt (#397)\n\n* bench: benchmarks for field sqrt\n\n* perf: weaken power in sqrt\n\n* perf: use primitive type for index",
          "timestamp": "2023-05-30T20:57:06Z",
          "tree_id": "81433da823002851ac453b6ecf98bc8c7b2bd04f",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/9378d38acc81c17d9c5a03e901bd8fbb1da21a3f"
        },
        "date": 1685480573909,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 130302132,
            "range": "± 2368549",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 256181833,
            "range": "± 667932",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 494253572,
            "range": "± 2461532",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 971375833,
            "range": "± 10945507",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 35083156,
            "range": "± 388876",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 68997423,
            "range": "± 962315",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 136115795,
            "range": "± 2042192",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 279696406,
            "range": "± 4338988",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 31232671,
            "range": "± 241964",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 60009140,
            "range": "± 605063",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 121749115,
            "range": "± 3204515",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 258818784,
            "range": "± 18190489",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 163977836,
            "range": "± 645174",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 331078812,
            "range": "± 963521",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 653363625,
            "range": "± 3859337",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1309901562,
            "range": "± 8034378",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 447042698,
            "range": "± 1409411",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 898188687,
            "range": "± 8291405",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1765620479,
            "range": "± 4886444",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3474232292,
            "range": "± 17826404",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mario.rugiero@nextroll.com",
            "name": "Mario Rugiero",
            "username": "Oppen"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "65b3780abaf7c61d0fc8ba8ea83d35788e69ebad",
          "message": "fix: Pippenger substraction overflow (#376)",
          "timestamp": "2023-05-30T21:11:22Z",
          "tree_id": "30e333bf674959ec4ef55bef0a483a52200956de",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/65b3780abaf7c61d0fc8ba8ea83d35788e69ebad"
        },
        "date": 1685481424259,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 129012752,
            "range": "± 2575695",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 255811822,
            "range": "± 1698687",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 488277489,
            "range": "± 5168413",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 978042187,
            "range": "± 11745231",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 34338684,
            "range": "± 402931",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 67953521,
            "range": "± 640262",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 132988525,
            "range": "± 747120",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 277923041,
            "range": "± 2853095",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 31215038,
            "range": "± 318037",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 58277373,
            "range": "± 1841538",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 122130811,
            "range": "± 4457343",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 241124770,
            "range": "± 23444388",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 162979561,
            "range": "± 853066",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 329875833,
            "range": "± 2570536",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 649700437,
            "range": "± 5819351",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1310380583,
            "range": "± 13777842",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 452259333,
            "range": "± 2469609",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 898374604,
            "range": "± 5666748",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1756034104,
            "range": "± 15605378",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3640858520,
            "range": "± 290336832",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mario.rugiero@nextroll.com",
            "name": "Mario Rugiero",
            "username": "Oppen"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "31403d4ae6df411a5a32c84f378816716e7e9805",
          "message": "bench: benchmark all gen twiddles configs (#373)",
          "timestamp": "2023-05-30T21:15:49Z",
          "tree_id": "444ddd6f3d33e0d30dfb569a7ea01e5b7352ffe0",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/31403d4ae6df411a5a32c84f378816716e7e9805"
        },
        "date": 1685481681099,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 178254823,
            "range": "± 193604630",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 281264271,
            "range": "± 425737931",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 493502198,
            "range": "± 2509393",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 964256687,
            "range": "± 12767533",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 34521088,
            "range": "± 372410",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 68146915,
            "range": "± 498939",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 133122083,
            "range": "± 908138",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 274416333,
            "range": "± 3198657",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 31105772,
            "range": "± 159459",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 59893867,
            "range": "± 2108056",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 120948370,
            "range": "± 4225593",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 268875010,
            "range": "± 24070018",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 164650421,
            "range": "± 1004733",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 330647458,
            "range": "± 2240034",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 646934375,
            "range": "± 5440547",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1306593146,
            "range": "± 10955771",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 443431291,
            "range": "± 7204455",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 900374791,
            "range": "± 6221261",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1764544000,
            "range": "± 14753788",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3450165625,
            "range": "± 18196811",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mario.rugiero@nextroll.com",
            "name": "Mario Rugiero",
            "username": "Oppen"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f49c67abe35fd39403b0a4be978f61f771f26b77",
          "message": "perf: weaken power in twiddles generation (#375)\n\n* bench: benchmark all gen twiddles configs\n\n* perf: weaken power in twiddles generation\n\nKeep a running product of roots rather than computing the power each\ntime.\nReplace the batch inversion by eager inversion of the primitive root.\n\nThese two changes give a 25% throughput boost in FFT and 2x in\npolynomial evaluation and interpolation.\nDetailed benchmarks in PR #375 description.\n\n* chore: document the optimization a bit",
          "timestamp": "2023-05-30T21:26:25Z",
          "tree_id": "fcbb71147032ce694d78772ab95443afd1125818",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/f49c67abe35fd39403b0a4be978f61f771f26b77"
        },
        "date": 1685482320600,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 130435241,
            "range": "± 1648758",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 255463604,
            "range": "± 2131082",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 485751562,
            "range": "± 3659960",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 973115479,
            "range": "± 9046579",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 35252901,
            "range": "± 403211",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 68724389,
            "range": "± 318027",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 132980500,
            "range": "± 495228",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 277344666,
            "range": "± 3491735",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 31276454,
            "range": "± 261892",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 56084366,
            "range": "± 2782406",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 122322850,
            "range": "± 8706994",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 231302802,
            "range": "± 13824903",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 163151290,
            "range": "± 852275",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 330866625,
            "range": "± 1211960",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 651544854,
            "range": "± 5928788",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1308229437,
            "range": "± 19859917",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 448632979,
            "range": "± 2046269",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 893291042,
            "range": "± 7273612",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1740824687,
            "range": "± 9629590",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3440714354,
            "range": "± 20065930",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mario.rugiero@nextroll.com",
            "name": "Mario Rugiero",
            "username": "Oppen"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "9378d38acc81c17d9c5a03e901bd8fbb1da21a3f",
          "message": "perf: weaken power in sqrt (#397)\n\n* bench: benchmarks for field sqrt\n\n* perf: weaken power in sqrt\n\n* perf: use primitive type for index",
          "timestamp": "2023-05-30T20:57:06Z",
          "tree_id": "81433da823002851ac453b6ecf98bc8c7b2bd04f",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/9378d38acc81c17d9c5a03e901bd8fbb1da21a3f"
        },
        "date": 1685482474816,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 1439360853,
            "range": "± 736629",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 3296976324,
            "range": "± 15853701",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 3004469562,
            "range": "± 2010400",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 7376449615,
            "range": "± 8860140",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 6277735644,
            "range": "± 3465593",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 15886085805,
            "range": "± 48310236",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 13069778216,
            "range": "± 8087155",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 33866678122,
            "range": "± 81447673",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential",
            "value": 850030029,
            "range": "± 255799",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #2",
            "value": 1785802444,
            "range": "± 443697",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #3",
            "value": 3744595658,
            "range": "± 5237979",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #4",
            "value": 7834015683,
            "range": "± 5475164",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 78798649,
            "range": "± 990070",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 159056461,
            "range": "± 526076",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 314914822,
            "range": "± 581495",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 642012744,
            "range": "± 3325092",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 2318321184,
            "range": "± 1483461",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 4850240393,
            "range": "± 2325200",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 10130955324,
            "range": "± 5604405",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 21113592388,
            "range": "± 7702540",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 2509133563,
            "range": "± 1932748",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 5234756478,
            "range": "± 14404869",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 10884139777,
            "range": "± 4138044",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 22637923955,
            "range": "± 8174049",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 1540,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 98747,
            "range": "± 38",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 1145,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 23,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 1591,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 1225,
            "range": "± 7",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 1214,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mario.rugiero@nextroll.com",
            "name": "Mario Rugiero",
            "username": "Oppen"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "65b3780abaf7c61d0fc8ba8ea83d35788e69ebad",
          "message": "fix: Pippenger substraction overflow (#376)",
          "timestamp": "2023-05-30T21:11:22Z",
          "tree_id": "30e333bf674959ec4ef55bef0a483a52200956de",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/65b3780abaf7c61d0fc8ba8ea83d35788e69ebad"
        },
        "date": 1685483331880,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 1438139682,
            "range": "± 1259342",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 3347487921,
            "range": "± 8607068",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 3008013778,
            "range": "± 1755482",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 7339124378,
            "range": "± 18271013",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 6271618239,
            "range": "± 4588012",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 15811281734,
            "range": "± 19698641",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 13076195294,
            "range": "± 7299904",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 33562904681,
            "range": "± 54567777",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential",
            "value": 857003192,
            "range": "± 240923",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #2",
            "value": 1800604512,
            "range": "± 258077",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #3",
            "value": 3776456551,
            "range": "± 926439",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Sequential #4",
            "value": 7901372745,
            "range": "± 2037870",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 71222691,
            "range": "± 1165295",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 154476910,
            "range": "± 942217",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 317373411,
            "range": "± 1421088",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 646953527,
            "range": "± 2092418",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 2322916161,
            "range": "± 1515945",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 4865941723,
            "range": "± 2948861",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 10163092966,
            "range": "± 5420899",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 21181556416,
            "range": "± 4875461",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 2497082289,
            "range": "± 1822114",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 5208388717,
            "range": "± 1920738",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 10842917312,
            "range": "± 4694531",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 22546472934,
            "range": "± 8163256",
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
            "value": 35,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 85,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 17,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 157,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 122,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 122,
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
          "id": "d1b853cea4b24c74fb47beff815d2814308a976d",
          "message": "Optimize invert (#396)\n\n* Initial version\n\n* Optimize invert!\n\n* Update\n\n* Invert\n\n* Optimize invert\n\n* Delete launch.json\n\n* fix inv\n\n* remove comment\n\n* remove div2 and add shr_inplace\n\n* fmt\n\n* fix test\n\n* rustify shl_inplace\n\n* remove duplicated test\n\n* fmt\n\n* move implementation of shl_inplace to ShlAssign trait impl\n\n* fmt\n\n---------\n\nCo-authored-by: juanbono <juanbono94@gmail.com>\nCo-authored-by: Sergio Chouhy <sergio.chouhy@gmail.com>",
          "timestamp": "2023-05-30T21:53:24Z",
          "tree_id": "8d3a266e86f8bae755dd04fadfe70c6e54c93558",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/d1b853cea4b24c74fb47beff815d2814308a976d"
        },
        "date": 1685483939159,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 128823375,
            "range": "± 2143700",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 254602968,
            "range": "± 2248357",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 487245906,
            "range": "± 4626894",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 977672854,
            "range": "± 4867631",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 34056288,
            "range": "± 346487",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 68346447,
            "range": "± 804284",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 133071844,
            "range": "± 382673",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 279689281,
            "range": "± 2891371",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 31320597,
            "range": "± 341480",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 57035363,
            "range": "± 2902618",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 120608432,
            "range": "± 13603505",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 266525312,
            "range": "± 20608637",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 163213836,
            "range": "± 985762",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 329862239,
            "range": "± 1755807",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 650604145,
            "range": "± 3840462",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1302141958,
            "range": "± 13445681",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 448228573,
            "range": "± 2874592",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 894275041,
            "range": "± 5312732",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1767001229,
            "range": "± 7455913",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3440634791,
            "range": "± 27809048",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mario.rugiero@nextroll.com",
            "name": "Mario Rugiero",
            "username": "Oppen"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f49c67abe35fd39403b0a4be978f61f771f26b77",
          "message": "perf: weaken power in twiddles generation (#375)\n\n* bench: benchmark all gen twiddles configs\n\n* perf: weaken power in twiddles generation\n\nKeep a running product of roots rather than computing the power each\ntime.\nReplace the batch inversion by eager inversion of the primitive root.\n\nThese two changes give a 25% throughput boost in FFT and 2x in\npolynomial evaluation and interpolation.\nDetailed benchmarks in PR #375 description.\n\n* chore: document the optimization a bit",
          "timestamp": "2023-05-30T21:26:25Z",
          "tree_id": "fcbb71147032ce694d78772ab95443afd1125818",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/f49c67abe35fd39403b0a4be978f61f771f26b77"
        },
        "date": 1685484028170,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 1438198286,
            "range": "± 1609825",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 3706165280,
            "range": "± 23560279",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 3002333411,
            "range": "± 2380909",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 8096081105,
            "range": "± 20455779",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 6254589048,
            "range": "± 3696902",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 17502497170,
            "range": "± 104963632",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 13030878103,
            "range": "± 15290840",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 37187809978,
            "range": "± 173889039",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 37521868,
            "range": "± 196981",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 37400929,
            "range": "± 111908",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 82750425,
            "range": "± 395079",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 82876475,
            "range": "± 267025",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 75257389,
            "range": "± 173991",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 75495759,
            "range": "± 126686",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 165463505,
            "range": "± 295958",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 165975654,
            "range": "± 354007",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 151010469,
            "range": "± 72270",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 150901569,
            "range": "± 181833",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 337254667,
            "range": "± 986867",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 337630983,
            "range": "± 1217893",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 300863704,
            "range": "± 155323",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 300717932,
            "range": "± 237198",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 680297656,
            "range": "± 2624892",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 678344874,
            "range": "± 1789992",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 90032660,
            "range": "± 503440",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 182104587,
            "range": "± 529216",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 367439824,
            "range": "± 616677",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 749923439,
            "range": "± 3426228",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 1549107455,
            "range": "± 2086014",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 3227395918,
            "range": "± 2986141",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 6710063949,
            "range": "± 4737314",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 13939694797,
            "range": "± 13374184",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 1632839801,
            "range": "± 2565420",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 3392496723,
            "range": "± 2307762",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 7036744959,
            "range": "± 2987371",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 14582913496,
            "range": "± 9757880",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 41,
            "range": "± 0",
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
            "value": 99,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 18,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 180,
            "range": "± 7",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 137,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 137,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mario.rugiero@nextroll.com",
            "name": "Mario Rugiero",
            "username": "Oppen"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "31403d4ae6df411a5a32c84f378816716e7e9805",
          "message": "bench: benchmark all gen twiddles configs (#373)",
          "timestamp": "2023-05-30T21:15:49Z",
          "tree_id": "444ddd6f3d33e0d30dfb569a7ea01e5b7352ffe0",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/31403d4ae6df411a5a32c84f378816716e7e9805"
        },
        "date": 1685484583823,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 1677116084,
            "range": "± 17831195",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 3896616518,
            "range": "± 15032830",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 3472391833,
            "range": "± 14665109",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 8496446526,
            "range": "± 10241600",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 7319501265,
            "range": "± 38657916",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 18347258777,
            "range": "± 105191131",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 15229737653,
            "range": "± 26381218",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 39362439242,
            "range": "± 134975003",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 1032527396,
            "range": "± 3691439",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 1146734638,
            "range": "± 7182679",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 1029830905,
            "range": "± 7164051",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 1146174096,
            "range": "± 4874342",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 2166807221,
            "range": "± 8409103",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 2391150776,
            "range": "± 5144752",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 2158984601,
            "range": "± 8923149",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 2388067521,
            "range": "± 8328314",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 4529251561,
            "range": "± 12914747",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 4981619197,
            "range": "± 12896127",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 4529528981,
            "range": "± 20204272",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 4983522598,
            "range": "± 16774932",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 9450647528,
            "range": "± 17974500",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 10396667311,
            "range": "± 33695396",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 9470476717,
            "range": "± 31667985",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 10372419376,
            "range": "± 20867542",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 88045746,
            "range": "± 565036",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 175752852,
            "range": "± 540340",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 344448957,
            "range": "± 1181225",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 693519044,
            "range": "± 2949885",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 2733252014,
            "range": "± 6191311",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 5673388371,
            "range": "± 23642987",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 11859400255,
            "range": "± 35729537",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 24512979300,
            "range": "± 163648743",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 2939748339,
            "range": "± 29042995",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 6047396477,
            "range": "± 44400726",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 12690075125,
            "range": "± 85283012",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 26629544678,
            "range": "± 125812282",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 215,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 1699,
            "range": "± 20",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 268,
            "range": "± 5",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 20,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 500,
            "range": "± 12",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 314,
            "range": "± 5",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 318,
            "range": "± 12",
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
          "id": "d1b853cea4b24c74fb47beff815d2814308a976d",
          "message": "Optimize invert (#396)\n\n* Initial version\n\n* Optimize invert!\n\n* Update\n\n* Invert\n\n* Optimize invert\n\n* Delete launch.json\n\n* fix inv\n\n* remove comment\n\n* remove div2 and add shr_inplace\n\n* fmt\n\n* fix test\n\n* rustify shl_inplace\n\n* remove duplicated test\n\n* fmt\n\n* move implementation of shl_inplace to ShlAssign trait impl\n\n* fmt\n\n---------\n\nCo-authored-by: juanbono <juanbono94@gmail.com>\nCo-authored-by: Sergio Chouhy <sergio.chouhy@gmail.com>",
          "timestamp": "2023-05-30T21:53:24Z",
          "tree_id": "8d3a266e86f8bae755dd04fadfe70c6e54c93558",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/d1b853cea4b24c74fb47beff815d2814308a976d"
        },
        "date": 1685485548441,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 1438399975,
            "range": "± 3348944",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 3396328100,
            "range": "± 25764476",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 3003077443,
            "range": "± 2815501",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 7361939887,
            "range": "± 39016441",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 6261173915,
            "range": "± 3877485",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 15856502634,
            "range": "± 57038742",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 13041833325,
            "range": "± 10256249",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 33829749924,
            "range": "± 62115074",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 37293048,
            "range": "± 75126",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 37352651,
            "range": "± 176361",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 78103215,
            "range": "± 323380",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 78320472,
            "range": "± 336155",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 74770191,
            "range": "± 228448",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 74874169,
            "range": "± 142031",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 157352294,
            "range": "± 1530559",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 158086483,
            "range": "± 1424467",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 149854155,
            "range": "± 233361",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 149683101,
            "range": "± 240668",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 317971269,
            "range": "± 2070054",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 316496747,
            "range": "± 601798",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 298078542,
            "range": "± 351777",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 298126919,
            "range": "± 240415",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 633612120,
            "range": "± 1518874",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 634178071,
            "range": "± 5730210",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 78086521,
            "range": "± 1739813",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 159882668,
            "range": "± 702583",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 321028822,
            "range": "± 933530",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 657088150,
            "range": "± 3828596",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 1542706092,
            "range": "± 1618402",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 3216804658,
            "range": "± 3102332",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 6692871794,
            "range": "± 8512935",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 13877699319,
            "range": "± 24349112",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 1623898220,
            "range": "± 6284702",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 3378854956,
            "range": "± 5004845",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 7008068679,
            "range": "± 7775256",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 14528364435,
            "range": "± 15096328",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 84,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 348,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 122,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 17,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 205,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 159,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 159,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mario.rugiero@nextroll.com",
            "name": "Mario Rugiero",
            "username": "Oppen"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "46dd588e9263a2f36935a2c41f28dc2e3352edc4",
          "message": "perf: compute inverse of 2 only once in sqrt (#399)",
          "timestamp": "2023-05-31T20:41:18Z",
          "tree_id": "840741905a8248fcdff4a6126abc8308dd7b247b",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/46dd588e9263a2f36935a2c41f28dc2e3352edc4"
        },
        "date": 1685566032416,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 129219259,
            "range": "± 3433374",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 256687270,
            "range": "± 1595102",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 486372479,
            "range": "± 4783006",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 979011145,
            "range": "± 6664970",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 34062043,
            "range": "± 277612",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 67955521,
            "range": "± 722692",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 133724220,
            "range": "± 2506489",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 276255208,
            "range": "± 2312316",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 31148116,
            "range": "± 108181",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 59033163,
            "range": "± 530936",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 121233901,
            "range": "± 2958483",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 262802801,
            "range": "± 24816566",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 163747206,
            "range": "± 894645",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 330160791,
            "range": "± 739442",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 651557187,
            "range": "± 5730550",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1307901791,
            "range": "± 11887856",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 451625062,
            "range": "± 2768955",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 897664479,
            "range": "± 5263826",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1738798458,
            "range": "± 8190242",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3470673542,
            "range": "± 31908810",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mario.rugiero@nextroll.com",
            "name": "Mario Rugiero",
            "username": "Oppen"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "46dd588e9263a2f36935a2c41f28dc2e3352edc4",
          "message": "perf: compute inverse of 2 only once in sqrt (#399)",
          "timestamp": "2023-05-31T20:41:18Z",
          "tree_id": "840741905a8248fcdff4a6126abc8308dd7b247b",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/46dd588e9263a2f36935a2c41f28dc2e3352edc4"
        },
        "date": 1685568025275,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 1669127993,
            "range": "± 16051848",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 4454461572,
            "range": "± 51666667",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 3521801458,
            "range": "± 40734163",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 9846331931,
            "range": "± 49739479",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 7172282336,
            "range": "± 104414666",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 21088016737,
            "range": "± 129757383",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 15335119260,
            "range": "± 172030754",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 44974943780,
            "range": "± 190966352",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 38515962,
            "range": "± 1294681",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 38520778,
            "range": "± 1312113",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 84080642,
            "range": "± 2355782",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 84627381,
            "range": "± 1582861",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 82520521,
            "range": "± 1411831",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 84058709,
            "range": "± 3318443",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 175242377,
            "range": "± 1623707",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 175499716,
            "range": "± 3853033",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 172599772,
            "range": "± 1782273",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 172421247,
            "range": "± 4466044",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 353498342,
            "range": "± 4656813",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 354099654,
            "range": "± 4594915",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 352463205,
            "range": "± 6332244",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 348429274,
            "range": "± 5150266",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 703525486,
            "range": "± 8383273",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 716492008,
            "range": "± 16177541",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 97660151,
            "range": "± 1840589",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 201055325,
            "range": "± 2495207",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 424127363,
            "range": "± 8628195",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 852495241,
            "range": "± 16409442",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 1830127611,
            "range": "± 26911084",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 3830819934,
            "range": "± 32742110",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 7866349303,
            "range": "± 250508652",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 15813750262,
            "range": "± 587212700",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 1755040157,
            "range": "± 27068344",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 3868703840,
            "range": "± 35895754",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 8048228177,
            "range": "± 120385881",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 16515682053,
            "range": "± 689758225",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 896,
            "range": "± 36",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 28629,
            "range": "± 1077",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 805,
            "range": "± 26",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 23,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 1182,
            "range": "± 40",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 973,
            "range": "± 44",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 864,
            "range": "± 63",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mario.rugiero@nextroll.com",
            "name": "Mario Rugiero",
            "username": "Oppen"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a02b3167d3a8d25d00ea09bde27837e08554eefb",
          "message": "perf: specialize is_neutral_element for each curve (#403)",
          "timestamp": "2023-06-02T13:01:19Z",
          "tree_id": "da6d217a92dd504c9b736d78d2a2e3586b61c7b7",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/a02b3167d3a8d25d00ea09bde27837e08554eefb"
        },
        "date": 1685711226243,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 130048143,
            "range": "± 1853471",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 256278073,
            "range": "± 1514504",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 488830458,
            "range": "± 4635592",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 967024729,
            "range": "± 13638472",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 34162661,
            "range": "± 207127",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 67778622,
            "range": "± 838967",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 133192126,
            "range": "± 566432",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 275522593,
            "range": "± 3294141",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 27603308,
            "range": "± 792974",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 59116882,
            "range": "± 1952266",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 118278643,
            "range": "± 6612821",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 231441520,
            "range": "± 16510845",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 164720103,
            "range": "± 1019578",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 329744218,
            "range": "± 1971306",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 650772916,
            "range": "± 4338385",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1300020542,
            "range": "± 15641346",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 447116656,
            "range": "± 1495654",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 901960667,
            "range": "± 8278710",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1764044875,
            "range": "± 5005925",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3477518646,
            "range": "± 25970834",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mario.rugiero@nextroll.com",
            "name": "Mario Rugiero",
            "username": "Oppen"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a02b3167d3a8d25d00ea09bde27837e08554eefb",
          "message": "perf: specialize is_neutral_element for each curve (#403)",
          "timestamp": "2023-06-02T13:01:19Z",
          "tree_id": "da6d217a92dd504c9b736d78d2a2e3586b61c7b7",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/a02b3167d3a8d25d00ea09bde27837e08554eefb"
        },
        "date": 1685713261221,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 1660553725,
            "range": "± 29579445",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 4521524020,
            "range": "± 29659987",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 3492784070,
            "range": "± 74551740",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 9934834262,
            "range": "± 91322667",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 7203543136,
            "range": "± 124590079",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 21361320706,
            "range": "± 81203556",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 15152374378,
            "range": "± 167529036",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 45731012624,
            "range": "± 225465086",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 42008940,
            "range": "± 1018114",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 42983840,
            "range": "± 1019131",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 88517470,
            "range": "± 1320015",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 88450192,
            "range": "± 998736",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 86269983,
            "range": "± 883426",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 87408726,
            "range": "± 2048208",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 178776310,
            "range": "± 2644516",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 179028552,
            "range": "± 1959283",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 177254582,
            "range": "± 2505863",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 175058821,
            "range": "± 3312520",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 367075761,
            "range": "± 8926130",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 359536268,
            "range": "± 2820116",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 353255529,
            "range": "± 5584008",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 347111296,
            "range": "± 4902910",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 719194286,
            "range": "± 8852131",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 725294496,
            "range": "± 9159437",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 98617407,
            "range": "± 2691860",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 211340333,
            "range": "± 2385529",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 434661991,
            "range": "± 6303984",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 880486983,
            "range": "± 12342223",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 1763170480,
            "range": "± 21046295",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 3747687923,
            "range": "± 45438050",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 7807340972,
            "range": "± 80275460",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 16122272763,
            "range": "± 111658726",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 1898237356,
            "range": "± 28729146",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 3941454173,
            "range": "± 46879931",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 8163646575,
            "range": "± 82743192",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 17021851843,
            "range": "± 361824055",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 22,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 40,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 97,
            "range": "± 4",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 17,
            "range": "± 2",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 183,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 134,
            "range": "± 10",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 130,
            "range": "± 7",
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
          "id": "7bb6d220b12dcb7d7e478757e7ed546abffa2f59",
          "message": "Add mixed addition optimization for the miller loop of the BLS 12 381 curve (#364)\n\n* add mixed addition optimization for the miller loop of the bls12381\ncurve\n\n* clippy, fmt\n\n* add missing test\n\n* add benches for ate pairing\n\n* fmt",
          "timestamp": "2023-06-02T14:18:23Z",
          "tree_id": "93308cd71434f51cfbaed1fd9080dd69594fba20",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/7bb6d220b12dcb7d7e478757e7ed546abffa2f59"
        },
        "date": 1685715841984,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 130851135,
            "range": "± 3659336",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 254275479,
            "range": "± 1670623",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 494241500,
            "range": "± 1331675",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 980858124,
            "range": "± 8835134",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 34313041,
            "range": "± 194481",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 67749073,
            "range": "± 462373",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 133230201,
            "range": "± 1566964",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 276941104,
            "range": "± 3967575",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 31276682,
            "range": "± 324979",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 59833848,
            "range": "± 2996569",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 121047913,
            "range": "± 4704465",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 239318062,
            "range": "± 18108360",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 163976253,
            "range": "± 953359",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 330249135,
            "range": "± 1557263",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 650869417,
            "range": "± 4360797",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1306967354,
            "range": "± 13066061",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 448159198,
            "range": "± 1229719",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 899732708,
            "range": "± 2250037",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1764314812,
            "range": "± 5005130",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3441581771,
            "range": "± 20887434",
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
          "id": "7bb6d220b12dcb7d7e478757e7ed546abffa2f59",
          "message": "Add mixed addition optimization for the miller loop of the BLS 12 381 curve (#364)\n\n* add mixed addition optimization for the miller loop of the bls12381\ncurve\n\n* clippy, fmt\n\n* add missing test\n\n* add benches for ate pairing\n\n* fmt",
          "timestamp": "2023-06-02T14:18:23Z",
          "tree_id": "93308cd71434f51cfbaed1fd9080dd69594fba20",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/7bb6d220b12dcb7d7e478757e7ed546abffa2f59"
        },
        "date": 1685717504390,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 1417155868,
            "range": "± 1303323",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 3699207786,
            "range": "± 21104686",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 2952793899,
            "range": "± 2119419",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 8008650717,
            "range": "± 23972098",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 6163377837,
            "range": "± 3862433",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 17267196403,
            "range": "± 38979475",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 12831932179,
            "range": "± 6875634",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 36973317072,
            "range": "± 61879001",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 37321397,
            "range": "± 259768",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 37306098,
            "range": "± 212371",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 76790443,
            "range": "± 672956",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 77420204,
            "range": "± 534565",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 74940068,
            "range": "± 231767",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 75180737,
            "range": "± 273073",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 155224204,
            "range": "± 510366",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 155256456,
            "range": "± 538339",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 150602688,
            "range": "± 446274",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 150646302,
            "range": "± 349424",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 315391469,
            "range": "± 1100786",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 314983139,
            "range": "± 1990286",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 299719497,
            "range": "± 335679",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 299738140,
            "range": "± 906720",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 638412647,
            "range": "± 6285699",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 637266870,
            "range": "± 4018282",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 78681382,
            "range": "± 828330",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 159430920,
            "range": "± 445286",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 323166285,
            "range": "± 1650885",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 661534436,
            "range": "± 3925216",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 1519032325,
            "range": "± 2308995",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 3167599675,
            "range": "± 3587988",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 6588739420,
            "range": "± 6295606",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 13693423719,
            "range": "± 17969870",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 1604025484,
            "range": "± 2098039",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 3326890396,
            "range": "± 2573320",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 6902741311,
            "range": "± 4464892",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 14326637924,
            "range": "± 12158703",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 377,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 6037,
            "range": "± 3",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 387,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 18,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 577,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 399,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 400,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mario.rugiero@nextroll.com",
            "name": "Mario Rugiero",
            "username": "Oppen"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a80c169928706ea6423991a420657b123dd11050",
          "message": "perf: use `2` as first non-qr candidate (#400)\n\n`1` is a quad residue no matter the field, so we can just skip it when\nwe look for our first non-residue for `sqrt`. This saves us a call to\n`legendre_symbol`, giving a tiny but non-negligible boost.",
          "timestamp": "2023-06-02T14:57:07Z",
          "tree_id": "845858e5d855e408b42fc29ebccf0d2a74a66036",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/a80c169928706ea6423991a420657b123dd11050"
        },
        "date": 1685718178226,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 130310160,
            "range": "± 1924639",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 256185552,
            "range": "± 2409556",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 484811625,
            "range": "± 3255579",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 977968874,
            "range": "± 7143520",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 34149320,
            "range": "± 354000",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 68003170,
            "range": "± 452421",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 132901609,
            "range": "± 318989",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 280093510,
            "range": "± 3415111",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 31432688,
            "range": "± 823689",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 52038016,
            "range": "± 1583511",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 120984933,
            "range": "± 6934445",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 261054784,
            "range": "± 17312012",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 163654637,
            "range": "± 1555592",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 330909719,
            "range": "± 950888",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 652786562,
            "range": "± 4163357",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1307688500,
            "range": "± 12239115",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 446803437,
            "range": "± 2334575",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 894876166,
            "range": "± 5716687",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1736722333,
            "range": "± 7624469",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3474517854,
            "range": "± 17443738",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mario.rugiero@nextroll.com",
            "name": "Mario Rugiero",
            "username": "Oppen"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a80c169928706ea6423991a420657b123dd11050",
          "message": "perf: use `2` as first non-qr candidate (#400)\n\n`1` is a quad residue no matter the field, so we can just skip it when\nwe look for our first non-residue for `sqrt`. This saves us a call to\n`legendre_symbol`, giving a tiny but non-negligible boost.",
          "timestamp": "2023-06-02T14:57:07Z",
          "tree_id": "845858e5d855e408b42fc29ebccf0d2a74a66036",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/a80c169928706ea6423991a420657b123dd11050"
        },
        "date": 1685720306008,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 1873826030,
            "range": "± 35148431",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 4760272524,
            "range": "± 48149155",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 3891367593,
            "range": "± 55024189",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 10387541242,
            "range": "± 84337063",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 7794397187,
            "range": "± 59694153",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 22261502254,
            "range": "± 86661132",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 16451342379,
            "range": "± 155248800",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 47550521306,
            "range": "± 90694506",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 48003489,
            "range": "± 912071",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 47645845,
            "range": "± 1114374",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 95778737,
            "range": "± 1457740",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 96310978,
            "range": "± 1761330",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 93874789,
            "range": "± 1603055",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 95603066,
            "range": "± 1722312",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 189170505,
            "range": "± 2332929",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 193779577,
            "range": "± 3683185",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 189658802,
            "range": "± 3110959",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 187667442,
            "range": "± 2702197",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 386129231,
            "range": "± 4732968",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 379095928,
            "range": "± 4287207",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 375039536,
            "range": "± 6409171",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 378233012,
            "range": "± 3525049",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 747574564,
            "range": "± 10627753",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 759338344,
            "range": "± 11981309",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 106854900,
            "range": "± 2455850",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 224710493,
            "range": "± 6490839",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 460083723,
            "range": "± 6052760",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 913774321,
            "range": "± 12415741",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 1973956447,
            "range": "± 41567382",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 4074295245,
            "range": "± 46990509",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 8521017099,
            "range": "± 84529643",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 17402956429,
            "range": "± 163981722",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 2049312705,
            "range": "± 21036885",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 4229945325,
            "range": "± 38018020",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 8852645134,
            "range": "± 153217604",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 18277162288,
            "range": "± 252943725",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 468,
            "range": "± 27",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 7517,
            "range": "± 375",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 526,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 21,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 802,
            "range": "± 35",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 550,
            "range": "± 27",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 556,
            "range": "± 31",
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
          "id": "6dd47c23a33d2fe312b8f77947ddc00222c2e8fb",
          "message": "sqrt_qfe for BLS12381TwistCurveFieldElement (#407)\n\nCo-authored-by: Pablo Deymonnaz <deymonnaz@gmail.com>",
          "timestamp": "2023-06-02T18:03:52Z",
          "tree_id": "3f89bb34025f27b860f712164198285d0e478392",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/6dd47c23a33d2fe312b8f77947ddc00222c2e8fb"
        },
        "date": 1685729376979,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 128790729,
            "range": "± 2313615",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 255883781,
            "range": "± 3030925",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 485699875,
            "range": "± 4087018",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 973337937,
            "range": "± 8787241",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 34024031,
            "range": "± 293689",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 68514061,
            "range": "± 944441",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 133321052,
            "range": "± 949728",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 280051083,
            "range": "± 3934313",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 31542271,
            "range": "± 233270",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 58958358,
            "range": "± 776808",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 120679936,
            "range": "± 4668256",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 259308208,
            "range": "± 18878994",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 163420469,
            "range": "± 1041197",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 330638250,
            "range": "± 1396618",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 645162958,
            "range": "± 6870973",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1298846625,
            "range": "± 13194333",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 452576666,
            "range": "± 1736659",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 900828250,
            "range": "± 4252332",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1762781020,
            "range": "± 5307951",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3481707958,
            "range": "± 11010515",
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
          "id": "6dd47c23a33d2fe312b8f77947ddc00222c2e8fb",
          "message": "sqrt_qfe for BLS12381TwistCurveFieldElement (#407)\n\nCo-authored-by: Pablo Deymonnaz <deymonnaz@gmail.com>",
          "timestamp": "2023-06-02T18:03:52Z",
          "tree_id": "3f89bb34025f27b860f712164198285d0e478392",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/6dd47c23a33d2fe312b8f77947ddc00222c2e8fb"
        },
        "date": 1685731468983,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 1633637615,
            "range": "± 30633663",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 4590393331,
            "range": "± 58086501",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 3649324109,
            "range": "± 48983156",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 10050331939,
            "range": "± 30615958",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 7691742055,
            "range": "± 193601354",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 21771833914,
            "range": "± 141364235",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 16049599376,
            "range": "± 225859319",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 46750561003,
            "range": "± 268066547",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 44756195,
            "range": "± 414556",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 43276949,
            "range": "± 1402982",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 90739018,
            "range": "± 1611287",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 94179884,
            "range": "± 4023734",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 92340070,
            "range": "± 1720630",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 92032670,
            "range": "± 1370631",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 188197780,
            "range": "± 3189347",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 189308785,
            "range": "± 3142525",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 185532320,
            "range": "± 4950132",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 184258817,
            "range": "± 5027591",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 382794690,
            "range": "± 5490091",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 371516783,
            "range": "± 3661656",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 406549319,
            "range": "± 25449086",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 356056328,
            "range": "± 6893189",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 734734672,
            "range": "± 9019720",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 741411316,
            "range": "± 13665498",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 101882653,
            "range": "± 2309504",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 220965509,
            "range": "± 3055936",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 447539980,
            "range": "± 5212444",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 893415816,
            "range": "± 12643068",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 1944783576,
            "range": "± 32623540",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 3965159940,
            "range": "± 103153315",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 8118062175,
            "range": "± 121760306",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 16841697007,
            "range": "± 342080941",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 1968807670,
            "range": "± 44923587",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 4046424634,
            "range": "± 46965681",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 8781967048,
            "range": "± 123461792",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 18296998768,
            "range": "± 152772433",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 938,
            "range": "± 38",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 29189,
            "range": "± 1558",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 849,
            "range": "± 36",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 34,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 1228,
            "range": "± 89",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 993,
            "range": "± 66",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 836,
            "range": "± 47",
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
          "id": "15246ff9e92984ec3deec911ed9a99ca5676fde0",
          "message": "Using msm pippenger in kzg commit (#408)\n\nCo-authored-by: Pablo Deymonnaz <deymonnaz@gmail.com>",
          "timestamp": "2023-06-02T19:07:40Z",
          "tree_id": "a4f16bf0caaa1550dc8249aea4569d9de8376c87",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/15246ff9e92984ec3deec911ed9a99ca5676fde0"
        },
        "date": 1685733205015,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 130532397,
            "range": "± 3430082",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 255994916,
            "range": "± 3959352",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 489037770,
            "range": "± 6511889",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 976864625,
            "range": "± 7626470",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 35221881,
            "range": "± 514067",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 69074319,
            "range": "± 636630",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 136485650,
            "range": "± 1876680",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 281050646,
            "range": "± 4197297",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 28476938,
            "range": "± 796247",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 55179391,
            "range": "± 2662311",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 120180298,
            "range": "± 7950011",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 245244409,
            "range": "± 18337948",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 165077293,
            "range": "± 2489869",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 330042718,
            "range": "± 1607319",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 655402771,
            "range": "± 5799272",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1304820396,
            "range": "± 14136339",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 445696791,
            "range": "± 6909778",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 895382250,
            "range": "± 5538010",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1742387646,
            "range": "± 6440818",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3444762000,
            "range": "± 14747359",
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
          "id": "15246ff9e92984ec3deec911ed9a99ca5676fde0",
          "message": "Using msm pippenger in kzg commit (#408)\n\nCo-authored-by: Pablo Deymonnaz <deymonnaz@gmail.com>",
          "timestamp": "2023-06-02T19:07:40Z",
          "tree_id": "a4f16bf0caaa1550dc8249aea4569d9de8376c87",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/15246ff9e92984ec3deec911ed9a99ca5676fde0"
        },
        "date": 1685734667481,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 1484833178,
            "range": "± 2393916",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 2397628304,
            "range": "± 31027970",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 3103169099,
            "range": "± 7195656",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 5409291740,
            "range": "± 23224496",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 6466561985,
            "range": "± 6803717",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 11741259733,
            "range": "± 22581075",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 13481835210,
            "range": "± 14594899",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 25324114675,
            "range": "± 60383625",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 42270277,
            "range": "± 91894",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 42048372,
            "range": "± 91383",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 70466214,
            "range": "± 220878",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 71142210,
            "range": "± 754611",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 84073198,
            "range": "± 136195",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 84337400,
            "range": "± 177502",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 142095538,
            "range": "± 527205",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 141135192,
            "range": "± 696671",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 167639984,
            "range": "± 373301",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 166763279,
            "range": "± 448857",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 278366733,
            "range": "± 748550",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 278242306,
            "range": "± 1247502",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 331938786,
            "range": "± 372796",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 331646880,
            "range": "± 577805",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 552719520,
            "range": "± 2203426",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 552617556,
            "range": "± 2934122",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 68582786,
            "range": "± 1731441",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 150145341,
            "range": "± 1104924",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 294972290,
            "range": "± 2753855",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 602101240,
            "range": "± 2150564",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 1571708154,
            "range": "± 3721010",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 3285581519,
            "range": "± 3774725",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 6844879547,
            "range": "± 2150582",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 14232380364,
            "range": "± 13481685",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 1645275871,
            "range": "± 2423938",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 3432208911,
            "range": "± 3322011",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 7129284819,
            "range": "± 7960436",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 14773209218,
            "range": "± 11931427",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 503,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 16003,
            "range": "± 8",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 403,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 16,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 677,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 418,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 378,
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
          "id": "818ca85ab17e03b18fd5aeb94ef073ab7a6ef7bb",
          "message": "Move default sqrt implementation to IsPrimeField trait (#402)\n\n* move default sqrt implementation to IsPrimeField trait\n\n* refactor sqrt\n\n* minor refactor\n\n* remove old tests\n\n* test both square roots",
          "timestamp": "2023-06-05T14:45:58Z",
          "tree_id": "ffd16497a9fd7b36f5a516db42c44631ed0e497a",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/818ca85ab17e03b18fd5aeb94ef073ab7a6ef7bb"
        },
        "date": 1685976697876,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 128803088,
            "range": "± 2859410",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 252598583,
            "range": "± 2057236",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 489177937,
            "range": "± 4416882",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 981345688,
            "range": "± 6662033",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 34516673,
            "range": "± 325259",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 68324435,
            "range": "± 637389",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 133974605,
            "range": "± 1055974",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 277186864,
            "range": "± 3602203",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 30087744,
            "range": "± 1500376",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 59076371,
            "range": "± 743009",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 123598046,
            "range": "± 4172940",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 258850756,
            "range": "± 15631429",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 164890392,
            "range": "± 678893",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 330788937,
            "range": "± 1627924",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 648551229,
            "range": "± 2491797",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1307033187,
            "range": "± 19597377",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 452968020,
            "range": "± 1376476",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 897510792,
            "range": "± 6060990",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1760139542,
            "range": "± 9123120",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3446940250,
            "range": "± 23331021",
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
          "id": "818ca85ab17e03b18fd5aeb94ef073ab7a6ef7bb",
          "message": "Move default sqrt implementation to IsPrimeField trait (#402)\n\n* move default sqrt implementation to IsPrimeField trait\n\n* refactor sqrt\n\n* minor refactor\n\n* remove old tests\n\n* test both square roots",
          "timestamp": "2023-06-05T14:45:58Z",
          "tree_id": "ffd16497a9fd7b36f5a516db42c44631ed0e497a",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/818ca85ab17e03b18fd5aeb94ef073ab7a6ef7bb"
        },
        "date": 1685978768019,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 1691730359,
            "range": "± 18384056",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 4623480871,
            "range": "± 45731328",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 3585520901,
            "range": "± 64111744",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 10101915674,
            "range": "± 83550177",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 7557208068,
            "range": "± 469623458",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 21695241194,
            "range": "± 50903450",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 15620545233,
            "range": "± 157705479",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 46566166575,
            "range": "± 108238917",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 43914222,
            "range": "± 1156423",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 43104856,
            "range": "± 1189377",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 93288473,
            "range": "± 3129476",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 89470444,
            "range": "± 1494350",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 90889225,
            "range": "± 1260724",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 92109252,
            "range": "± 1899122",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 184513883,
            "range": "± 4421910",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 182777796,
            "range": "± 3413526",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 179550289,
            "range": "± 3290232",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 179906152,
            "range": "± 3864066",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 365440752,
            "range": "± 4333146",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 366608251,
            "range": "± 5201380",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 355366951,
            "range": "± 9029675",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 365449755,
            "range": "± 5986424",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 730974239,
            "range": "± 14291039",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 725769522,
            "range": "± 9690325",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 100320088,
            "range": "± 2254369",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 212725432,
            "range": "± 1301504",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 439353175,
            "range": "± 4464026",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 904242878,
            "range": "± 22650268",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 1855073940,
            "range": "± 29194982",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 3841069565,
            "range": "± 50637007",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 8047633927,
            "range": "± 71531759",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 16653179207,
            "range": "± 205029475",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 1968109197,
            "range": "± 26248666",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 4096119636,
            "range": "± 29790417",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 8460210770,
            "range": "± 100037998",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 17760597445,
            "range": "± 559596432",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 97,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 429,
            "range": "± 22",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 141,
            "range": "± 6",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 18,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 238,
            "range": "± 13",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 184,
            "range": "± 7",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 202,
            "range": "± 9",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mario.rugiero@nextroll.com",
            "name": "Mario Rugiero",
            "username": "Oppen"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c517e5c581f273ac6028ea1e293e8e00d0d33233",
          "message": "perf: improve accuracy of FFT benches (#411)\n\n- Use `iter_batched` rather than cloning input inside the benchmarked\n  functions\n- Properly fix symbol names and exclude from inlining the helper\n  functions to help IAI exclude them from the measurement\n- Exclude bit reversal from ordered FFT benchmarks, as that's measured\n  separately",
          "timestamp": "2023-06-05T17:09:27Z",
          "tree_id": "06b797b35eacf5ca2a3d75df2696119777e40683",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/c517e5c581f273ac6028ea1e293e8e00d0d33233"
        },
        "date": 1685985305116,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 130591786,
            "range": "± 3100559",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 254213437,
            "range": "± 2733442",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 486680031,
            "range": "± 3696186",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 983118104,
            "range": "± 5924517",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 34227321,
            "range": "± 297648",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 67513068,
            "range": "± 492486",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 133238874,
            "range": "± 1239506",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 278403687,
            "range": "± 4552307",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 31272775,
            "range": "± 255511",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 58191860,
            "range": "± 2284804",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 125099698,
            "range": "± 4635315",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 267069427,
            "range": "± 18535704",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 163590516,
            "range": "± 3541480",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 323206791,
            "range": "± 11160388",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 646302042,
            "range": "± 8848957",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1311018229,
            "range": "± 9187941",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 450900864,
            "range": "± 1973128",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 899649063,
            "range": "± 2782997",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1765026041,
            "range": "± 4272359",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3475449771,
            "range": "± 16296087",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mario.rugiero@nextroll.com",
            "name": "Mario Rugiero",
            "username": "Oppen"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "844f912664d74ce918d0237123ba720c7d90b296",
          "message": "perf: improve accuracy of field IAI benchmarks (#412)\n\n- Extract FP element sample creation into a `util` function to allow\n  exclusion from measurement.\n- Likewise for getting a quadratic residue by squaring the sample.\n- Replace the apprpriate call in IAI and Criterion benches.",
          "timestamp": "2023-06-05T17:36:10Z",
          "tree_id": "85d77b4d73c23b29c48de48701d726ba4e18c2d0",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/844f912664d74ce918d0237123ba720c7d90b296"
        },
        "date": 1685986920879,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 130916250,
            "range": "± 2484817",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 256456614,
            "range": "± 2848197",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 484700666,
            "range": "± 3023891",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 978701646,
            "range": "± 8050398",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 34533771,
            "range": "± 353203",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 68083458,
            "range": "± 609976",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 132864407,
            "range": "± 801874",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 277195322,
            "range": "± 3317624",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 31251258,
            "range": "± 1129979",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 56643798,
            "range": "± 2864754",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 117950972,
            "range": "± 5503554",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 264116156,
            "range": "± 16066397",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 165203418,
            "range": "± 966060",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 329746385,
            "range": "± 2446209",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 648281125,
            "range": "± 5964618",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1309259625,
            "range": "± 12577508",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 451204375,
            "range": "± 6767478",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 934087333,
            "range": "± 78960592",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 2039085187,
            "range": "± 29869859",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 4152415937,
            "range": "± 682417719",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mario.rugiero@nextroll.com",
            "name": "Mario Rugiero",
            "username": "Oppen"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c517e5c581f273ac6028ea1e293e8e00d0d33233",
          "message": "perf: improve accuracy of FFT benches (#411)\n\n- Use `iter_batched` rather than cloning input inside the benchmarked\n  functions\n- Properly fix symbol names and exclude from inlining the helper\n  functions to help IAI exclude them from the measurement\n- Exclude bit reversal from ordered FFT benchmarks, as that's measured\n  separately",
          "timestamp": "2023-06-05T17:09:27Z",
          "tree_id": "06b797b35eacf5ca2a3d75df2696119777e40683",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/c517e5c581f273ac6028ea1e293e8e00d0d33233"
        },
        "date": 1685987014239,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 1335505544,
            "range": "± 1671760",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 3727222514,
            "range": "± 17088163",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 2794268579,
            "range": "± 3172589",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 8204785640,
            "range": "± 20999255",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 5834901822,
            "range": "± 4432129",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 17827343496,
            "range": "± 51613618",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 12168595049,
            "range": "± 4281767",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 38141555371,
            "range": "± 99129524",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 36972215,
            "range": "± 190262",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 36967673,
            "range": "± 312317",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 83746183,
            "range": "± 539432",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 84212948,
            "range": "± 719400",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 74299358,
            "range": "± 193638",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 74477174,
            "range": "± 286380",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 172013027,
            "range": "± 1409066",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 171494752,
            "range": "± 1738009",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 149807855,
            "range": "± 361996",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 149246464,
            "range": "± 217404",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 344026175,
            "range": "± 2337774",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 344712290,
            "range": "± 2385007",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 298697579,
            "range": "± 1137275",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 298389011,
            "range": "± 666478",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 671821658,
            "range": "± 11635474",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 679835705,
            "range": "± 3696177",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 61633917,
            "range": "± 1286058",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 132774170,
            "range": "± 1368890",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 261823505,
            "range": "± 2020125",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 534751752,
            "range": "± 4708390",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 1535810074,
            "range": "± 2527233",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 3202500161,
            "range": "± 3154181",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 6650667046,
            "range": "± 8729963",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 13789634329,
            "range": "± 9601588",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 1620813316,
            "range": "± 3475769",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 3374814193,
            "range": "± 4190671",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 6991062244,
            "range": "± 8956774",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 14464805540,
            "range": "± 6822938",
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
            "value": 35,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 85,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 15,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 157,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 118,
            "range": "± 0",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 118,
            "range": "± 0",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mario.rugiero@nextroll.com",
            "name": "Mario Rugiero",
            "username": "Oppen"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "72c3dba9d10b8ad08e5471defc9fe4c66558337e",
          "message": "perf: optimize MSM routines (#405)\n\n- Replace multiplication by small constants with addition chains, as\n  their construction in Montgomery is expensive.\n- Reduce duplication of operations by storing temporaries.\n- Replace an order check by its equivalent inequality check to avoid\n  conversion to normal form.",
          "timestamp": "2023-06-05T17:39:47Z",
          "tree_id": "b7f13acd10fda09d28e73374d9bf0051f0059f31",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/72c3dba9d10b8ad08e5471defc9fe4c66558337e"
        },
        "date": 1685987132473,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Parallel (Metal)",
            "value": 189675947,
            "range": "± 181196314",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #2",
            "value": 462513208,
            "range": "± 351022248",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #3",
            "value": 841251958,
            "range": "± 812254260",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Parallel (Metal) #4",
            "value": 2031071250,
            "range": "± 1014498395",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal)",
            "value": 50623584,
            "range": "± 14772861",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #2",
            "value": 73541039,
            "range": "± 4000230",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #3",
            "value": 132754158,
            "range": "± 351317",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/Parallel (Metal) #4",
            "value": 278320281,
            "range": "± 3106630",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal)",
            "value": 31083529,
            "range": "± 257434",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #2",
            "value": 59036012,
            "range": "± 1463520",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #3",
            "value": 120337295,
            "range": "± 4669278",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Parallel (Metal) #4",
            "value": 250296159,
            "range": "± 16129656",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal",
            "value": 163579449,
            "range": "± 754572",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #2",
            "value": 329907010,
            "range": "± 2150442",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #3",
            "value": 650568000,
            "range": "± 5528754",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_fft_metal #4",
            "value": 1306848979,
            "range": "± 12348332",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal",
            "value": 447115395,
            "range": "± 2138106",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #2",
            "value": 897028458,
            "range": "± 6068265",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #3",
            "value": 1739366416,
            "range": "± 28506381",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/interpolate_fft_metal #4",
            "value": 3427920291,
            "range": "± 58230885",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mario.rugiero@nextroll.com",
            "name": "Mario Rugiero",
            "username": "Oppen"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "72c3dba9d10b8ad08e5471defc9fe4c66558337e",
          "message": "perf: optimize MSM routines (#405)\n\n- Replace multiplication by small constants with addition chains, as\n  their construction in Montgomery is expensive.\n- Reduce duplication of operations by storing temporaries.\n- Replace an order check by its equivalent inequality check to avoid\n  conversion to normal form.",
          "timestamp": "2023-06-05T17:39:47Z",
          "tree_id": "b7f13acd10fda09d28e73374d9bf0051f0059f31",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/72c3dba9d10b8ad08e5471defc9fe4c66558337e"
        },
        "date": 1685988782515,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 1291195878,
            "range": "± 13301871",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 3740815586,
            "range": "± 131585078",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 2658166672,
            "range": "± 44330104",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 8152623021,
            "range": "± 101178561",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 5781155044,
            "range": "± 156350362",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 17698329995,
            "range": "± 84029239",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 11787439922,
            "range": "± 142029172",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 38039945097,
            "range": "± 159401881",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 34686777,
            "range": "± 568297",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 34217899,
            "range": "± 834722",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 76322643,
            "range": "± 1183098",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 75869900,
            "range": "± 1017156",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 72962520,
            "range": "± 2432083",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 73226688,
            "range": "± 1634604",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 159280439,
            "range": "± 1571690",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 162878119,
            "range": "± 1437537",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 147958673,
            "range": "± 4281272",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 149144545,
            "range": "± 4274349",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 320235570,
            "range": "± 2342046",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 323781713,
            "range": "± 2352369",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 298300879,
            "range": "± 7680139",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 304232365,
            "range": "± 5360314",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 650344109,
            "range": "± 9261529",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 639791525,
            "range": "± 8256956",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 47459474,
            "range": "± 789257",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 117482977,
            "range": "± 1244576",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 243133718,
            "range": "± 1089113",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 502835814,
            "range": "± 2627805",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 1436752588,
            "range": "± 31248466",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 3087335275,
            "range": "± 43662002",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 6385504379,
            "range": "± 126659682",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 13378465525,
            "range": "± 163200328",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 1526557501,
            "range": "± 26930296",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 3188243151,
            "range": "± 28950108",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 6651998982,
            "range": "± 117045640",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 13827275093,
            "range": "± 146110596",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 738,
            "range": "± 40",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 23064,
            "range": "± 1190",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 585,
            "range": "± 34",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/neg",
            "value": 17,
            "range": "± 1",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/sub",
            "value": 853,
            "range": "± 39",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 582,
            "range": "± 29",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 581,
            "range": "± 36",
            "unit": "ns/iter"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mario.rugiero@nextroll.com",
            "name": "Mario Rugiero",
            "username": "Oppen"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "844f912664d74ce918d0237123ba720c7d90b296",
          "message": "perf: improve accuracy of field IAI benchmarks (#412)\n\n- Extract FP element sample creation into a `util` function to allow\n  exclusion from measurement.\n- Likewise for getting a quadratic residue by squaring the sample.\n- Replace the apprpriate call in IAI and Criterion benches.",
          "timestamp": "2023-06-05T17:36:10Z",
          "tree_id": "85d77b4d73c23b29c48de48701d726ba4e18c2d0",
          "url": "https://github.com/lambdaclass/lambdaworks/commit/844f912664d74ce918d0237123ba720c7d90b296"
        },
        "date": 1685988799515,
        "tool": "cargo",
        "benches": [
          {
            "name": "Ordered FFT/Sequential from NR radix2",
            "value": 1561354967,
            "range": "± 9633731",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2",
            "value": 3903897142,
            "range": "± 36836051",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #2",
            "value": 3279307843,
            "range": "± 28310577",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #2",
            "value": 8637007207,
            "range": "± 106941834",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #3",
            "value": 6819664062,
            "range": "± 43709846",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #3",
            "value": 18898090538,
            "range": "± 65265302",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from NR radix2 #4",
            "value": 14252770866,
            "range": "± 56789354",
            "unit": "ns/iter"
          },
          {
            "name": "Ordered FFT/Sequential from RN radix2 #4",
            "value": 40492178574,
            "range": "± 48162995",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural",
            "value": 42261246,
            "range": "± 496720",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed",
            "value": 42331032,
            "range": "± 318719",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed",
            "value": 85464394,
            "range": "± 523960",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed",
            "value": 86466877,
            "range": "± 518288",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #2",
            "value": 84862552,
            "range": "± 1094733",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #2",
            "value": 85255281,
            "range": "± 770840",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #2",
            "value": 173319531,
            "range": "± 924110",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #2",
            "value": 173229011,
            "range": "± 1890842",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #3",
            "value": 170176294,
            "range": "± 1618152",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #3",
            "value": 170641638,
            "range": "± 1810163",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #3",
            "value": 350128485,
            "range": "± 3354356",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #3",
            "value": 350423994,
            "range": "± 1690182",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural #4",
            "value": 340203160,
            "range": "± 2408307",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/natural inversed #4",
            "value": 337591259,
            "range": "± 3736032",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed #4",
            "value": 704654326,
            "range": "± 3573441",
            "unit": "ns/iter"
          },
          {
            "name": "FFT twiddles generation/bit-reversed inversed #4",
            "value": 699852962,
            "range": "± 7025097",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential",
            "value": 55962948,
            "range": "± 1667096",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #2",
            "value": 125070917,
            "range": "± 1854379",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #3",
            "value": 248040186,
            "range": "± 3289402",
            "unit": "ns/iter"
          },
          {
            "name": "Bit-reverse permutation/Sequential #4",
            "value": 511336301,
            "range": "± 3561900",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT",
            "value": 1754659566,
            "range": "± 10372138",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #2",
            "value": 3635759405,
            "range": "± 23363692",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #3",
            "value": 7510945329,
            "range": "± 46558526",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial evaluation/Sequential FFT #4",
            "value": 15665555220,
            "range": "± 41276580",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT",
            "value": 1829320395,
            "range": "± 7597250",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #2",
            "value": 3804711783,
            "range": "± 20593126",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #3",
            "value": 7917440428,
            "range": "± 27518734",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial interpolation/Sequential FFT #4",
            "value": 16261511052,
            "range": "± 45567853",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate",
            "value": 437,
            "range": "± 8",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/evaluate_slice",
            "value": 6917,
            "range": "± 207",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/add",
            "value": 439,
            "range": "± 15",
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
            "value": 675,
            "range": "± 14",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/mul",
            "value": 470,
            "range": "± 10",
            "unit": "ns/iter"
          },
          {
            "name": "Polynomial/div",
            "value": 602,
            "range": "± 13",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}