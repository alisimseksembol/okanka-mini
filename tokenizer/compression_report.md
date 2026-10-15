# Tokenizer compression on Turkish text

Higher chars/token = fewer tokens for the same Turkish text = cheaper training and longer effective context.

## FineWeb-2 tur test (held-out web)

| tokenizer | tokens | chars/token | bytes/token |
|---|---:|---:|---:|
| Okanka-32K (ours) | 1,674,821 | 4.845 | 5.300 |
| Okanka-65K (same data, Turkish pattern) | 1,543,242 | 5.259 | 5.752 |
| Okanka-32K (upstream English split pattern) | 1,677,158 | 4.839 | 5.293 |
| tiktoken gpt2 | 4,000,415 | 2.029 | 2.219 |
| tiktoken cl100k_base | 3,151,988 | 2.575 | 2.816 |
| tiktoken o200k_base | 2,349,035 | 3.455 | 3.779 |
| HF ytu-ce-cosmos/turkish-gpt2 | 1,617,780 | 5.016 | 5.487 |

## Türkçe Vikipedi (sample)

| tokenizer | tokens | chars/token | bytes/token |
|---|---:|---:|---:|
| Okanka-32K (ours) | 330,107 | 3.885 | 4.185 |
| Okanka-65K (same data, Turkish pattern) | 306,410 | 4.186 | 4.508 |
| Okanka-32K (upstream English split pattern) | 332,622 | 3.856 | 4.153 |
| tiktoken gpt2 | 605,137 | 2.120 | 2.283 |
| tiktoken cl100k_base | 502,925 | 2.550 | 2.747 |
| tiktoken o200k_base | 403,387 | 3.180 | 3.424 |
| HF ytu-ce-cosmos/turkish-gpt2 | 330,693 | 3.879 | 4.177 |

## ğşıİçöü sentences

| tokenizer | tokens | chars/token | bytes/token |
|---|---:|---:|---:|
| Okanka-32K (ours) | 107 | 4.757 | 5.449 |
| Okanka-65K (same data, Turkish pattern) | 97 | 5.247 | 6.010 |
| Okanka-32K (upstream English split pattern) | 110 | 4.627 | 5.300 |
| tiktoken gpt2 | 286 | 1.780 | 2.038 |
| tiktoken cl100k_base | 229 | 2.223 | 2.546 |
| tiktoken o200k_base | 171 | 2.977 | 3.409 |
| HF ytu-ce-cosmos/turkish-gpt2 | 103 | 4.942 | 5.660 |

## Example segmentation (ours)

- `Çağrı, İstanbul'daki ağaçların gölgesinde ılık bir öğle yemeği yedi.` -> 14 tokens: Çağ | rı | , |  İstanbul | 'daki |  ağaçların |  gölg | esinde |  ılık |  bir |  öğle |  yemeği |  yedi | .
- `Şişli'de oturan öğretmenimiz, öğrencilerine güzel bir öykü okudu.` -> 13 tokens: Ş | işli | 'de |  oturan |  öğretmen | imiz | , |  öğrencilerine |  güzel |  bir |  öykü |  okudu | .
- `Ilgaz Dağı'nın eteklerinde yağmur yağınca ırmaklar coşkuyla akar.` -> 16 tokens: I | l | gaz |  Dağı | 'nın |  etek | lerinde |  yağmur |  yağ | ınca |  ır | mak | lar |  coşkuyla |  akar | .
- `Türkiye Büyük Millet Meclisi 23 Nisan 1920'de Ankara'da açıldı.` -> 15 tokens: Türkiye |  Büyük |  Millet |  Meclisi |   | 23 |  Nisan |   | 19 | 20 | 'de |  Ankara | 'da |  açıldı | .
