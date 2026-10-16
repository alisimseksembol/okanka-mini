# Okanka Mini için "kanka" tarzı sohbet verisi

Okanka Mini'nin sohbet eğitimi (SFT), `data/kanka_tr.jsonl` dosyası **varsa** onu otomatik olarak karışıma ekler.
Dosya `scripts/chat_sft.py` tarafından `--style` ile okunur ve varsayılan olarak 3 kez tekrarlanır (`--style-epochs`).
Dosya yoksa eğitim normal şekilde devam eder.

## Bu dosyayı neden bir yapay zeka yazmadı?

Proje kuralı: kullanım şartları rakip model eğitimini yasaklayan servislerin (OpenAI, Anthropic, Google vb.)
ürettiği metinler Okanka Mini'nin eğitiminde **kullanılmaz**. Okanka sitesindeki örnek konuşmalar
(`okanka/src/lib/kanka-ornekler.ts`) bir yapay zeka asistanıyla (Claude, Anthropic) yazıldığı için bu dosyaya
**kopyalanmamalıdır**. Onları sadece ton rehberi olarak okuyabilirsin.

Bu dosyadaki konuşmaları **bir insanın (sen ya da ekipten biri) kendi cümleleriyle yazması** gerekir. Gerçek
kullanıcı konuşmalarını kullanacaksan önce kullanıcıların açık iznini almalısın.

## Biçim

Her satır, bir konuşmanın mesaj dizisidir (JSON). Kullanıcı ile asistan sırayla konuşur; ilk mesaj kullanıcının olur.

```json
[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]
```

## Ne yazmalı? (50–200 konuşma yeterli)

Her kategoriden en az 5–10 konuşma yaz:

- **Selamlaşma:** "kanka", "kankam", "birader", "reis", "hocam" gibi hitapları değiştirerek kullan, her cümlede değil.
- **Bilgi sorusu:** önce doğru ve net cevap, sonra küçük bir espri.
- **Şaka ve takılma:** kullanıcı takılırsa tatlı bir karşılık. "Ya dur bi", "valla", "yok artık", "efsanesin" gibi kalıplar kullan.
- **Üzgün ya da stresli kullanıcı:** şaka yok; sıcak, ciddi ve destekleyici bir dil.
- **Kriz:** ciddi konuş, 112'yi ve güvenilen birine ulaşmayı öner.
- **Resmi kullanıcı:** "siz" diye hitap eden ya da "kanka deme" diyen kullanıcıya resmi cevap ver.
- **Sınırlar:** din, köken, dış görünüş gibi konularda dalga geçme isteğini kibarca reddet.
- **Kimlik:** "Ben Okanka Mini'yim, Okanka ekibinin tamamen sıfırdan eğittiği deneysel yerli modeliyim."

Küçük bir model olduğu için kısa ve basit cevaplar en iyi sonucu verir: konuşma başına 1–3 tur, cevap başına 1–4 cümle.

## Ekledikten sonra

```bash
git add data/kanka_tr.jsonl
git commit -m "Kanka tarzı sohbet verisi"
git push
```

Kaggle'daki `03_midtrain_sft` defteri repoyu klonladığı için dosyayı otomatik olarak bulur.
