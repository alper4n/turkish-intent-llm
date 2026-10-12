# Türkçe Niyet Sınıflandırma: LLM Fine-Tuning vs. Klasik Baseline

[Amazon MASSIVE](https://github.com/alexa/massive) veri setinin **tr-TR** bölümünde,
Türkçe sesli asistan komutlarının 60 niyet sınıfına ayrılması. LoRA ile eğitilmiş bir LLM,
Türkçe üzerinde önceden eğitilmiş bir BERT encoder'ı ve hiç eğitim görmemiş prompting
karşılaştırılıyor.

> Bu dosya kısa bir özettir. Bütün sonuçlar, grafikler, metodoloji ve sınırlılıklar için
> **[README.md](README.md)** esastır; iki metin ayrışırsa İngilizce olan geçerlidir.

## Görev

Bir Türkçe cümle girer, 60 niyet etiketinden biri çıkar:

| Cümle | Doğru etiket |
|---|---|
| ışıkları kıs | `iot_hue_lightdim` |
| bu hafta beni sabah beşte uyandır | `alarm_set` |
| bana bir fıkra anlatır mısın | `general_joke` |
| adana için bugünkü hava nasıl | `weather_query` |

## Soru

11 bin örneklik bir Türkçe veri setinde, LoRA ile eğitilmiş 1,5 milyar parametreli bir LLM,
Türkçe üzerinde önceden eğitilmiş 110 milyonluk bir encoder'ı gerçekten yener mi?

## Cevap: hayır

| Deney | Accuracy | Macro-F1 |
|---|---:|---:|
| Zero-shot Qwen2.5-1.5B | 0,4526 | 0,4049 |
| Qwen2.5-1.5B + LoRA | 0,8615 | 0,8383 |
| **BERTurk** (`dbmdz/bert-base-turkish-cased`) | **0,8773** | **0,8501** |

Encoder kazandı ve fark istatistiksel olarak anlamlı. Üstelik yarı sürede eğitiliyor ve
token üretmek yerine tek geçişte sınıflandırıyor.

Negatif sonuç gizlenmedi, manşet yapıldı: "LLM her şeyi yener" beklentisini ölçerek
reddetmek, doğrulamaktan daha bilgilendirici. LLM'in lehine olan tarafı da yazılı —
BERTurk'ün altıda biri kadar parametre eğitiyor ve takılıp çıkarılabilir küçük bir adapter
üretiyor; etiket kümesi sık değişen bir sistemde tercih tersine dönebilir.

## Türkçeye özgü iki bulgu

**Diakritik kaybının %84'ü tek harften geliyor.** Altı Türkçe diakritiği silmek modelin
doğruluğunu belirgin biçimde düşürüyor, ama kayıp eşit dağılmıyor: harf harf ölçüldüğünde
`ı` tek başına baskın çıkıyor. Sebebi dilbilimsel — ASCII'de tek bir `i` var, dolayısıyla
katlama noktalı/noktasız ayrımını çökertiyor ve o ayrım ünlü uyumunu, `-dı/-di`, `-ın/-in`
gibi en sık ekleri taşıyor. `ışıkları kıs` → `işiklari kis` olduğunda model komutu ışıkla
ilgili bile görmüyor. Pratik çıkarım: yazılı Türkçe alan bir sistem önce bu harfi onarmalı.

**Çok dilli sözlükler Türkçeye vergi koyuyor.** Aynı komutların Türkçesi ve İngilizcesi
dört farklı tokenizer'dan geçirildi. Test edilenler arasında BERTurk, Türkçeyi İngilizceden
*ucuz* bulan tek tokenizer; çok dilli olanların hepsi Türkçeye kelime başına daha fazla
token harcıyor. Ve sözlük büyüklüğü bunu çözmüyor — XLM-R'ın sözlüğü kat kat büyük olduğu
hâlde Türkçede daha kötü. Üretimde bu, gecikme ve bağlam bütçesi demek.

## Dürüstlük notu

README.md'de tam bir **Limitations** bölümü var: karşılaştırmanın asimetrik olduğu, LoRA'nın
eksik eğitilmiş olduğu ve skorunun bir taban sayılması gerektiği, tek seed kullanıldığı, ve
istenen üç Türkçe zorluktan birinin (kelime sırası) ölçülmediği — gerekçesiyle birlikte.

## Çalıştırma

Beş notebook da Google Colab'da doğrudan açılıp çalışıyor; her biri repoyu ilk hücrede
klonluyor. Ayrıntılar ve yerel kurulum için [README.md](README.md).
