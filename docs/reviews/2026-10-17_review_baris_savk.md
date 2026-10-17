# DocBenchTR Deposu: Kapsamlı İnceleme Raporu

Oct 17, 2026 · @Barış Şavk

## Özet

Bu çalışma, uzun ve madde yapılı Türkçe hukuk metinlerinde RAG erişim yöntemlerini karşılaştıran DocBenchTR pilot deposunu (commit `97d7240`) bağımsız olarak incelemekte ve yayımlanan sayıların doğru, ancak bu sayıları üreten soru setinin ve onlardan çıkarılan başlık iddiasının zayıf olduğunu göstermektedir. Kaynak kod, testler, Colab defterleri ve altı koşunun soru başına çıktıları okunmuş; tüm metrikler ham verilerden yeniden hesaplanmış, altın etiketler belgelerin tam metniyle karşılaştırılmış ve PageIndex ile Hybrid arasındaki farklar eşleştirilmiş bootstrap ve işaret testiyle sınanmıştır. README'deki sayılar ham verilerle tutarlıdır (tek sapma bir yuvarlamadır) ve 12 birim testinin tamamı geçmektedir. Buna karşın “span doğrulaması” tanım gereği her soruyu geçirmekte ve dört soruda cevap soruyla uyuşmamaktadır. 36 soru yalnızca 27 benzersiz hedefe dayanır; 6502 sayılı Kanun belgesinde beş sorunun cevabı yönetmeliklerde aynı ya da eşdeğer biçimde bulunduğu hâlde yalnızca kanun sayfası doğru sayılır; KVKK'da ilk beş sayfayı döndüren basit bir yöntem Recall@5 = 1,000 alır. 27B model ve 64k bağlamla PageIndex Tree Search'ün Hybrid'e üstünlüğü +0,125 Recall@5'tir; ancak %95 güven aralığı \[−0,028; +0,278\], işaret testi p = 0,23'tür. Depo, şeffaflığı ve tekrarlanabilirlik altyapısıyla güçlü bir başlangıçtır; tam çalışmadan önce soru seti, altın etiketler ve istatistiksel raporlama yeniden ele alınmalıdır.

**Anahtar kelimeler:** bilgi erişimi, RAG, Türkçe hukuk metinleri, benchmark, PageIndex, BM25, tekrarlanabilirlik, kod incelemesi

## 1. Giriş

DocBenchTR, uzun ve madde yapılı Türkçe hukuk metinlerinde RAG için erişim (retrieval) yöntemlerini karşılaştıran bir pilot benchmark'tır. Depo, Ümit Altınöz (Beykent Üniversitesi) tarafından hazırlanmış ve MIT lisansıyla yayımlanmıştır ([depo](https://github.com/umitaltinozzz/docbenchtr)). Türkçe, yaygın erişim değerlendirmelerinde az temsil edilir; hukuk metinlerinde ise doğru hükmü anahtar kelimeyle değil anlam ve yapıyla bulmak gerekir. Projenin çıkış noktası budur.

Pilot, üç resmî PDF üzerinde 36 soru ile yedi yöntemi sayfa düzeyinde ölçer. Yalnızca erişim değerlendirilir; cevap üretimi ve sadakat kapsam dışıdır.

| Belge | Kimlik | Fiziksel sayfa | Soru |
| --- | --- | --: | --: |
| Kişisel Verilerin Korunması Kanunu | `kvkk_6698` | 18 | 12 |
| Tüketicinin Korunması Hakkında Kanun ve İkincil Mevzuatı (Ocak 2021) | `tuketicinin_korunmasi_6502` | 340 | 12 |
| Yükseköğretim Kanunu | `yuksekogretim_2547` | 142 | 12 |

Karşılaştırılan yöntemler: BM25, Türkçe köklendirmeli BM25, yoğun erişim (`multilingual-e5-base`), hibrit RRF ve PageIndex'in üç çeşidi (iki ajan tabanlı varyant ve LLM Tree Search). Metrikler Recall@1/3/5, MRR ve sorgu gecikmesidir.

Bu rapor dört soruya cevap arar:

1. Kod, README'de anlatılan işi doğru yapıyor mu?
2. Soru seti ve altın sayfa etiketleri sonuçları taşıyacak nitelikte mi?
3. README'deki sayılar ham sorgu çıktılarıyla tutarlı mı?
4. Başlık iddiaları, özellikle "PageIndex klasik yöntemleri geçer" iddiası, istatistiksel olarak destekleniyor mu?

## 2. İnceleme Yöntemi

İncelenen sürüm `main` dalındaki [`97d7240`](https://github.com/umitaltinozzz/docbenchtr/tree/97d7240a6afaf3af83a29a4c471841d611c62c2b) commit'idir (17 Ekim 2026). Depo 16–17 Ekim 2026 arasında tek yazarın attığı 25 commit'ten oluşur. İzlenen 76 dosya yaklaşık 16 MB tutar: yaklaşık 2.800 satır Python (paket kodu 2.450 satır; bunun 1.247 satırı `retrievers.py`), 3 test dosyası, 2 Colab defteri, 6 koşunun sonuçları, 3 kaynak PDF ve PageIndex ağaç önbelleği.

İnceleme altı adımda yapıldı:

1. **Statik inceleme.** Tüm modüller, testler, iki defter ve README satır satır okundu.
2. **Sonuçların yeniden hesaplanması.** Altı koşunun `per_query.json` dosyalarından (her biri 252 satır) tüm Recall@k ve MRR değerleri yeniden hesaplandı ve README tablolarıyla karşılaştırıldı.
3. **Soru seti analizi.** Sorulara planlanan anahtar terimler, seçilen cevap parçalarıyla karşılaştırıldı; tekrar eden hedef sayfalar ve basit taban çizgileri (“ilk 5 sayfa”, rastgele) hesaplandı.
4. **Altın etiket tamlığı.** Cevap metinleri ve plan terimleri, PageIndex önbelleğindeki sayfa metinlerinin tamamında arandı.
5. **İstatistik.** PageIndex ile Hybrid arasındaki fark için soru bazında eşleştirilmiş bootstrap (10.000 örnek, %95 güven aralığı) ve iki yönlü işaret testi uygulandı.
6. **Birim testleri.** `torch` ve `sentence-transformers` taklit modüllerle (stub) değiştirilerek pytest 8.4.1 ile çalıştırıldı: 12 testin 12'si geçti.

**Yapılmayanlar.** Uçtan uca akış yeniden çalıştırılmadı. Bu, CUDA'lı PyTorch, Ollama modelleri ve saatler süren PageIndex sorguları gerektirir. PDF metin çıkarımı ve gömme modeli bu yüzden yeniden üretilmedi; sayılar yayımlanmış ham çıktılar üzerinden doğrulandı. Kod atıfları bu commit'e sabitlenmiş bağlantılarla verilmiştir.

## 3. Depo Yapısı ve Mimari

Depo, tek bir komutla (`python -m docbenchtr.run_pilot`) uçtan uca çalışan doğrusal bir akış kurar: indir, metni çıkar, soruları üret, değerlendir, özetle ve README'nin üretilen bölümünü yeniden yaz.

&#91;embedded content: veri akışı · 5 adım, 2 yöntem ailesi\]

PageIndex ayrı bir işçi sürecinde çalışır ve yerel Ollama'ya bağlanır; ağaçlar 7B modelle bir kez kurulup önbellekten kullanılır. Colab defterleri aynı akışı A100'de farklı modellerle koşar ve yalnızca `results/` klasörünü geri getirir (B9).

| Bileşen | Boyut | Görev |
| --- | --: | --- |
| `docbenchtr/config.py` | 53 satır | Yollar, model adları, Ollama ayarları (ortam değişkeniyle değiştirilebilir), belge listesi ve SHA-256 değerleri |
| `docbenchtr/documents.py` | 131 satır | PDF indirme, SHA-256 kontrolü, `pdftotext` ile sayfa bazında metin çıkarımı |
| `docbenchtr/questions.py` | 256 satır | Elle yazılmış soru planı, madde ayrıştırma, cevap parçası seçimi ve doğrulama |
| `docbenchtr/retrievers.py` | 1.247 satır | BM25, Dense, Hybrid; PageIndex ajan ve Tree Search adaptörleri, işçi süreci, LiteLLM yamaları |
| `docbenchtr/evaluate.py` | 311 satır | Metrikler, soru başına değerlendirme, özet |
| `docbenchtr/reporting.py`, `plotting.py` | 213 satır | README'nin üretilen bölümü ve grafik |
| `docbenchtr/run_pilot.py`, `rerun_failed.py` | 236 satır | Giriş noktaları, kilit dosyası, başarısız satırların yeniden koşulması |
| `tests/` | 215 satır | 12 birim testi |
| `colab/` | 2 defter | A100 donanım kontrolü ve güncel modeller |
| `results/` | 6 koşu | Soru başına çıktılar, özet, grafik, ortam kaydı |
| `data/`, `.pageindex/` | 3 PDF, 3 ağaç | Sorular, doğrulama kaydı, kaynak listesi, PageIndex ağaç önbelleği |
| `reports/pilot_ozet/` | HTML + 3 şekil | Türkçe kısa pilot raporu |

PageIndex çağrıları, Windows'taki iç içe süreç havuzu sorununu önlemek için ayrı bir işçi sürecinde çalışır ve LiteLLM üzerinden yerel Ollama'ya gider. Tree Search ise Ollama'nın `/api/chat` uç noktasını JSON şemasıyla doğrudan çağırır. Bağımlılıklar `pyproject.toml` ve `uv.lock` ile sabitlenmiştir (Python 3.12, PyTorch 2.7.1 + CUDA 12.6, sentence-transformers 5.4.1, rank-bm25 0.2.2, pageindex 0.4.0).

## 4. Bulgular

Yayımlanan sayılar ham verilerle tutarlı; zayıf halka sayıların kendisi değil, onları üreten soru seti ve onlardan çıkarılan başlık iddiası. Aşağıdaki tablo bulguları önem sırasıyla özetler; ayrıntılar alt bölümlerdedir.

| # | Bulgu | Önem | Konum |
| --- | --- | --- | --- |
| B1 | “PageIndex tüm klasik yöntemleri geçer” iddiası istatistiksel olarak desteklenmiyor (fark +0,125, %95 GA \[−0,028; +0,278\]) | Yüksek | README §Current models, Türkçe rapor |
| B2 | Span doğrulaması tanım gereği her soruyu geçirir; 4 soruda cevap soruyla uyuşmuyor | Yüksek | [`questions.py` L91–193](https://github.com/umitaltinozzz/docbenchtr/blob/97d7240a6afaf3af83a29a4c471841d611c62c2b/docbenchtr/questions.py#L91-L193) |
| B3 | Altın sayfalar eksik: 6502'de 5 sorunun cevabı yönetmeliklerde aynı ya da eşdeğer biçimde geçiyor | Yüksek | `data/pilot_questions.json` |
| B4 | KVKK soruları ayırt edici değil: “ilk 5 sayfayı döndür” Recall@5 = 1,000 alır | Orta | `data/pilot_questions.json` |
| B5 | 36 sorunun yalnızca 27 benzersiz hedefi var; distractor tipi ölçülmüyor | Orta | [`questions.py` L15–82](https://github.com/umitaltinozzz/docbenchtr/blob/97d7240a6afaf3af83a29a4c471841d611c62c2b/docbenchtr/questions.py#L15-L82) |
| B6 | Ajan sıralaması okuma sırasıdır, ilgi sırası değil; atıf yapılan sayfalar puanlanmıyor | Orta | [`retrievers.py` L1173–1247](https://github.com/umitaltinozzz/docbenchtr/blob/97d7240a6afaf3af83a29a4c471841d611c62c2b/docbenchtr/retrievers.py#L1173-L1247) |
| B7 | İki farklı puanlama kuralı (başarısız sorgu: dışlanan / sıfır) yan yana kullanılıyor | Orta | [`evaluate.py` L180–186](https://github.com/umitaltinozzz/docbenchtr/blob/97d7240a6afaf3af83a29a4c471841d611c62c2b/docbenchtr/evaluate.py#L180-L186) |
| B8 | Ortalama gecikme ısınma ve GPU paylaşımıyla çarpık | Orta | [`evaluate.py` L77–95](https://github.com/umitaltinozzz/docbenchtr/blob/97d7240a6afaf3af83a29a4c471841d611c62c2b/docbenchtr/evaluate.py#L77-L95) |
| B9 | Colab defterleri sabit commit yerine `main` dalını çekiyor; çıkarıcı sürümü kaydedilmiyor | Orta | `colab/*.ipynb` |
| B10 | Özel API'lere bağımlılık ve global monkeypatch'ler | Orta | [`retrievers.py` L595–916](https://github.com/umitaltinozzz/docbenchtr/blob/97d7240a6afaf3af83a29a4c471841d611c62c2b/docbenchtr/retrievers.py#L595-L916) |
| B11 | README iç çelişkileri ve tekrarlar | Düşük | [`README.md` L28, L194–196, L211](https://github.com/umitaltinozzz/docbenchtr/blob/97d7240a6afaf3af83a29a4c471841d611c62c2b/README.md#L28) |
| B12 | Test kapsamı dar, CI yok; 3 kullanılmayan import | Düşük | `tests/`, `docbenchtr/` |

### 4.1 Raporlanan sayıların doğruluğu

README'deki bütün sonuç tablolarının (pilot, soru tipi, donanım kontrolü, güncel modeller, belge bazında) tüm hücreleri `per_query.json` dosyalarından yeniden üretildi. Tek sapma yuvarlamadadır: gpt-oss 64k ajan (varsayılan) için README 0,791 yazıyor, doğrusu 28,5/36 = 0,792.

Altı koşunun hepsinde soru kimlikleri ve altın sayfalar `data/pilot_questions.json` ile birebir aynıdır. Bu yüzden koşular arası karşılaştırmalar etiket açısından geçerlidir.

**B7 — İki puanlama kuralı.** Kod, hata veren (`failed`) sorguları ortalamanın dışında tutar; `summary.json` ve üretilen tablo buna göre yazılır. Elle yazılan “Current models” tablosu ise bunları sıfır sayar. Fark küçük değil: qwen3.8 27B, 32k, ajan (varsayılan) için `summary.json` 0,913 (23 soru) der, README 0,583 (36 soru). README iki değeri de verdiği için yanıltıcı değil; ama yalnızca `summary.json`'u okuyan biri 0,913 görür. Bulgularda bu rapor “başarısız = 0” kuralını kullanır.

### 4.2 Soru seti ve altın etiketler

**B2 — Doğrulama tanım gereği geçer.** `draft_questions` cevap parçasını bir sayfanın metninden keser ve o sayfayı altın sayfa yapar. `verify_answer_spans` ise parçanın aynı sayfada geçip geçmediğine bakar. Sonuç: 36 taslağın 36'sı geçer, 0'ı elenir.

`_pick_span` içindeki yorum, terimlerle uyuşmayan parçanın doğrulayıcı tarafından eleneceğini söyler. Oysa iki dal da aynı `ranked[0]` değerini döndürür ve doğrulayıcı terimlere hiç bakmaz. README kontrolün yalnızca kaynağı (provenance) doğruladığını açıkça yazıyor; yani sorun gizlenmemiş, kod ile yorum arasında bir uyumsuzluk var. Planlanan terimlerden biri seçilen cevapta bulunmayan dört soru:

| Soru | Eksik terim | Sonuç |
| --- | --- | --- |
| `kvkk_6698-paraphrase-2` (m. 11) | “bilgi talep etme” | Cevap listenin ortasından, “e)” bendinden başlıyor |
| `yuksekogretim_2547-paraphrase-1` (m. 22) | “eğitim - öğretim” | Cevap dipnot metnine taşıyor: “…yayımlar yapmak, 23 22/2/2018 tarihli ve 7100 sayılı Kanunun…” |
| `yuksekogretim_2547-cross_section-1` (m. 13) | “üniversiteyi temsil” | Rektörün görevi yerine rektör yardımcısı seçimi |
| `yuksekogretim_2547-cross_section-2` (m. 22) | “eğitim - öğretim” | Aynı dipnotlu parça |

Ayrıca soru kökleri otomatik üretilmiyor: 36 soru `QUESTION_PLAN` içinde elle yazılmış, otomatik olan yalnızca cevap parçası ve sayfa seçimi. “Auto-drafted” etiketi bu yüzden tam doğru değil.

**B3 — Eksik altın sayfalar.** 6502 PDF'i kanunla birlikte ikincil mevzuatı da içerir (340 sayfa); tüm altın sayfalar 8–28 aralığında, yani kanun metnindedir. Cevap metinleri bütün sayfalarda arandığında üç hükmün yönetmeliklerde aynen ya da eşdeğer biçimde tekrarlandığı görüldü:

| Hüküm (kanun sayfası) | Tekrarlandığı sayfa | Etkilenen sorular |
| --- | --- | --- |
| Mesafeli sözleşme tanımı, m. 48 (s. 21) | s. 139, Mesafeli Sözleşmeler Yönetmeliği tanımları | direct\_fact-3, distractor-3, cross\_section-3 |
| Taksitli satışta cayma, m. 18 (s. 12) | s. 54 | paraphrase-1 |
| Fiyat etiketi, m. 54 (s. 26) | s. 188, Fiyat Etiketi Yönetmeliği m. 5 | paraphrase-2 |

Bu etiketler yöntemleri fiilen cezalandırıyor. Hybrid, `distractor-3`'te aynı tanımı birebir içeren s. 139'u birinci sıraya koyuyor ve 0 alıyor; Dense iki soruda aynı durumda. Büyük model koşularında (qwen3.8 27B, gpt-oss 20B) PageIndex varyantları da `cross_section-3`'te s. 147'yi (Finansal Hizmetlere İlişkin Mesafeli Sözleşmeler Yönetmeliği) getiriyor. Raporlanan bütün recall değerleri bu nedenle alt sınırdır. Aynı arama 2547 için yerel `pdftotext` çıktısıyla tekrarlandı; orada cevaplar yalnızca altın sayfalarda bulundu.

**B4 — Tavan etkisi.** KVKK 18 sayfadır ve 12 sorunun tüm altın sayfaları 2–5 aralığındadır. Belgenin ilk beş sayfasını sırayla döndüren bir “yöntem” KVKK'da Recall@5 = 1,000, 36 sorunun tamamında 0,361 alır. Dense, Hybrid ve 64k Tree Search KVKK'da zaten 1,000'dadır; soruların üçte biri yöntemleri ayırt etmiyor. Karşılaştırma için rastgele sıralamanın beklenen Recall@5 değeri KVKK'da 0,278, 2547'de 0,035, 6502'de 0,015'tür.

**B5 — Tekrar eden hedefler.** 36 sorunun 27 benzersiz (belge, sayfa) hedefi ve 29 benzersiz cevabı var; KVKK s. 3 tek başına dört sorunun hedefi. Dokuz distractor sorusunun yedisi bir direct\_fact veya paraphrase sorusuyla aynı cevabı paylaşıyor. `distractor_article` alanı kaydediliyor ama hiçbir metrikte kullanılmıyor; dolayısıyla çeldirici tipi, yöntemin yanlış maddeye kanıp kanmadığını ölçmüyor. Çapraz bölüm sorularında iki altın sayfa olduğundan Recall@1 en fazla 0,5 olabilir.

### 4.3 Değerlendirme metodolojisi

**B6 — Ajan “sıralaması” bir okuma sırasıdır.** PageIndex ajanı sayfa puanı döndürmez. Kod, başarılı `get_page_content` çağrılarında dönen sayfaları çağrı sırasıyla dizer ve bunu sıralama sayar; bu, ajanın keşif sırasıdır, ilgi sırası değil. Ajanın cevabında atıf yaptığı sayfalar (`cited_pages`) kaydedilir ama puanlanmaz. Bu sayfalarla puanlandığında qwen3.8 27B, 64k koşusunda Recall@5 varsayılan ajan için 0,833, read-pages için 0,861 olur; okuma sırasıyla bu değerler 0,833 ve 0,792'dir.

**Tree Search sıralaması.** Seçilen düğümün sayfa aralığı artan sırayla açılır; 7B koşusunda bir sıralama böyle 129 sayfaya uzuyor. Bu tasarım ağacın ayrıntı düzeyini metriğin parçası yapar: aynı doğru düğümü seçen iki model, düğüm kaba ise farklı skor alabilir.

**Farklı metin çıkarımı.** Klasik yöntemler Xpdf `pdftotext -layout` metnini, PageIndex ise kendi PDFium tabanlı çıkarımını kullanır. Aynı fiziksel sayfa iki yöntem ailesinde farklı metinle temsil edilir; bu, yöntem farkına karışan bir değişkendir.

**B8 — Gecikme ölçümü.** Ortalama gecikme ilk sorgunun ısınmasına ve ortam gürültüsüne duyarlı. qwen3.8 27B, 64k, varsayılan ajan: ilk sorgu 411 s, ortalama 47,2 s, medyan 37,7 s. Dizüstü koşusunda Dense'in ortalaması 652 ms iken aynı Dense çağrısını içeren Hybrid'inki 17,5 ms; Dense'in ilk sorgusu ise 10,6 ms. Her soruda Dense, bir önceki sorunun PageIndex çağrılarından kısa süre sonra çalışıyor; 8 GB GPU'nun Ollama ile paylaşılması en olası açıklama. A100'de Dense 14 ms ölçülüyor.

### 4.4 İstatistiksel geçerlilik ve başlık iddiası

**B1.** README ve Türkçe pilot raporu, 27B model ve 64k bağlamla PageIndex'in tüm klasik yöntemleri geçtiğini öne çıkarır (Tree Search Recall@5 0,917, Hybrid 0,792). Soru bazında eşleştirilmiş karşılaştırma bu cümleyi desteklemiyor.

&#91;embedded content: results/colab\_a100\_\*/per\_query.json · 5 koşu × 3 PageIndex varyantı, bu raporda yeniden hesaplandı\]

Sıfırın sağı PageIndex lehinedir. 7B ve 32k satırlarının çoğunda güven aralığı tamamen solda kalıyor; 64k satırlarının hepsinde sıfırı kesiyor.

- qwen3.8 27B, 64k Tree Search: +0,125, %95 GA \[−0,028; +0,278\]; 8 kazanç, 3 kayıp, 25 eşitlik; işaret testi p = 0,23.
- 7B Tree Search: −0,347 \[−0,542; −0,153\], p = 0,007. 7B ve 32k yapılandırmalarının dokuzda yedisinde güven aralığı tamamen sıfırın solunda.

B2 ve B3'te işaretlenen 9 soru çıkarıldığında fark +0,130 \[−0,019; +0,296\] olur (5 kazanç, 1 kayıp, p = 0,22); sonuç değişmez. qwen'in 8 kazancından ikisi (6502 paraphrase-2 ve distractor-3), Hybrid'in eşdeğer yönetmelik sayfasını getirip 0 aldığı sorulardır.

Güven aralıkları soruları bağımsız varsayar; 36 soru yalnızca 27 hedef paylaştığı için gerçek belirsizlik daha geniştir. Her yapılandırma tek kez koşulmuştur. Aynı 7B model ve ayarla dizüstü ile A100 arasında ilk beş sayfa BM25'te 36/36, Hybrid'de 33/36 soruda aynı kalırken Tree Search'te yalnızca 14/36, ajanlarda 16/36 ve 20/36 soruda aynı kalıyor.

Savunulabilir cümle şudur: bu pilotta 64k bağlamlı 27B modelle PageIndex Tree Search, Hybrid'den sayısal olarak yüksek ama istatistiksel olarak ayırt edilemeyen bir skor aldı.

### 4.5 Kod kalitesi ve testler

Kod genel olarak dikkatli yazılmış. Kaynak PDF'ler SHA-256 ile sabitlenmiş; her sorgu için hata, iz ve token sayıları saklanıyor; attempted / scored / zero ranking / excluded sayımları açıkça raporlanıyor; eşzamanlı koşulara karşı bir kilit dosyası var.

**B10 — Kırılgan entegrasyon.** `retrievers.py` (1.247 satır) klasik yöntemleri, PageIndex işçi sürecini ve kütüphane yamalarını tek modülde tutar.

- Kod çalışma anında `litellm.completion`/`acompletion`, `agents.Runner.run_streamed` ve `pageindex.flash.api.extract_toc` fonksiyonlarını değiştirir.
- `LocalAPI._extract_page_texts`, `client._api._store` ve `pageindex.local_chat._openai_usage` gibi özel API'lere dayanır. `pageindex==0.4.0` sabit kaldıkça çalışır; bir sürüm yükseltmesi sessizce kırabilir.
- `PageIndexLLMIdleRetriesExhausted`, PageIndex'in kendi yeniden deneme döngüsünü durdurmak için sahte bir `status_code = 401` taşır ([L702–707](https://github.com/umitaltinozzz/docbenchtr/blob/97d7240a6afaf3af83a29a4c471841d611c62c2b/docbenchtr/retrievers.py#L702-L707)). Çalışan ama belgelenmesi gereken bir hile.
- `_pick_span` içinde iki dal aynı değeri döndürür ([L190–193](https://github.com/umitaltinozzz/docbenchtr/blob/97d7240a6afaf3af83a29a4c471841d611c62c2b/docbenchtr/questions.py#L190-L193)).
- pyflakes üç kullanılmayan import buldu: `documents.py` (`re`), `evaluate.py` (`Path`), `run_pilot.py` (`json`). `ROOT_PAGEINDEX_DIR` bir fonksiyon ama sabit gibi adlandırılmış; `extract_pdf_pages` içinde `page_count == 0` kontrolü iki kez yapılıyor; gömme modeli adı `summarize` içinde yapılandırma sabiti yerine elle yazılmış.

**B12 — Testler.** 12 birim testinin 12'si geçiyor. Kapsanan: metrik hesabı, PageIndex sıralama türetimi, Tree Search ayrıştırma, özetleme kuralı, span doğrulayıcı ve yapılandırma. Kapsanmayan: PDF metin çıkarımı, `ARTICLE_RE` ile madde ayrıştırma, soru taslağı üretimi, BM25/Dense/Hybrid, README üretimi ve `rerun_failed`. B2'deki dört hatalı soru tam da test edilmeyen kısımdan çıkıyor. Depoda CI tanımı yok.

**Güvenlik.** Kod ağda yalnızca sabitlenmiş kaynak URL'lerine ve yerel Ollama'ya (127.0.0.1) gider. Alt süreçler kabuk kullanmadan liste argümanlarıyla çağrılır ve depoda gizli anahtar yoktur. Colab defterleri Ollama'yı `curl | sh` ile kurar; sürüm sabitlenmiş ama betik doğrulanmıyor.

### 4.6 Tekrarlanabilirlik ve dokümantasyon

Tekrarlanabilirlik altyapısı bir pilot için güçlü: `uv.lock`, sabit paket sürümleri, gömme modeli revizyonu, PDF SHA-256'ları, PageIndex önbelleğinin sayfa metniyle doğrulanması ve her Colab koşusu için GPU, commit ve Modelfile kaydı. Donanım kontrolü ile düzeltme geçmişinin README'de açıkça yazılması örnek bir şeffaflık.

**B9 — Colab koşuları.**

- Defterler sabit commit yerine `main` dalını klonlayıp `pull` ediyor. Yayımlanan A100 sonuçları üç farklı commit'te (`c90cfd4`, `4d54c57`, `9cae8ff`) üretilmiş, incelenen HEAD ise `97d7240`. Defter yeniden çalıştırıldığında hangi kodun koşacağı garanti değil.
- Defterler yalnızca `results/` klasörünü kopyalıyor; Colab'daki `pdftotext` sürümü ve yeniden üretilen `data/sources.json` saklanmıyor. BM25 tam sıralamaları dizüstü ile A100 arasında 14/36 soruda (11. sıradan sonra) farklılaşıyor; nedeni kayıtlardan anlaşılamıyor.

**B11 — README ve rapor.**

- 28. satır kaynak PDF'lerin Git dışında tutulduğunu, 211. satır ise PDF'lerin ve PageIndex önbelleğinin commit edildiğini söylüyor. `.gitignore` `data/raw/` ve `.pageindex/` klasörlerini listeliyor ama ikisi de izleniyor; `download_documents` belge dizgisi hâlâ “never commits PDFs” diyor.
- Sınırlılıklar bölümündeki bir madde (L194 ve L196) iki kez yazılmış; bazı sayılar yazıyla yazılmış (“Python three point twelve”).
- Türkçe pilot raporu, TRLawBench sonuçlarına (59 ve 83 puan) kaynak bağlantısı vermeden atıf yapıyor.
- KVKK PDF'i resmî kaynaktan değil, Wikimedia Commons'taki 2020 tarihli arşiv kopyasından indiriliyor; README bunu açıkça belirtiyor.

**Lisans.** Kod MIT lisanslıdır; kanun metinleri 5846 sayılı FSEK m. 31 kapsamında serbestçe kullanılabilir. Depo bu ayrımı doğru yapıyor.

## 5. Tartışma

DocBenchTR'nin değeri altyapısında; zayıflığı, ölçüm aracının (soru seti ve altın etiketler) henüz yöntem farkını taşıyacak kadar keskin olmamasında.

**Güçlü yönler**

- Türkçe hukuk metinlerinde erişim gibi az çalışılmış bir boşluğu hedefliyor.
- Şeffaflık yüksek: geçersiz sonuçlar geri çekilip düzeltme geçmişi yazılmış, her sorgunun izi saklanmış, başarısız sorgular gizlenmemiş.
- Donanım kontrolü iyi kurgulanmış bir deney: aynı model ve aynı ağaçlarla farklı GPU. GPU'nun sonucu değil hızı değiştirdiğini ve PageIndex çıktılarının sıcaklık 0'da bile kararlı olmadığını göstermesi tek başına değerli bir bulgu.
- Tekrarlanabilirlik altyapısı (kilit dosyası, SHA-256, önbellek doğrulama, ortam kayıtları) bir pilot için beklenenin üstünde.

**Zayıf yönler**

- 36 soru 27 hedefe düşüyor; KVKK soruları tavanda; 6502'de altın sayfalar eksik; çeldirici tipi ölçülmüyor.
- “Span-verified” etiketi, soru kalitesine dair bir güvence izlenimi veriyor ama yalnızca cevabın o sayfadan kesildiğini kanıtlıyor.
- Başlık iddiası, verinin desteklediğinden güçlü.

**Geçerlilik tehditleri**

| Tehdit | Açıklama | Muhtemel etki |
| --- | --- | --- |
| Farklı metin çıkarımı | Klasik yöntemler Xpdf, PageIndex PDFium metni kullanıyor | Yöntem farkına karışır; yönü belirsiz |
| Ajan sıralaması = okuma sırası | İlgi sıralaması değil; atıflar puanlanmıyor | Yönü belirsiz: atıflarla puanlama 7B'de Recall@5'i 0,083'e düşürür, 27B 64k'da korur ya da artırır (0,833; 0,861) |
| Eksik altın sayfalar | 6502'de 5 soru, yönetmelik tekrarları | Tüm recall değerleri alt sınır; iki aileyi de vuruyor |
| Küçük ve bağımlı örneklem | 36 soru, 27 hedef, tek koşu | Birkaç yüzdelik farklar gürültü |
| Model ile yöntemin karışması | PageIndex büyük bir LLM kullanıyor, klasik yöntemler kullanmıyor | Kazancın yöntemden mi modelden mi geldiği ayrılamıyor |
| Dar kapsam | 3 belge, tek alan, tek gömme modeli; KVKK 2020 metni | Sonuçlar başka mevzuata genellenemez |
| Ağaçlar 7B ile kurulmuş | 27B ve 20B modeller 7B'nin özetleriyle çalışıyor | Büyük modelleri dezavantajlı bırakır (README de belirtiyor) |

Model ile yöntemin karışması en önemli açık soru. Yazarın Türkçe raporunda önerdiği kontrol, yani Hybrid'in ilk 20 sonucunu aynı LLM ile yeniden sıralamak, bu soruyu doğrudan cevaplar.

## 6. Öneriler

Öneriler önceliğe göre sıralanmıştır; parantez içindeki kodlar 4. bölümdeki bulgulara karşılık gelir.

**Öncelik 1 — sonuçları yeniden yayımlamadan önce**

- [ ] Başlık iddiasını “istatistiksel olarak ayırt edilemeyen” diye yumuşat; eşleştirilmiş güven aralığını ve işaret testini tabloya ekle (B1).
- [ ] Altın sayfaları tamamla: her cevabı belgenin tamamında ara, eşdeğer sayfaları ekle ya da kanun metnini ayrı bir alt küme olarak değerlendir (B3).
- [ ] Doğrulayıcıyı gerçek bir kontrole dönüştür: plan terimlerinin seçilen parçada geçmesini şart koş, dipnot ve sayfa numarası gürültüsünü temizle, `_pick_span`'deki ölü dalı kaldır (B2).
- [ ] Tek puanlama kuralı kullan: `summary.json`'a “başarısız = 0” değerlerini de yaz; README ve Türkçe rapor aynı kuralı izlesin (B7).

**Öncelik 2 — tam çalışmanın tasarımında**

- [ ] Soru setini yeniden tasarla: benzersiz hedefler, belge içinde dengeli sayfa dağılımı; “ilk 5 sayfa” ve rastgele taban çizgilerini tabloya koy (B4, B5).
- [ ] Çeldirici tipini gerçekten ölç: `distractor_article` sayfasının hedeften önce gelip gelmediğini ayrı bir metrikle raporla (B5).
- [ ] Ajan için atıf yapılan sayfalarla (`cited_pages`) puanlama ekle; okuma sırasını ek bilgi olarak ver (B6).
- [ ] Adâlet kontrolü ekle: Hybrid'in ilk 20 sonucunu aynı LLM ile yeniden sırala; bütün yöntemler aynı metin çıkarımını kullansın.
- [ ] Her PageIndex yapılandırmasını en az üç kez koş; ortalama, sapma ve güven aralığı ver (B1).
- [ ] Gecikmeyi ısıtılmış ve yöntem başına ayrı koşularda ölç; medyan ve p90 raporla (B8).

**Öncelik 3 — mühendislik**

- [ ] Colab defterlerinde commit'i sabitle; `pdftotext` sürümünü ve `data/*.json` dosyalarını sonuçlarla birlikte sakla (B9).
- [ ] `retrievers.py`'yi klasik yöntemler, PageIndex işçisi ve yamalar olarak böl; özel API kullanımını tek bir adaptör katmanında topla ve `pageindex` sürümünü çalışma anında kontrol et (B10).
- [ ] Madde ayrıştırma, soru üretimi ve klasik yöntemler için test yaz; GitHub Actions ile pytest ve bir linter çalıştır (B12).
- [ ] README çelişkilerini ve tekrarı düzelt, `.gitignore`'u izlenen dosyalarla uyumlu hale getir, TRLawBench atıfına kaynak ekle (B11).

## 7. Sonuç

DocBenchTR, Türkçe mevzuatta erişim yöntemlerini karşılaştırmak için şeffaf ve tekrarlanabilir bir pilot altyapı sunuyor; raporlanan her sayı ham verilerle tutarlı. Buna karşın mevcut soru seti ve altın etiketler yöntemler arasındaki farkı güvenilir biçimde ölçecek nitelikte değil.

Verinin desteklediği sonuç iki cümledir. 7B modelle PageIndex, klasik yöntemlerin açık ve anlamlı biçimde gerisinde kalıyor. 27B model ve 64k bağlamla ise Hybrid ile başa baş; “geçer” demek için daha büyük, temiz etiketli bir soru seti ve tekrarlı koşular gerekiyor.

Öncelik 1'deki dört madde, tam çalışmaya (yaklaşık 1.000 insan incelemeli soru) geçmeden önce en yüksek getiriyi sağlayacak adımlardır.
