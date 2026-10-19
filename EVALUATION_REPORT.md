# İkinci El Araç Fiyat Tahmini: Model Değerlendirme ve Test Raporu

Bu rapor, eğitilen **XGBoost Regresyon** modelinin test seti (`802` araç) üzerindeki genel performansını görselleştirmekte ve farklı segmentlerden (sedan, SUV, spor, elektrikli, kamyonet, lüks ve ekonomik) seçilen araçlar üzerinden **Gerçek Fiyat vs Model Tahmini** karşılaştırmalarını sunmaktadır.

---

## 1. Eğitim ve Çapraz Doğrulama (Train CV) Sonuçları

Veri seti (`4009` satır) `%80` eğitim (`3207` satır) ve `%20` test (`802` satır) olarak ayrılmıştır (`reports/cv_results.csv`):

| Model | Train CV MAE (\$) | CV Standart Sapma (\$) | Durum |
|---|---:|---:|---|
| **XGBoost** | **\$8,352** | ± \$948 | **Seçilen Model** |
| Random Forest | \$8,865 | ± \$968 | Aday |
| SVR | \$9,756 | ± \$985 | Aday |
| Ridge | \$11,419 | ± \$968 | Aday |
| Baseline (Medyan) | \$25,601 | ± \$1,586 | Referans |

---

## 2. Görsel Değerlendirme Paneli

Aşağıdaki panoda model karşılaştırma metrikleri, log ölçeğindeki gerçek-tahmin uyumu, yüzde hata dağılımı ve araç yaşına göre hata oranları yer almaktadır:

![Model Değerlendirme Paneli](reports/evaluation_dashboard.png)

### Test Seti Metrik Özeti (`reports/metrics.json`)

| Metrik | Seçilen Model (XGBoost) | Baseline (Medyan) | Yorum |
|---|---:|---:|---|
| **$R^2$ (Log-Fiyat Uzayı)** | **0.8408** | -0.0036 | Log-fiyat varyansının **%84.1**'i açıklanmaktadır. |
| **Medyan APE (Tipik Yüzde Hata)** | **%13.21** | %47.70 | Araçların yarısında hata **%13.2'nin altındadır**. |
| **MAPE (Ortalama Yüzde Hata)** | **%20.65** | %90.84 | Baseline'a göre >4 kat iyileşme. |
| **%90 Tahmin Aralığı Kapsaması** | **%89.78** | — | Hedeflenen %90 kalibrasyon ile birebir uyumlu. |
| **MAE / $R^2$ (Tüm Test Seti - Dolar)** | \$15,528 / 0.110 | \$32,866 / -0.018 | Test setindeki 3 adet >\$1.5M uç kayıttan etkilenmektedir. |
| **MAE / $R^2$ (<\$1M Araçlar, 799/802 satır)** | **\$7,763 / 0.788** | — | 3 uç/hatalı ilan hariç tutulduğunda CV ile tam tutarlıdır. |

---

## 3. SHAP (XAI) Açıklanabilirlik Görselleştirmesi

Modelin fiyat kararlarında hangi özelliklerin ne yönde etki ettiğini gösteren SHAP özet (summary) dağılımı:

![SHAP Özet Dağılımı](reports/shap_summary.png)

### İlk 10 Özellik (`mean |SHAP|`, Log-Fiyat Katkısı)

| Sıra | Özellik | Ortalama \|SHAP\| | Etki Yorumu |
|:---:|---|---:|---|
| 1 | `mileage` | `0.2901` | Kilometre (mil) arttıkça fiyatı en güçlü düşüren faktör |
| 2 | `age` | `0.2272` | Araç yaşı arttıkça değer kaybı |
| 3 | `horsepower` | `0.1287` | Yüksek beygir gücü fiyatı yukarı çeker |
| 4 | `engine_volume` | `0.0941` | Motor hacmi etkisi |
| 5 | `model` | `0.0868` | Model bazlı hedef kodlama (TargetEncoder) |
| 6 | `brand` | `0.0866` | Marka prestiji hedef kodlaması |
| 7 | `cylinders` | `0.0159` | Silindir sayısı katkısı |
| 8 | `accident_reported` | `0.0092` | Hasar/kaza kaydı varlığı fiyatı düşürür |
| 9 | `fuel_category_diesel` | `0.0085` | Dizel motor tipi primi |
| 10 | `missingindicator_horsepower` | `0.0084` | Beygir gücü bilgisinin ilanda yer almaması |

---

## 4. Karşılaştırmalı Testler: Farklı Segmentlerden 9 Örnek Araç

Modelin eğitimde hiç görmediği test setinden seçilen, farklı marka, yaş, yakıt türü ve kilometreye sahip 9 temsilci araç üzerinde yapılan test sonuçları:

| # | Araç Segmenti | Marka / Model | Yaş | Mil | Gerçek Fiyat | Model Tahmini | %90 Güven Aralığı | Mutlak Hata (\$) | Yüzde Hata (%) | Aralıkta mı? |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | Lüks Cabriolet | **Mercedes-Benz SL 550** | 13 | 24,933 | **\$40,250** | **\$40,046** | \$25,837 – \$62,069 | **\$204** | **%0.51** | Evet |
| **2** | Aile / SUV | **Toyota Sequoia Limited** | 20 | 190,656 | **\$8,900** | **\$9,483** | \$6,118 – \$14,699 | **\$583** | **%6.55** | Evet |
| **3** | Minivan / Aile | **Honda Odyssey EX-L** | 4 | 18,385 | **\$37,950** | **\$34,072** | \$21,983 – \$52,810 | **\$3,878** | **%10.22** | Evet |
| **4** | Spor / İkonik | **Porsche 911 Carrera 4S** | 17 | 25,500 | **\$79,995** | **\$70,010** | \$45,170 – \$108,510 | **\$9,985** | **%12.48** | Evet |
| **5** | Performans SUV | **BMW X5 M Base** | 3 | 4,348 | **\$105,900** | **\$82,944** | \$53,515 – \$128,557 | **\$22,956** | **%21.68** | Evet |
| **6** | Kamyonet (Truck) | **Ford F-150 XLT** | 6 | 92,000 | **\$29,999** | **\$37,935** | \$24,475 – \$58,797 | **\$7,936** | **%26.45** | Evet |
| **7** | Elektrikli (EV) | **Tesla Model 3 Std Range+** | 5 | 21,600 | **\$31,000** | **\$40,764** | \$26,300 – \$63,182 | **\$9,764** | **%31.50** | Evet |
| **8** | Eski Sedan | **Chevrolet Impala Base** | 22 | 64,500 | **\$5,900** | **\$10,214** | \$6,590 – \$15,831 | **\$4,314** | **%73.12** | Hayır |
| **9** | Yüksek Mil Ekonomik | **Dodge Avenger SE** | 12 | 124,705 | **\$6,499** | **\$13,560** | \$8,749 – \$21,018 | **\$7,061** | **%108.65** | Hayır |

### Sonuçların Değerlendirilmesi
- **Ana Piyasa (\$15,000 – \$100,000):** Model piyasanın ana gövdesinde yüksek doğrulukla çalışmaktadır (`Mercedes-Benz SL 550`'de yalnızca **\$204 / %0.51** sapma, `Honda Odyssey` ve `Porsche 911`'de **%10–12** sapma).
- **Düşük Fiyatlı Eski Araçlar (< \$7,000):** `Chevrolet Impala` (\$5,900) ve `Dodge Avenger` (\$6,499) gibi araçlarda mutlak dolar hatası \$4k–\$7k seviyesinde olsa da taban fiyat düşük olduğu için yüzde hata yüksek görünmektedir. Bu segmentte mekanik/kozmetik kondisyon belirleyicidir.
- **Uç Değerler (>\$1.5M):** Test setindeki 21 yaşındaki `2005 Maserati Quattroporte Base` aracın veri setinde hatalı şekilde **\$2,954,083** olarak yer alması (modelin tahmini gerçekçi biçimde **\$22,975**'tir) ve iki adet tekil hiper aracın (`Bugatti Veyron`, `Porsche Carrera GT`) bulunması ham dolar $R^2$ değerini aşağı çekmektedir; bu 3 kayıt dışarıda bırakıldığında test dolar $R^2$ değeri **0.788**, MAE ise **\$7,763** olmaktadır.
