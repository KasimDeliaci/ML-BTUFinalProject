# İhracat Maliyet Analizi ve Anomali Tespiti Projesi

## İçindekiler

- [1. Proje Genel Bakış](#1-proje-genel-bakış)
- [2. Özellikler](#2-özellikler)
- [3. Gereksinimler](#3-gereksinimler)
- [4. Projenin Akışı](#4-proje-yapısı)
- [Görselleştirme](#Görselleştirme)
- [5. Veri Ön İşleme](#5-veri-ön-işleme)
- [6. Modeller](#6-modeller)
    - [6.1. Denetimli (Supervised) Öğrenme (Regresyon)](#61-denetimli-supervised-öğrenme-regresyon)
        - [6.1.1. Lineer Regresyon](#611-lineer-regresyon)
        - [6.1.2. Lasso Regresyonu](#612-lasso-regresyonu)
        - [6.1.3. KNN Regresyonu](#613-knn-regresyonu)
        - [6.1.4. Random Forest Regresyonu](#614-random-forest-regresyonu)
        - [6.1.5. XGBoost Regresyonu](#615-xgboost-regresyonu)
    - [6.2. Denetimsiz (Unsupervised) Öğrenme (Anomali Tespiti)](#62-denetimsiz-unsupervised-öğrenme-anomali-tespiti)
        - [6.2.1. K-Means Kümeleme](#621-k-means-kümeleme)
        - [6.2.2. Hiyerarşik Kümeleme](#622-hiyerarşik-kümeleme)
        - [6.2.3. Isolation Forest](#623-isolation-forest)
- [7. Sonuçlar](#7-sonuçlar)
- [8. Kurulum ve Kullanım](#8-kurulum-ve-kullanım)
- [9. Katkıda Bulunma](#9-katkıda-bulunma)

## 1. Proje Genel Bakış

Bu projede kullanılan veri seti, ATEZ YAZILIM TEKNOLOJİLERİ A.Ş.'nin sağlamış olduğu ham gümrük/ihracat verilerinden oluşturulmuştur. Veriler, başlangıçta XML formatında olup, her bir dosya farklı bir ihracat işlemine ait bilgileri içermektedir. Bu karmaşık ve yapılandırılmamış verileri analiz etmek için, pandas'ın read_xml() fonksiyonu katmanlı ve iç içe geçmiş yapıları okuyamadığı için csv dosyası oluşturmak için önce kendi parser fonksiyonumu oluşturarak her bir dosyayı parse edip ardından pandas kütüphanesindeki read_xml() fonksiyonunu kullanabildim. Her bir dosyası okuyup oluşturduğum csv dosyasına satır olarak kaydettim ve file_name değişkeni ile hangi dosya olduğunu tuttum.

Bu  veri işleme süreci, ham XML verilerinin makine öğrenmesi algoritmaları tarafından kullanılabilir hale getirilmesini sağlamıştır. Proje, bu veriler üzerinde uygulanan analizler ve elde edilen sonuçlar ile ihracat maliyetlerinin tahmin edilmesi ve anormalliklerin tespit edilmesi konularında değerli bilgiler sunmaktadır.

Desteklerinden ötürü kendilerine teşekkür ediyorum. Projede bolca pratik yapmak istedim bu yüzden hem *Supervised Learning* hem de *Unsupervised Learning* problemleri inceledim.

|**Sütun Adı**|**Veri Tipi**|**Açıklama**|
|:---|:---|:---|
|`Gonderici_ulke_kodu`|Kategorik|Gönderici firmanın ülke kodu|
|`Gonderici_sehir`|Kategorik|Gönderici firmanın şehir ismi|
|`Gonderici_posta_kodu`|Kategorik|Gönderici firmanın posta kodu|
|`Alici_ulke_kodu`|Kategorik|Alıcı firmanın ülke kodu|
|`Alici_sehir`|Kategorik|Alıcı firmanın şehir ismi|
|...|...|...|
|`Sigorta_miktarinin_dovizi`|Kategorik|Sigorta miktarının para birimi|
|`Toplam_sigorta_bedeli`|Sayısal|Toplam sigorta bedeli|
|`Toplam_sigorta_dovizi`|Kategorik|Toplam sigorta bedelinin para birimi|

### 1.1. Maliyet Tahmini (Supervised Learning)

- **Amaç**: İhracat işlemlerinin toplam maliyetini, çeşitli faktörleri göz önünde bulundurarak doğru bir şekilde tahmin etmek. Bu, ihracatçılara işlem maliyetleri konusunda daha fazla görünürlük ve kontrol sağlayacaktır.

- **Kullanılan Veriler**:

    - **Kap adedi ve ağırlık bilgileri**: Ürünlerin hacmi ve ağırlığı, nakliye maliyetlerini doğrudan etkileyen faktörlerdir.
    - **Nakliye ve sigorta maliyetleri**: Bu maliyetler, toplam ihracat maliyetinin önemli bir bölümünü oluşturur.
    - **Yurt içi/dışı harcamalar**: Gümrük vergileri, liman ücretleri, depolama maliyetleri gibi yurt içi ve yurt dışı harcamalar da toplam maliyeti etkiler.

- **İş Değeri**:

    - **İhracatçıların maliyet planlaması yapabilmesi**: Doğru maliyet tahminleri, ihracatçıların daha gerçekçi bütçeler oluşturmasına ve kaynaklarını daha verimli kullanmasına olanak tanır.
    - **Fiyatlandırma stratejilerinin geliştirilmesi**: Maliyet tahminleri, ihracatçıların ürünlerini rekabetçi bir şekilde fiyatlandırmasına yardımcı olur.
    - **Bütçe tahminlerinin iyileştirilmesi**: Daha doğru maliyet tahminleri, işletmelerin genel bütçe planlamasını iyileştirir.
    - **Operasyonel verimliliğin artırılması**: Maliyetleri etkileyen faktörlerin anlaşılması, işletmelerin operasyonel süreçlerini optimize etmelerine ve verimliliği artırmalarına yardımcı olur.

### 1.2. Anomali Tespiti (Unsupervised Learning)

- **Amaç**: Veri setindeki olağandışı ve beklenmedik desenleri (anomalileri) tespit etmek. Bu anomaliler, hatalı veri girişleri, dolandırıcılık girişimleri veya süreçteki verimsizlikleri işaret edebilir.

- **Kullanım Alanları**:

    - **Hatalı veri girişlerinin tespiti**: Manuel veri girişlerindeki hatalar veya sistem hataları tespit edilebilir.
    - **Potansiyel dolandırıcılık vakalarının belirlenmesi**: Olağandışı işlem desenleri, potansiyel dolandırıcılık faaliyetlerini ortaya çıkarabilir.
    - **Operasyonel anormalliklerin saptanması**: Süreçlerdeki beklenmedik değişiklikler veya aksaklıklar tespit edilebilir.
    - **Risk yönetimi ve kalite kontrol**: Anomalilerin tespiti, riskleri azaltmaya ve kaliteyi artırmaya yardımcı olur.

- **Metodoloji**:

    - **Kümeleme analizi ile benzer işlemlerin gruplandırılması**: Benzer işlemler gruplandırılarak, aykırı değerler daha kolay tespit edilebilir.
    - **İzolasyon ormanı ile aykırı değerlerin tespiti**: İzolasyon ormanı algoritması, anormallikleri normal veri noktalarından izole ederek tespit eder.
    - **Hiyerarşik kümeleme ile yapısal anomalilerin belirlenmesi**: Hiyerarşik kümeleme, veriler arasındaki hiyerarşik ilişkileri analiz ederek daha karmaşık anomalileri tespit edebilir.

### Proje Çıktıları

1. **Tahmin Modeli**:

- Yeni ihracat işlemlerinin maliyet tahminini yapabilen bir makine öğrenmesi modeli.
- Modelin performansını değerlendirmek için kullanılan metrikler (RMSE, R², MAE gibi) ve farklı modellerin karşılaştırılması.
- Modelin tahminlerini hangi faktörlere dayandırdığını anlamak için özellik önem dereceleri ve model yorumlanabilirliği.

2. **Anomali Tespit Sistemi**:

- Veri setindeki anomalileri otomatik olarak tespit eden bir sistem.
- Anomalileri derecelendirmek ve önceliklendirmek için kümeleme bazlı anomali skorlaması.
- Anomalileri görselleştirmek ve analiz etmek için görsel analiz araçları ve raporlama mekanizması.

### Teknik Detaylar

- **Veri Boyutu**:
    - Satır sayısı: 586
    - Değişken sayısı: 67

- **Performans Metrikleri**:
    - **Maliyet Tahmini - Supervied**: RMSE, R² ve MAE (Regresyon)
    - **Anomali Tespiti - Unsupervised**: Silhouette skoru ve küme kalite metrikleri (Clustering)

- **Kullanılan Teknolojiler**:
    - Python 3.x
    - Scikit-learn
    - XGBoost
    - Pandas & NumPy
    - Matplotlib & Seaborn

## 2. Özellikler

Bu projede aşağıdaki özellikler bulunmaktadır:

- **Veri Temizleme:** Eksik verilerin işlenmesi, aykırı değerlerin tespiti ve düzeltilmesi işbilgisine dayalı sıfırların analiz edilmesi.
- **Özellik Mühendisliği:** Yeni ve bilgilendirici özellikler oluşturulması.
- **Model Eğitimi:** Farklı makine öğrenmesi modellerinin eğitilmesi ve karşılaştırılması.
- **Anomali Tespiti:** Anormal veri noktalarının tespiti ve analizi.
- **Görselleştirme:** Verilerin ve sonuçların görselleştirilmesi.

## 3. Gereksinimler

Bu projeyi çalıştırmak için aşağıdaki gereksinimler vardır:

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn

## 4. Projenin Akışı

Bu projede izlenen adımlar genel olarak aşağıdaki gibidir:

1.  **Veri Yükleme:** `kasim_df.csv` dosyası Pandas kütüphanesi kullanılarak yüklenir. (`pd.read_csv('kasim_df.csv')`)

2.  **Veri İnceleme ve Temizleme:**
    -   `check_df()` fonksiyonu ile verinin genel özellikleri (boyut, veri tipleri, eksik değerler, vb.) incelenir.
    -   `delete_columns_with_high_missing_ratio()` fonksiyonu ile eksik değer oranı yüksek olan sütunlar silinir.
    -   Kategorik değişkenlerdeki eksik değerler mod ile doldurulur.
    -   `check_zero_ratio()` fonksiyonu ile sıfır değerlerinin oranı analiz edilir ve belirli bir oranın üzerinde sıfır değeri içeren sütunlar silinir.

3.  **Değişken Tanımlama:**
    -   `identify_columns()` fonksiyonu ile kategorik, numerik ve kardinal değişkenler tanımlanır.

4.  **Görselleştirme:**
    -   `plot_histogram_scatter()` fonksiyonu ile numerik değişkenlerin histogram ve scatter plotları çizdirilir.
    -   `plot_categorical_distributions()` fonksiyonu ile kategorik değişkenlerin dağılımları görselleştirilir.

5.  **Aykırı Değer Analizi:**
    -   `handle_outliers()` fonksiyonu ile aykırı değerler tespit edilir ve sınırlandırma (capping) yöntemi ile işlenir.

6.  **Özellik Mühendisliği:**
    -   Toplam maliyet, paket başına maliyet, paket başına ağırlık gibi yeni özellikler oluşturulur.

7.  **Hedef Değişkene Göre Kodlama:**
    -   `target_encode()` fonksiyonu ile kategorik değişkenler hedef değişkene göre ortalama kodlama yöntemi ile sayısal değerlere dönüştürülür.

8.  **Korelasyon Analizi ve Özellik Seçimi:**
    -   `correlation_heatmap()` fonksiyonu ile değişkenler arasındaki korelasyon görselleştirilir.
    -   `drop_correlated_features()` fonksiyonu ile yüksek korelasyona sahip değişkenler veri setinden çıkarılır.

9.  **Özellik Ölçeklendirme:**
    -   `StandardScaler()` kullanılarak özellikler standartlaştırılır.

10.  **Modelleme:**
    -   Denetimli öğrenme (regresyon) için `train_evaluate_linearRegression()`, `train_evaluate_LassoRegression()`, `train_evaluate_KNNRegressor()`, `train_evaluate_RandomForest_with_CV()`, `train_evaluate_XGBoost()` fonksiyonları kullanılarak farklı modeller eğitilir ve değerlendirilir.
    -   Denetimsiz öğrenme (anomali tespiti) için `train_evaluate_KMeans()`, `train_evaluate_HierarchicalKMeans()`, `train_evaluate_IsolationForest()` fonksiyonları kullanılarak farklı modeller eğitilir ve değerlendirilir.

---

## Görselleştirme 

<div align="center">

| | |
|---|---|
| <img width=400 src="https://github.com/KasimDeliaci/ML-BTUFinalProject/blob/main/img/toplam-yurt-disi-harcamalar.png"> | <img width=400 src="https://github.com/KasimDeliaci/ML-BTUFinalProject/blob/main/img/toplam-yurt-ici-harcamalar.png"> |
| <img width=400 src="https://github.com/KasimDeliaci/ML-BTUFinalProject/blob/main/img/toplamfatura.png"> | <img width=400 src="https://github.com/KasimDeliaci/ML-BTUFinalProject/blob/main/img/yurtici-diger.png"> |
| <img width=400 src="https://github.com/KasimDeliaci/ML-BTUFinalProject/blob/main/img/brut-agirlik.png"> | <img width=400 src="https://github.com/KasimDeliaci/ML-BTUFinalProject/blob/main/img/toplam-sigorta.png"> |
| <img width=400 src="https://github.com/KasimDeliaci/ML-BTUFinalProject/blob/main/img/toplam-navlun.png"> | <img width=400 src="https://github.com/KasimDeliaci/ML-BTUFinalProject/blob/main/img/sigorta.png"> |
| <img width=400 src="https://github.com/KasimDeliaci/ML-BTUFinalProject/blob/main/img/sigorta-miktari.png"> | <img width=400 src="https://github.com/KasimDeliaci/ML-BTUFinalProject/blob/main/img/net-agirlik.png"> |
| <img width=400 src="https://github.com/KasimDeliaci/ML-BTUFinalProject/blob/main/img/navlun-miktari.png"> | <img width=400 src="https://github.com/KasimDeliaci/ML-BTUFinalProject/blob/main/img/nakliye.png"> |
| <img width=400 src="https://github.com/KasimDeliaci/ML-BTUFinalProject/blob/main/img/kap-adedi.png"> | <img width=400 src="https://github.com/KasimDeliaci/ML-BTUFinalProject/blob/main/img/istatistiki-miktar.png"> |
| <img width=400 src="https://github.com/KasimDeliaci/ML-BTUFinalProject/blob/main/img/brut-agirlik.png"> | <img width=400 src="https://github.com/KasimDeliaci/ML-BTUFinalProject/blob/main/img/fatura-miktari.png"> |
| ... | ... | 

</div>

---

<div align="center">

| | |
|---|---|
| <img width=400 src="https://github.com/KasimDeliaci/ML-BTUFinalProject/blob/main/img/dist-odeme-sekli-kodu.png"> | <img width=400 src="https://github.com/KasimDeliaci/ML-BTUFinalProject/blob/main/img/dist-rejim.png"> |
| <img width=400 src="https://github.com/KasimDeliaci/ML-BTUFinalProject/blob/main/img/dist-teslim-sekli.png"> | <img width=400 src="https://github.com/KasimDeliaci/ML-BTUFinalProject/blob/main/img/dist-teslim-yeri.png"> |
| <img width=400 src="https://github.com/KasimDeliaci/ML-BTUFinalProject/blob/main/img/dist-vergi-kodu.png"> | <img width=400 src="https://github.com/KasimDeliaci/ML-BTUFinalProject/blob/main/img/toplam-sigorta_dovizi.png"> |
| ... | ... | 

</div>



## 5. Veri Ön İşleme

Bu bölümde, veri seti üzerinde yapılan ön işleme adımları detaylı bir şekilde açıklanmaktadır.

### 5.1 Eksik Verilerin İşlenmesi

Veri setindeki eksik veriler, aşağıdaki adımlar izlenerek ele alınmıştır:

1. **Eksik Veri Oranı Yüksek Sütunların Silinmesi:**

   - `delete_columns_with_high_missing_ratio()` fonksiyonu kullanılarak eksik veri oranı %40'ın üzerinde olan sütunlar veri setinden silinmiştir. Bu fonksiyon, her sütundaki eksik veri oranını hesaplar ve belirlenen eşik değerini aşan sütunları siler. Bunun sonucunda 11 sütun veri setinden silinmiştir.

2. **Kategorik Değişkenlerdeki Eksik Verilerin Mod ile Doldurulması:**

   - `fillna()` fonksiyonu kullanılarak `Navlun_miktarinin_dovizi`, `Toplam_sigorta_dovizi` ve `Sigorta_miktarinin_dovizi` sütunlarındaki eksik veriler, o sütunun en sık görülen değeri (mod) ile doldurulmuştur. Bu sütundaki eksik veri oranı az olduğu için ve kategorik değişken oldukları için modu ile doldurulması uygun görülmüştür.

3. **Gereksiz Sütunların Silinmesi:**

   - `drop()` fonksiyonu kullanılarak `gonderici_id` ve `file_name` sütunları veri setinden silinmiştir. Bu sütunlar, XML dosyalarını ayrıştırırken kullanılan ve modelleme için gerekli olmayan sütunlardır.

**Eksik Veri Analizi Sonuçları:**

-   Başlangıçta veri setinde 67 özellik bulunmaktadır.
-   Eksik veriler sonucunda 11 özellik silinmiş, 3 özellikteki eksik veriler doldurulmuş ve 2 özellik katkısız olduğu için silinmiştir.
-   Sonuç olarak, veri setinde 54 özellik kalmıştır ve hiçbirinde eksik veri bulunmamaktadır.

**Sıfır Değer Analizi ve Sütun Silinmesi:**
-   Bu noktada veri setindeki bazı sütunların tamamının sıfırlardan oluştuğunu fark ettiğim için onları ayrıca incelemek istedim.

-   `check_zero_ratio()` fonksiyonu ile veri setindeki sıfır değerlerinin oranı analiz edilmiştir. Bu fonksiyon, her sütundaki sıfır değerlerinin oranını hesaplar.
-   Sıfır oranı %63'ün üzerinde olan 5 özellik (`Ihracat_fatura_tutari`, `Esya_bedeli`, `Toplam_esya_bedeli`, `Toplam_navlun`, `Toplam_sigorta_bedeli`) `drop()` fonksiyonu ile veri setinden silinmiştir.
-   Kalan oranları işbilgine dayanarak ve araştırmalarım sonucu anlamlı olabileceğini fark ettiğim için bıraktım, mesela bazı ürünlerde devlet teşviki gibi nedenlerden dolayı bazı kalemlerin giderleri sıfırlanabiliyor.
-   Anlamsız ve yüksek sayıda sıfır içeren sütunlar da silindikten sonra veri setinde 33 sütun kalmıştır.

### 5.2 Aykırı Değerlerin Tespiti ve Düzeltilmesi

Aykırı değerler, veri setinde diğer gözlemlerden önemli ölçüde farklı olan değerlerdir ve analiz sonuçlarını olumsuz etkileyebilirler. Bu projede, aykırı değerleri tespit etmek ve düzeltmek için `handle_outliers()` fonksiyonu kullanılmıştır.

**`handle_outliers()` Fonksiyonu:**

Bu fonksiyon, verilen bir DataFrame'deki sayısal sütunlarda aykırı değerleri tespit eder ve sınırlandırma (capping) yöntemi ile işler. 

**Fonksiyonun Yapısı:**

1. **IQR (Interquartile Range) Hesaplama:**  İlk olarak, her bir sayısal sütun için IQR değeri hesaplanır. IQR, verinin %75'lik ve %25'lik çeyrekleri arasındaki farktır ve verinin dağılımı hakkında bilgi verir.

2. **Alt ve Üst Eşik Değerlerin Belirlenmesi:** IQR değerine dayanarak, aykırı değerleri belirlemek için alt ve üst eşik değerleri hesaplanır. Genellikle, alt eşik değeri Q1 - 1.5 \* IQR, üst eşik değeri ise Q3 + 1.5 \* IQR olarak belirlenir.

3. **Sınırlandırma (Capping):**  Alt eşik değerinin altında kalan değerler, alt eşik değeri ile; üst eşik değerinin üstünde kalan değerler ise üst eşik değeri ile sınırlandırılır. Bu sayede, aykırı değerlerin etkisi azaltılırken, veri setindeki bilgi kaybı da minimize edilir.

**Sınırlandırma Tercih Edilmesinin Nedeni:**

Bu projede, veri setinin boyutu nispeten küçük olduğundan, aykırı değerleri silmek yerine sınırlandırma yöntemi tercih edilmiştir. Sınırlandırma, aykırı değerlerin etkisini azaltırken veri setindeki bilgi kaybını önlemeye yardımcı olur.

**Sonuç:**

`handle_outliers()` fonksiyonu sayesinde, aykırı değerlerin olumsuz etkileri azaltılarak daha güvenilir ve doğru modeller elde edilmesi hedeflenmiştir.

### 5.3 Kategorik Değişkenlerin Kodlanması

Makine öğrenmesi algoritmalarının çoğu sayısal verilerle çalışır. Bu nedenle, kategorik değişkenleri modellemede kullanabilmek için sayısal değerlere dönüştürmek gerekir. Bu projede, kategorik değişkenleri kodlamak için **hedef değişkene göre ortalama kodlama (target encoding)** yöntemi kullanılmıştır. Bu yöntem, `target_encode()` fonksiyonu ile uygulanmıştır.

**`target_encode()` Fonksiyonu:**

Bu fonksiyon, kategorik değişkenleri hedef değişkenin ortalama değerlerine göre dönüştürür. 

**Fonksiyonun Yapısı:**

1. Her bir kategorik değişken için, hedef değişkenin her bir kategorideki ortalama değeri hesaplanır.
2. Kategorik değişkenin her bir değeri, karşılık gelen ortalama değer ile değiştirilir.

**Target Encoding Tercih Edilmesinin Nedenleri:**

- **Yüksek Kardinalite:** Veri setindeki bazı kategorik değişkenlerin kardinalitesi (benzersiz değer sayısı) yüksektir. One-Hot Encoding gibi yöntemler, yüksek kardinaliteli değişkenlerde çok sayıda yeni sütun oluşturarak modelin karmaşıklığını ve boyutunu artırabilir. Target encoding, bu sorunu önleyerek değişkenleri tek bir sütunla temsil eder. One-hot encoding denediğimde 70 yeni sütun ekleniyordu, bu karmaşıklığı azaltmak için target encoding kullandım.
- **Bilgi Kaybını Önleme:** Target encoding, kategorik değişkenler ile hedef değişken arasındaki ilişkiyi koruyarak bilgi kaybını önler.
- **Model Performansını Artırma:** Target encoding, modelin kategorik değişkenlerdeki bilgileri daha etkili bir şekilde kullanmasını sağlayarak tahmin performansını artırabilir.

**Sonuç:**

`target_encode()` fonksiyonu ile kategorik değişkenler sayısal değerlere dönüştürülerek, makine öğrenmesi modellerinde kullanılabilir hale getirilmiştir. Bu yöntem, yüksek kardinaliteli değişkenlerde bilgi kaybını önleyerek ve model performansını artırarak daha doğru ve etkili modeller elde edilmesine yardımcı olur.

### 5.4 Değişken Mühendisliği

Mevcut değişkenlerden yeni değişkenler türetilerek modelin performansını artırmak ve veriye yeni bilgiler eklemek hedeflenir. Bu projede aşağıdaki yeni değişkenler oluşturulmuştur:

- **`Toplam_Maliyet`**: `Yurt_ici_harcama` ve `Yurt_disi_harcama` değişkenlerinin toplamı alınarak oluşturulmuştur. Bu değişken, bir ihracat işleminin toplam maliyetini temsil eder.

- **`Paket_Basina_Maliyet`**:  `Toplam_Maliyet` değişkeninin `Kap_adedi` değişkenine bölünmesiyle elde edilmiştir. Bu değişken, her bir kap için ortalama maliyeti gösterir.

- **`Paket_Basina_Agirlik`**: `Net_agirlik` değişkeninin `Kap_adedi` değişkenine bölünmesiyle elde edilmiştir. Bu değişken, her bir kaptaki ortalama ağırlığı temsil eder.

- **`Agirlik_Yogunlugu`**:  `Brut_agirlik` değişkeninin `Kap_adedi` değişkenine bölünmesiyle elde edilmiştir. Bu değişken, her bir kap için brüt ağırlığın yoğunluğunu gösterir.

- **`Nakliye_Orani`**: `Toplam_navlun` değişkeninin `Toplam_Maliyet` değişkenine bölünmesiyle elde edilmiştir. Bu değişken, toplam maliyet içindeki nakliye maliyetlerinin oranını temsil eder.

- **`Sigorta_Orani`**: `Toplam_sigorta_bedeli` değişkeninin `Toplam_Maliyet` değişkenine bölünmesiyle elde edilmiştir. Bu değişken, toplam maliyet içindeki sigorta maliyetlerinin oranını temsil eder.

Bu yeni değişkenler, modelin ihracat maliyetlerini daha iyi anlamasına ve daha doğru tahminler yapmasına yardımcı olabilir.

### 5.5 Korelasyon Analizi

Değişkenler arasındaki korelasyon, bir değişkendeki değişimin diğer değişkendeki değişimi nasıl etkilediğini gösterir. Yüksek korelasyona sahip değişkenler, modelde benzer bilgileri taşıdıkları için gereksiz yere karmaşıklığa neden olabilirler. Bu nedenle, bu projede yüksek korelasyona sahip özellikleri belirlemek ve veri setinden çıkarmak için korelasyon analizi yapılmıştır. Bu analiz, `correlation_heatmap()` ve `drop_correlated_features()` fonksiyonları kullanılarak gerçekleştirilmiştir.

**`correlation_heatmap()` Fonksiyonu:**

Bu fonksiyon, verilen DataFrame'deki değişkenler arasındaki korelasyonu hesaplar ve bir ısı haritası (heatmap) olarak görselleştirir. Isı haritası, değişkenler arasındaki korelasyonu renklerle göstererek, yüksek pozitif korelasyonu kırmızı, yüksek negatif korelasyonu mavi ve düşük korelasyonu ise beyaz tonlarıyla ifade eder.

**`drop_correlated_features()` Fonksiyonu:**

Bu fonksiyon, verilen DataFrame'deki yüksek korelasyona sahip özellikleri belirleyerek siler. Fonksiyon, hem hedef değişkenle düşük korelasyona sahip özellikleri hem de birbirleriyle yüksek korelasyona sahip özellikleri siler. Bu işlem, modelin gereksiz bilgilerden arındırılarak daha sade ve etkili hale getirilmesini sağlar.

**Fonksiyonun Yapısı:**

1. **Hedef Değişkenle Korelasyon:** İlk olarak, her bir özellik ile hedef değişken arasındaki korelasyon hesaplanır. En düşük korelasyona sahip 3 değişken silinir.

2. **Özellikler Arası Korelasyon:** Daha sonra, özellikler arasındaki korelasyon matrisi hesaplanır. Korelasyonun mutlak değeri belirli bir eşik değerin  0.90) üzerinde olan özellik çiftlerinden biri silinir. Bu işlem, yüksek korelasyona sahip ve benzer bilgileri taşıyan özelliklerin modelden çıkarılmasını sağlar.

**Sonuç:**

Korelasyon analizi ve özellik silme işlemleri sonucunda, modelde gereksiz yere karmaşıklığa neden olan değişkenler veri setinden çıkarılmıştır. Bu sayede, modelin daha sade, yorumlanabilir ve etkili olması hedeflenmiştir.

**Görselleştirme:**

Aşağıda, `correlation_heatmap()` fonksiyonu ile oluşturulan ısı haritası görülmektedir.

| ![Korelasyon Haritası](https://github.com/KasimDeliaci/ML-BTUFinalProject/blob/main/img/corrheatmap.png) |
|:-------------------------------------------------------------------------------------------:|

### 5.6 Özellik Ölçeklendirme

Özellik ölçeklendirme, farklı ölçeklere sahip sayısal özelliklerin aynı ölçeğe getirilmesi işlemidir. Bu, birçok makine öğrenmesi algoritmasının performansını artırmak için önemli bir adımdır. Çünkü farklı ölçeklerdeki özellikler, modelin bazı özelliklere diğerlerinden daha fazla ağırlık vermesine neden olabilir. Bu projede, özellik ölçeklendirme için `StandardScaler()` kullanılmıştır.

**`StandardScaler()`**

Bu fonksiyon, her bir özelliğin ortalamasını 0, standart sapmasını ise 1 olacak şekilde standartlaştırır. Bu işlem, özelliklerin aynı ölçeğe getirilmesini ve modelin tüm özelliklere eşit ağırlık vermesini sağlar.

**Neden Standartlaştırma?**

-   **Gradyan inişli algoritmalar:** Standartlaştırma, gradyan inişli algoritmaların daha hızlı ve daha kararlı bir şekilde yakınsamasına yardımcı olabilir.
-   **Uzaklık bazlı algoritmalar:** KNN ve K-Means gibi uzaklık bazlı algoritmalar, farklı ölçeklerdeki özelliklerden olumsuz etkilenebilir. Standartlaştırma, bu algoritmaların performansını artırabilir. Özellikle KNN algoritması, değişkenlerin ölçeklerine duyarlıdır ve standartlaştırma bu algoritmanın daha doğru sonuçlar üretmesine yardımcı olur.
-   **Düzenlileştirme (Regularization):** Lasso ve Ridge gibi düzenlileştirme yöntemleri, standartlaştırılmış verilerle daha iyi çalışır.

**Sonuç**

`StandardScaler()` ile özelliklerin standartlaştırılması, modelin performansını artırmak, eğitim süresini kısaltmak ve daha doğru sonuçlar elde etmek için önemli bir adımdır. Bu işlem, özellikle uzaklık bazlı algoritmaların kullanıldığı durumlarda modelin daha doğru ve güvenilir sonuçlar üretmesini sağlar.

## 6. Modeller

Bu bölümde, projede kullanılan modeller detaylı bir şekilde açıklanmaktadır.

### 6.1. Denetimli (Supervised) Öğrenme (Regresyon)

İhracat maliyetini tahmin etmek için aşağıdaki regresyon modelleri kullanılmıştır:

#### 6.1.1 Lineer Regresyon

`train_evaluate_linearRegression()` fonksiyonu ile Lineer Regresyon modeli eğitilmiş ve değerlendirilmiştir. Bu fonksiyon, verilen eğitim ve test verilerini kullanarak bir Lineer Regresyon modeli oluşturur ve modelin performansını çeşitli metriklerle değerlendirir.

**Fonksiyonun Yapısı:**

1. **Model Oluşturma:** `LinearRegression()` sınıfından bir model nesnesi oluşturulur.
2. **Model Eğitimi:** `fit()` metodu kullanılarak model, eğitim verileri ile eğitilir.
3. **Tahmin Yapma:** `predict()` metodu kullanılarak test verileri üzerinde tahminler yapılır.
4. **Performans Değerlendirmesi:** `mean_squared_error()`, `r2_score()` gibi metrikler kullanılarak modelin performansı değerlendirilir.


**Sonuçlar:**

Lineer Regresyon modelinin performans metrikleri aşağıdaki gibidir:

-   MSE: 0.0000
-   RMSE: 0.0000
-   R^2: 1.0000

**Overfitting:**

Elde edilen sonuçlar, modelin aşırı uyum (overfitting) yaptığını göstermektedir. MSE ve RMSE değerlerinin 0, R^2 değerinin ise 1 olması, modelin eğitim verilerini mükemmel bir şekilde öğrendiğini, ancak test verileri üzerinde genelleme yapamadığını gösterir. Bu durum, modelin eğitim verilerindeki gürültüyü ve rastgele dalgalanmaları da öğrenmesi nedeniyle gerçek dünya verilerine uygulanamaz hale gelmesine neden olur.

**Overfitting Nedenleri:**

-   **Veri Setinin Küçük Olması:**  Küçük veri setleri, modelin genelleme yeteneğini azaltarak overfitting'e yol açabilir.
-   **Çok Sayıda Değişken:**  Çok sayıda değişken, modelin karmaşıklığını artırarak overfitting riskini yükseltir.
-   **Modelin Karmaşıklığı:** Lineer Regresyon gibi basit bir model bile, veri setine göre çok karmaşık olabilir ve overfitting yapabilir.

**Lasso Regresyonu:**

Overfitting problemini çözmek için, bir sonraki adımda Lasso Regresyonu kullanılacaktır. Lasso Regresyonu, L1 düzenlileştirme (regularization) kullanarak modelin karmaşıklığını azaltır ve overfitting'i önlemeye yardımcı olur. L1 düzenlileştirme, modeldeki katsayıların mutlak değerlerinin toplamını cezalandırarak bazı katsayıları sıfıra eşitler ve böylece modelde özellik seçimi yapar. Bu sayede, model daha sade ve genelleme yeteneği daha yüksek hale gelir.

### 6.1.2 Lasso Regresyonu

`train_evaluate_LassoRegression()` fonksiyonu kullanılarak Lasso regresyonu modeli eğitilmiş ve değerlendirilmiştir. Bu fonksiyon, verilen eğitim ve test verilerini kullanarak bir Lasso Regresyon modeli oluşturur ve modelin performansını çeşitli metriklerle değerlendirir. Lasso Regresyonu, Lineer Regresyon'a L1 düzenlileştirme (regularization) ekleyerek modelin karmaşıklığını azaltır ve aşırı uyumu (overfitting) önlemeye yardımcı olur.

**Fonksiyonun Yapısı:**

1.  **Model Oluşturma:** `Lasso()` sınıfından bir model nesnesi oluşturulur. `alpha` parametresi, düzenlileştirmenin gücünü kontrol eder.
2.  **Model Eğitimi:** `fit()` metodu kullanılarak model, eğitim verileri ile eğitilir.
3.  **Tahmin Yapma:** `predict()` metodu kullanılarak test verileri üzerinde tahminler yapılır.
4.  **Performans Değerlendirmesi:** `mean_squared_error()`, `r2_score()` gibi metrikler kullanılarak modelin performansı değerlendirilir.
5.  **Baseline Model:** Modelin performansını karşılaştırmak için basit bir baseline model oluşturulur. Bu model, tüm tahminleri eğitim verilerinin ortalaması olarak yapar.

**Sonuçlar:**

Lasso Regresyon modelinin performans metrikleri aşağıdaki gibidir:

-   MSE: 4571250.1678
-   RMSE: 2138.0482
-   R^2: 0.9907

Baseline Model Performansı:

-   MSE: 504459010.6809
-   RMSE: 22460.1650
-   R^2: -0.0294

**Değerlendirme:**

Lasso Regresyon modeli, baseline modelden önemli ölçüde daha iyi performans göstermiştir. MSE ve RMSE değerleri baseline modele göre çok daha düşük, R^2 değeri ise çok daha yüksektir. Bu, Lasso Regresyon modelinin ihracat maliyetlerini tahmin etmede daha başarılı olduğunu göstermektedir.

**Overfitting:**

Lineer Regresyon modeline kıyasla overfitting problemi aşılmış olsa da R^2 değeri hala oldukça yüksek olduğu için modelin genelleştirme performansı çok iyi değildir. KNN kullanarak daha dengeli bir model elde etmeye çalışalım.


### 6.1.3 KNN Regresyonu

`train_evaluate_KNNRegressor()` fonksiyonu kullanılarak KNN (K-En Yakın Komşu) regresyonu modeli eğitilmiş ve değerlendirilmiştir. Bu fonksiyon, verilen eğitim ve test verilerini kullanarak bir KNN Regresyon modeli oluşturur ve modelin performansını çeşitli metriklerle değerlendirir. KNN Regresyonu, bir veri noktasının değerini, en yakın komşularının değerlerinin ortalaması olarak tahmin eder.

**Fonksiyonun Yapısı:**

1.  **Model Oluşturma:** `KNeighborsRegressor()` sınıfından bir model nesnesi oluşturulur. `n_neighbors` parametresi, komşu sayısını belirler.
2.  **Model Eğitimi:** `fit()` metodu kullanılarak model, eğitim verileri ile eğitilir. 
3.  **Tahmin Yapma:** `predict()` metodu kullanılarak test verileri üzerinde tahminler yapılır. Her bir test verisi için, en yakın `n_neighbors` komşusu bulunur ve bu komşuların hedef değişken değerlerinin ortalaması alınarak tahmin yapılır.
4.  **Performans Değerlendirmesi:** `mean_squared_error()`, `r2_score()` gibi metrikler kullanılarak modelin performansı değerlendirilir.
5.  **Baseline Model:** Modelin performansını karşılaştırmak için basit bir baseline model oluşturulur. Bu model, tüm tahminleri eğitim verilerinin ortalaması olarak yapar.

**Sonuçlar:**

KNN Regresyon modelinin performans metrikleri aşağıdaki gibidir:

-   MSE: 38312002.9416
-   RMSE: 6189.6690
-   R^2: 0.9218

Baseline Model Performansı:

-   MSE: 504459010.6809
-   RMSE: 22460.1650
-   R^2: -0.0294

**Değerlendirme:**

KNN Regresyon modeli, baseline modelden önemli ölçüde daha iyi performans göstermiştir. MSE ve RMSE değerleri baseline modele göre çok daha düşük, R^2 değeri ise çok daha yüksektir. Bu, KNN Regresyon modelinin ihracat maliyetlerini tahmin etmede daha başarılı olduğunu göstermektedir. Ayrıca Lineer ve Lasso'da yaşanılan overfitting probleminin önüne geçilerek daha genelleştirilebilir bir model elde edilmiştir. Ancak farklı algoritmalar ile R^2 skoru arttırılabilir, bunun için RandomForest uygun bir tercih olacaktır.

**Modelin Optimizasyonu:**

KNN Regresyon modelinin performansını artırmak için, `n_neighbors` parametresi gibi hiperparametreler optimize edilebilir. Ayrıca, farklı uzaklık metrikleri denenebilir.

**Sonuç:**

KNN Regresyon modeli, ihracat maliyetlerini tahmin etmede baseline modelden çok daha başarılıdır ve verileri iyi bir şekilde açıklayabilir. Bu model, işletmelere ihracat maliyetlerini tahmin etme ve planlama konusunda yardımcı olabilir.

### 6.1.4 Random Forest Regresyonu

`train_evaluate_RandomForest_with_CV()` fonksiyonu kullanılarak Random Forest regresyonu modeli eğitilmiş ve değerlendirilmiştir. Bu fonksiyon, verilen eğitim ve test verilerini kullanarak bir Random Forest Regresyon modeli oluşturur, çapraz doğrulama ile optimize eder ve modelin performansını çeşitli metriklerle değerlendirir. Random Forest, birden çok karar ağacını bir araya getirerek daha güçlü ve genelleme yeteneği yüksek bir model oluşturan bir topluluk öğrenme (ensemble learning) yöntemidir.

**Fonksiyonun Yapısı:**

1. **Model Oluşturma:** `RandomForestRegressor()` sınıfından bir model nesnesi oluşturulur. `n_estimators`, `max_depth` gibi parametreler, modelin karmaşıklığını kontrol eder.
2. **Çapraz Doğrulama:** `GridSearchCV()` fonksiyonu kullanılarak çapraz doğrulama yapılır ve modelin hiperparametreleri optimize edilir. Bu işlemde, veri seti 5 parçaya bölünür ve model her seferinde farklı bir parça üzerinde test edilerek en iyi hiperparametreler bulunur.
3. **Model Eğitimi:** `fit()` metodu kullanılarak model, eğitim verileri ile eğitilir.
4. **Tahmin Yapma:** `predict()` metodu kullanılarak test verileri üzerinde tahminler yapılır.
5. **Performans Değerlendirmesi:** `mean_squared_error()`, `r2_score()` gibi metrikler kullanılarak modelin performansı değerlendirilir.
6. **Baseline Model:** Modelin performansını karşılaştırmak için basit bir baseline model oluşturulur. Bu model, tüm tahminleri eğitim verilerinin ortalaması olarak yapar.

**Sonuçlar:**

Random Forest modelinin 5-kat çapraz doğrulama ile elde edilen ortalama performans metrikleri aşağıdaki gibidir:

- Ortalama MSE: 4314539.5705
- Ortalama RMSE: 2069.0952
- Ortalama R²: 0.9887

Baseline Model Performansı:

- MSE: 394124347.6153
- RMSE: 19852.5653
- R²: 0.0000

**Değerlendirme:**

Random Forest modeli, baseline modelden önemli ölçüde daha iyi performans göstermiştir. MSE ve RMSE değerleri baseline modele göre çok daha düşük, R^2 değeri ise çok daha yüksektir. Bu, Random Forest modelinin ihracat maliyetlerini tahmin etmede daha başarılı olduğunu göstermektedir. KNN'e göre hem overfit problemş aşılmış hem de R^2 değeri yükseltilmiştir.

**Çapraz Doğrulamanın Faydaları:**

Çapraz doğrulama, modelin hiperparametrelerini optimize etmek ve modelin genelleme yeteneğini artırmak için kullanılır. Bu sayede, modelin test verileri üzerinde de iyi bir performans göstermesi sağlanır.

**Sonuç:**

Random Forest modeli, ihracat maliyetlerini tahmin etmede baseline modelden çok daha başarılıdır ve verileri iyi bir şekilde açıklayabilir. Bu model, işletmelere ihracat maliyetlerini tahmin etme ve planlama konusunda yardımcı olabilir.

### 6.1.5 XGBoost Regresyonu

`train_evaluate_XGBoost()` fonksiyonu kullanılarak XGBoost regresyonu modeli eğitilmiş ve değerlendirilmiştir. Bu fonksiyon, verilen eğitim ve test verilerini kullanarak bir XGBoost Regresyon modeli oluşturur ve modelin performansını çeşitli metriklerle değerlendirir. XGBoost, yüksek performans ve hız sağlayan bir gradient boosting algoritmasıdır.

**Fonksiyonun Yapısı:**

1.  **Model Oluşturma:** `XGBRegressor()` sınıfından bir model nesnesi oluşturulur. `n_estimators`, `max_depth`, `learning_rate` gibi parametreler, modelin karmaşıklığını ve öğrenme hızını kontrol eder.
2.  **Model Eğitimi:** `fit()` metodu kullanılarak model, eğitim verileri ile eğitilir.
3.  **Tahmin Yapma:** `predict()` metodu kullanılarak test verileri üzerinde tahminler yapılır.
4.  **Performans Değerlendirmesi:** `mean_squared_error()`, `r2_score()` gibi metrikler kullanılarak modelin performansı değerlendirilir.
5.  **Baseline Model:** Modelin performansını karşılaştırmak için basit bir baseline model oluşturulur. Bu model, tüm tahminleri eğitim verilerinin ortalaması olarak yapar.

**Sonuçlar:**

XGBoost Regresyon modelinin performans metrikleri aşağıdaki gibidir:

-   MSE: 3658424.5340
-   RMSE: 1912.7008
-   R^2: 0.9925

Baseline Model Performansı:

-   MSE: 504459010.6809
-   RMSE: 22460.1650
-   R^2: -0.0294

**Değerlendirme:**

XGBoost Regresyon modeli, baseline modelden ve diğer modellerden (Lineer Regresyon, Lasso Regresyonu, KNN) önemli ölçüde daha iyi performans göstermiştir. MSE ve RMSE değerleri diğer modellere göre daha düşük, R^2 değeri ise daha yüksektir. Bu, XGBoost modelinin ihracat maliyetlerini tahmin etmede daha başarılı olduğunu göstermektedir.

**XGBoost'un Avantajları:**

-   **Yüksek Performans:** XGBoost, genellikle diğer makine öğrenmesi algoritmalarından daha yüksek tahmin performansı sağlar.
-   **Hız:** XGBoost, paralel hesaplama ve optimizasyon teknikleri sayesinde hızlı bir şekilde eğitilebilir.
-   **Esneklik:** XGBoost, çeşitli veri tipleri ve problemler için kullanılabilir.
-   **Overfitting'e Karşı Dayanıklılık:** XGBoost, düzenlileştirme teknikleri ve çapraz doğrulama ile overfitting'i önlemeye yardımcı olur.

**Sonuç:**

XGBoost Regresyon modeli, ihracat maliyetlerini tahmin etmede diğer modellere göre daha başarılıdır ve verileri iyi bir şekilde açıklayabilir. Bu model, işletmelere ihracat maliyetlerini tahmin etme ve planlama konusunda yardımcı olabilir.

**FİNAL: Genel Yorum:**

Projede kullanılan tüm modeller (Lineer Regresyon, Lasso Regresyonu, KNN, Random Forest ve XGBoost), baseline modelden daha iyi performans göstermiştir. Bu, makine öğrenmesi modellerinin ihracat maliyetlerini tahmin etmede etkili olabileceğini göstermektedir. Modeller arasında karşılaştırma yapıldığında, XGBoost modelinin en iyi performansı gösterdiği görülmüştür. Bu sonuç, XGBoost algoritmasının karmaşık problemlerde yüksek tahmin doğruluğu sağlama yeteneğini göstermektedir.

---

### 6.2 Denetimsiz (Unsupervised) Öğrenme (Anomali Tespiti)

Anormal veri noktalarını tespit etmek, veri setindeki beklenmedik veya olağandışı desenleri belirlemeyi amaçlar. Bu anormallikler, hatalı veri girişleri, dolandırıcılık girişimleri veya süreçteki verimsizlikler gibi çeşitli sorunları işaret edebilir. Bu projede, anormal veri noktalarını tespit etmek için K-Means Kümeleme, Hiyerarşik Kümeleme ve Isolation Forest gibi denetimsiz öğrenme modelleri kullanılmıştır.

**Problem:**

İhracat verilerindeki anormallikleri tespit ederek, potansiyel sorunları veya iyileştirme fırsatlarını belirlemek.

**Yeni Ölçeklendirme:**

Denetimsiz öğrenme modellerinin çoğu, verilerin ölçeklendirilmesinden etkilenir. Bu nedenle, `scale_features_unsupervised()` fonksiyonu kullanılarak tüm özellikler `StandardScaler()` ile standartlaştırılmıştır. Bu fonksiyon, verilen bir DataFrame'deki belirtilen özellikleri standartlaştırır. Standartlaştırma, her bir özelliğin ortalamasını 0, standart sapmasını ise 1 olacak şekilde dönüştürür. Bu işlem, özelliklerin aynı ölçeğe getirilmesini ve modelin tüm özelliklere eşit ağırlık vermesini sağlar. Eski veri setimizde label olarak kullandığımız "Toplam_Maaliyet" değişkeni ayıklanarak scale edilmişti burda onu da eklemiş olduk çünkü denetimsiz öğrenmede labellanmış data yok.

**Neden Ölçeklendirme?**

-   **Uzaklık Bazlı Algoritmalar:** K-Means ve Hiyerarşik Kümeleme gibi uzaklık bazlı algoritmalar, farklı ölçeklerdeki özelliklerden olumsuz etkilenebilir. Standartlaştırma, bu algoritmaların performansını artırabilir.
-   **Veri Görselleştirme:**  PCA (Temel Bileşen Analizi) gibi boyut indirgeme teknikleri, ölçeklendirilmiş verilerle daha iyi çalışır.

**Sonuç:**

Özelliklerin standartlaştırılması, denetimsiz öğrenme modellerinin daha doğru ve güvenilir sonuçlar üretmesine yardımcı olur. Bu sayede, ihracat verilerindeki anormallikler daha etkili bir şekilde tespit edilebilir.

### 6.2.1 K-Means Kümeleme

`train_evaluate_KMeans()` fonksiyonu kullanılarak K-Means kümeleme modeli eğitilmiş ve değerlendirilmiştir. Bu fonksiyon, verilen veri setini kullanarak bir K-Means kümeleme modeli oluşturur ve modelin performansını silhouette skoru ile değerlendirir. K-Means, veri noktalarını benzerliklerine göre belirli sayıda kümeye ayırır.

**Fonksiyonun Yapısı:**

1.  **Optimum Küme Sayısının Belirlenmesi:** Elbow yöntemi kullanılarak optimum küme sayısı belirlenir. Bu yöntemde, farklı küme sayıları için modelin within-cluster sum of squares (WCSS) değeri hesaplanır ve bu değerlerin değişimine göre optimum küme sayısı seçilir.
2.  **Model Oluşturma:** `KMeans()` sınıfından bir model nesnesi oluşturulur. `n_clusters` parametresi, küme sayısını belirler.
3.  **Model Eğitimi:** `fit()` metodu kullanılarak model, eğitim verileri ile eğitilir.
4.  **Küme Etiketlerinin Tahmin Edilmesi:** `predict()` metodu kullanılarak veri noktalarının hangi kümeye ait olduğu tahmin edilir.
5.  **Performans Değerlendirmesi:** `silhouette_score()` fonksiyonu kullanılarak modelin performansı değerlendirilir. Silhouette skoru, kümelerin ne kadar iyi ayrıldığını ve veri noktalarının kendi kümelerine ne kadar iyi ait olduğunu ölçer.

**Sonuçlar:**

K-Means kümeleme modelinin performans metriği ve optimum küme sayısı aşağıdaki gibidir:

-   Optimal Number of Clusters (k): 7
-   Silhouette Score: 0.20683424692694305

**Elbow ve PCA Grafiği:**

| ![Elbow Grafiği](https://github.com/KasimDeliaci/ML-BTUFinalProject/blob/main/img/kmeans-elbow.png) |
|:-------------------------------------------------------------------------------------------:|

| ![Image 1](https://github.com/KasimDeliaci/ML-BTUFinalProject/blob/main/img/kmeans-pca1.png) | 
|:-------------------------------------------------------------------------------------------:|

**Değerlendirme:**

Elbow grafiği incelenerek optimum küme sayısının 7 olduğu belirlenmiş ve K-Means modeli bu küme sayısı ile eğitilir.  0.2068'lik Silhouette skoru, kümelerin orta düzeyde ayrıldığını ve veri noktalarının kendi kümelerine orta düzeyde ait olduğunu gösterir.  Bu skor, kümelemenin mükemmel olmadığını, ancak veri setinde bazı anlamlı grupların olduğunu gösterebilir. 

**Sonuç:**

K-Means kümeleme modeli, ihracat verilerindeki anormallikleri tespit etmek için kullanılabilir. Modelin performansı, silhouette skoru ve elbow grafiği incelenerek değerlendirilebilir.  Ancak, daha yüksek bir silhouette skoru elde etmek için farklı kümeleme algoritmaları veya parametreleri denenebilir.

### 6.2.2 Hiyerarşik Kümeleme

`train_evaluate_HierarchicalKMeans()` fonksiyonu kullanılarak hiyerarşik kümeleme modeli eğitilmiş ve değerlendirilmiştir. Bu fonksiyon, verilen veri setini kullanarak bir hiyerarşik kümeleme modeli oluşturur ve modelin performansını silhouette skoru ile değerlendirir. Hiyerarşik kümeleme, veri noktalarını benzerliklerine göre hiyerarşik bir yapıda kümelere ayırır.

**Fonksiyonun Yapısı:**

1.  **Optimum Küme Sayısının Belirlenmesi:** K-Means algoritması ve Elbow yöntemi kullanılarak optimum küme sayısı belirlenir. Bu yöntemde, farklı küme sayıları için K-Means modelinin within-cluster sum of squares (WCSS) değeri hesaplanır ve bu değerlerin değişimine göre optimum küme sayısı seçilir.
2.  **Model Oluşturma:** `AgglomerativeClustering()` sınıfından bir model nesnesi oluşturulur. `n_clusters` parametresi, küme sayısını ve `linkage` parametresi kümelerin nasıl birleştirileceğini belirler.
3.  **Model Eğitimi:** `fit_predict()` metodu kullanılarak model eğitilir ve veri noktaları kümelere atanır.
4.  **Performans Değerlendirmesi:** `silhouette_score()` fonksiyonu kullanılarak modelin performansı değerlendirilir. Silhouette skoru, kümelerin ne kadar iyi ayrıldığını ve veri noktalarının kendi kümelerine ne kadar iyi ait olduğunu ölçer.

**Sonuçlar:**

Hiyerarşik kümeleme modelinin performans metriği ve optimum küme sayısı aşağıdaki gibidir:

-   Optimal Number of Clusters (k): 4
-   Silhouette Score: 0.25116389416176405

**Değerlendirme:**

Elbow yöntemi ile optimum küme sayısının 4 olduğu belirlenmiş ve hiyerarşik kümeleme modeli bu küme sayısı ile eğitilmiştir. 0.2512'lik Silhouette skoru, kümelerin orta düzeyde ayrıldığını ve veri noktalarının kendi kümelerine orta düzeyde ait olduğunu gösterir. Bu skor, K-Means kümeleme modelinden elde edilen silhouette skorundan biraz daha yüksektir, bu da hiyerarşik kümelemenin bu veri seti için biraz daha iyi performans gösterebileceğini düşündürmektedir.

**Elbow ve PCA Grafiği:**

| ![Elbow Grafiği](https://github.com/KasimDeliaci/ML-BTUFinalProject/blob/main/img/hier-elbow.png) |
|:-------------------------------------------------------------------------------------------:|

| ![Image 1](https://github.com/KasimDeliaci/ML-BTUFinalProject/blob/main/img/hier-pca1.png) | 
|:-------------------------------------------------------------------------------------------:|

**Sonuç:**

Hiyerarşik kümeleme modeli, ihracat verilerindeki anormallikleri tespit etmek için kullanılabilir. Modelin performansı, silhouette skoru incelenerek değerlendirilebilir. Ancak, daha yüksek bir silhouette skoru elde etmek için farklı kümeleme algoritmaları veya parametreleri denenebilir.

### 6.2.3 Isolation Forest

`train_evaluate_IsolationForest()` fonksiyonu kullanılarak Isolation Forest modeli eğitilmiş ve değerlendirilmiştir. Bu fonksiyon, verilen veri setini kullanarak bir Isolation Forest modeli oluşturur ve modelin performansını silhouette skoru ile değerlendirir. Isolation Forest, anormallikleri normal veri noktalarından izole ederek tespit eden bir algoritmadır.

**Fonksiyonun Yapısı:**

1.  **Model Oluşturma:** `IsolationForest()` sınıfından bir model nesnesi oluşturulur. `contamination` parametresi, veri setindeki anormalliklerin oranını belirler.
2.  **Model Eğitimi:** `fit()` metodu kullanılarak model, eğitim verileri ile eğitilir.
3.  **Anormallik Etiketlerinin Tahmin Edilmesi:** `predict()` metodu kullanılarak veri noktalarının anormal olup olmadığı tahmin edilir.
4.  **Performans Değerlendirmesi:** `silhouette_score()` fonksiyonu kullanılarak modelin performansı değerlendirilir. Silhouette skoru, kümelerin ne kadar iyi ayrıldığını ve veri noktalarının kendi kümelerine ne kadar iyi ait olduğunu ölçer.

**Sonuçlar:**

Isolation Forest modelinin performans metriği ve tespit edilen anormallik sayısı aşağıdaki gibidir:

-   Silhouette Score: 0.31314024357577397
-   Anomalies Detected: 53

**Değerlendirme:**

0.3131'lik Silhouette skoru, Isolation Forest modelinin K-Means ve Hiyerarşik Kümeleme modellerine göre daha iyi performans gösterdiğini ve anormallikleri daha iyi tespit ettiğini göstermektedir.

**Sonuç:**

Isolation Forest modeli, ihracat verilerindeki anormallikleri tespit etmek için etkili bir yöntemdir. Modelin performansı, silhouette skoru ve tespit edilen anormallik sayısı incelenerek değerlendirilebilir.


### Genel Yorum ve Karşılaştırma

Bu projede, denetimsiz öğrenme yöntemleri kullanılarak ihracat verilerindeki anormallikler tespit edilmeye çalışılmıştır. K-Means, Hiyerarşik Kümeleme ve Isolation Forest olmak üzere üç farklı kümeleme algoritması kullanılmıştır. Her algoritmanın kendine özgü avantajları ve dezavantajları vardır.

-   K-Means, basit ve hızlı bir algoritmadır, ancak küme sayısının önceden belirlenmesi gerekir ve kümelerin şekli ve boyutuna duyarlıdır.
-   Hiyerarşik Kümeleme, küme sayısının önceden belirlenmesini gerektirmez ve farklı şekil ve boyutlardaki kümeleri tespit edebilir, ancak K-Means'e göre daha yavaştır.
-   Isolation Forest, anormallikleri normal veri noktalarından izole ederek tespit eder ve özellikle yüksek boyutlu verilerde etkilidir.

Bu projede elde edilen sonuçlara göre, Isolation Forest modeli en yüksek silhouette skoruna sahip olup, anormallikleri tespit etmede diğer iki modelden daha başarılı olmuştur. Ancak, her veri seti için en iyi performansı gösteren algoritma farklılık gösterebilir. Bu nedenle, farklı algoritmaları denemek ve performanslarını karşılaştırmak önemlidir.

Sonuç olarak, denetimsiz öğrenme yöntemleri, ihracat verilerindeki anormallikleri tespit etmek ve potansiyel sorunları veya iyileştirme fırsatlarını belirlemek için etkili bir şekilde kullanılabilir.

# 7. Sonuçlar ve Öğrendiklerim 🎓

Bu projede ihracat maliyetlerini tahmin etmek ve anormallikleri bulmak için farklı makine öğrenmesi modelleri denedim. İşte bu süreçte öğrendiklerim ve elde ettiğim sonuçlar:

## Neler Öğrendim? 📚

### Veri Hazırlığının Önemi
Açıkçası projenin başında veri ön işlemenin bu kadar önemli olduğunu düşünmüyordum. Eksik verileri doldurmak, aykırı değerleri düzeltmek ve kategorik değişkenleri sayısal hale getirmek gibi işlemler, modellerimin performansını inanılmaz derecede artırdı.

### Özellik Mühendisliği Deneyimi
Yeni değişkenler oluşturmak başta korkutucu gelse de, zamanla bunun modelin başarısı için ne kadar kritik olduğunu gördüm. Özellikle ihracat verileriyle çalışırken, farklı para birimlerini ve ölçü birimlerini düzenlemenin önemini kavradım.

### Model Karşılaştırmaları
Birçok farklı model denedim:
* Lineer Regresyon (klasik ama etkili! ancak overfit oldu)
* Lasso Regresyonu 
* KNN (en yakın komşular)
* Random Forest
* XGBoost

XGBoost'un diğerlerinden daha iyi sonuç vermesi beni şaşırtmadı çünkü hem ağaç tabanlı hem de boosting kullanıyor. Başta karmaşık gelen bu model, zamanla en çok sevdiğim modellerden biri oldu. Ancak overfit problemini çözmek beni zorladı.

## Anormallik Tespiti Deneyimim 🔍

Üç farklı model kullandım:
* K-Means
* Hiyerarşik Kümeleme
* Isolation Forest

Isolation Forest'ın en iyi sonucu vermesini bekliyordum ağaç tabalı olduğu için, Silhouette skorları bunu açıkça gösterdi.

## Teknik Açıdan Kazanımlarım 💻

* Python'da kendimi çok geliştirdim
* Pandas ve NumPy kütüphanelerini artık daha iyi kullanabiliyorum
* Scikit-learn'ü keşfettim
* XGBoost gibi gelişmiş kütüphaneleri kullanmayı öğrendim

## Gelecek İçin Planlarım 🚀

* Daha fazla veri toplayıp modellerimi geliştirmek istiyorum
* Henüz denemediğim algoritmaları da deneyeceğim
* Anormallik tespiti için başka yöntemler de araştıracağım
* Modellerin sonuçlarını daha anlaşılır hale getirmeye çalışacağım

## Son Düşüncelerim 💭

Bu proje bana makine öğrenmesinin gerçek dünya problemlerinde nasıl kullanılabileceğini gösterdi. Başta zorlandığım konuları şimdi daha iyi anlıyorum. İhracat sektöründe veri biliminin önemini kavradım ve bu alanda daha fazla çalışma yapmak istiyorum.

Özellikle veri ön işleme ve model seçimi konularında çok şey öğrendim. Her ne kadar bazen zorlandığım anlar olsa da, sonuçta ortaya güzel bir iş çıkardığımı düşünüyorum. Bu deneyim, gelecekteki projelerim için bana güzel bir temel oluşturdu.

