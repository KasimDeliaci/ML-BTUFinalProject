# İhracat Maliyet Analizi ve Anomali Tespiti Projesi

## İçindekiler

- [1. Proje Genel Bakış](#1-proje-genel-bakış)
- [2. Özellikler](#2-özellikler)
- [3. Gereksinimler](#3-gereksinimler)
- [4. Projenin Akışı](#4-proje-yapısı)
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

### 5.3. Kategorik Değişkenlerin Kodlanması

`target_encode()` fonksiyonu kullanılarak kategorik değişkenler hedef değişkene göre ortalama değerleri ile kodlanmıştır.

### 5.4. Korelasyon Analizi

`correlation_heatmap()` ve `drop_correlated_features()` fonksiyonları kullanılarak yüksek korelasyona sahip özellikler veri setinden çıkarılmıştır.

### 5.5. Özellik Ölçeklendirme

`StandardScaler()` kullanılarak özellikler standartlaştırılmıştır.

## 6. Modeller

Bu bölümde, projede kullanılan modeller detaylı bir şekilde açıklanmaktadır.

### 6.1. Denetimli (Supervised) Öğrenme (Regresyon)

İhracat maliyetini tahmin etmek için aşağıdaki regresyon modelleri kullanılmıştır:

#### 6.1.1. Lineer Regresyon

`train_evaluate_linearRegression()` fonksiyonu kullanılarak lineer regresyon modeli eğitilmiş ve değerlendirilmiştir.

#### 6.1.2. Lasso Regresyonu

`train_evaluate_LassoRegression()` fonksiyonu kullanılarak Lasso regresyonu modeli eğitilmiş ve değerlendirilmiştir.

#### 6.1.3. KNN Regresyonu

`train_evaluate_KNNRegressor()` fonksiyonu kullanılarak KNN regresyonu modeli eğitilmiş ve değerlendirilmiştir.

#### 6.1.4. Random Forest Regresyonu

`train_evaluate_RandomForest_with_CV()` fonksiyonu kullanılarak Random Forest regresyonu modeli eğitilmiş ve değerlendirilmiştir.

#### 6.1.5. XGBoost Regresyonu

`train_evaluate_XGBoost()` fonksiyonu kullanılarak XGBoost regresyonu modeli eğitilmiş ve değerlendirilmiştir.

### 6.2. Denetimsiz (Unsupervised) Öğrenme (Anomali Tespiti)

Anormal veri noktalarını tespit etmek için aşağıdaki modeller kullanılmıştır:

#### 6.2.1. K-Means Kümeleme

`train_evaluate_KMeans()` fonksiyonu kullanılarak K-Means kümeleme modeli eğitilmiş ve değerlendirilmiştir.

#### 6.2.2. Hiyerarşik Kümeleme

`train_evaluate_HierarchicalKMeans()` fonksiyonu kullanılarak hiyerarşik kümeleme modeli eğitilmiş ve değerlendirilmiştir.

#### 6.2.3. Isolation Forest

`train_evaluate_IsolationForest()` fonksiyonu kullanılarak Isolation Forest modeli eğitilmiş ve değerlendirilmiştir.

## 7. Sonuçlar

Bu bölümde, modellerin performans metrikleri ve karşılaştırılması sunulmaktadır. Ayrıca, projede elde edilen önemli bulgular ve sonuçlar özetlenmektedir.

## 8. Kurulum ve Kullanım

Bu projeyi çalıştırmak için aşağıdaki adımları izleyin:

1. Proje kodlarını klonlayın veya indirin.
2. Gerekli kütüphaneleri yükleyin.
3. Veri setlerini `data/` dizinine yerleştirin.
4. Jupyter Notebook dosyalarını çalıştırın veya Python script dosyalarını çalıştırın.

## 9. Katkıda Bulunma

Bu projeye katkıda bulunmak isteyenler, aşağıdaki adımları izleyebilirler:

1. Proje kodlarını forklayın.
2. Değişikliklerinizi yapın.
3. Pull request oluşturun.
