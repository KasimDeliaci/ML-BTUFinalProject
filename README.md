# İhracat Maliyet Analizi ve Anomali Tespiti Projesi

## İçindekiler

- [1. Proje Genel Bakış](#1-proje-genel-bakış)
- [2. Özellikler](#2-özellikler)
- [3. Gereksinimler](#3-gereksinimler)
- [4. Proje Yapısı](#4-proje-yapısı)
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
    - Satır sayısı: [Veri setindeki toplam kayıt sayısı]
    - Değişken sayısı: [Toplam özellik sayısı]

- **Performans Metrikleri**:
    - **Maliyet Tahmini**: RMSE, R² ve MAE
    - **Anomali Tespiti**: Silhouette skoru ve küme kalite metrikleri

- **Kullanılan Teknolojiler**:
    - Python 3.x
    - Scikit-learn
    - XGBoost
    - Pandas & NumPy
    - Matplotlib & Seaborn

## 2. Özellikler

Bu projede aşağıdaki özellikler bulunmaktadır:

- **Veri Temizleme:** Eksik verilerin işlenmesi, aykırı değerlerin tespiti ve düzeltilmesi.
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

## 4. Proje Yapısı

Proje aşağıdaki dosya ve dizinlerden oluşmaktadır:

- `data/`: Veri setlerinin bulunduğu dizin.
- `notebooks/`: Jupyter Notebook dosyalarının bulunduğu dizin.
- `scripts/`: Python script dosyalarının bulunduğu dizin.
- `models/`: Eğitilmiş modellerin kaydedildiği dizin.
- `results/`: Sonuçların ve görsellerin kaydedildiği dizin.
- `README.md`: Proje hakkında bilgi veren dosya.

**Not:** Projenizin gerçek dosya ve dizin yapısına göre bu bölümü düzenlemeniz gerekebilir.

## 5. Veri Ön İşleme

Bu bölümde, veri seti üzerinde yapılan ön işleme adımları detaylı bir şekilde açıklanmaktadır.

### 5.1. Eksik Verilerin İşlenmesi

Veri setindeki eksik veriler, aşağıdaki yöntemler kullanılarak ele alınmıştır:

- **Silme:** `delete_columns_with_high_missing_ratio()` fonksiyonu kullanılarak eksik veri oranı yüksek olan sütunlar veri setinden silinmiştir.
- **Doldurma:** `fillna()` fonksiyonu kullanılarak eksik veriler mod ile doldurulmuştur.

### 5.2. Aykırı Değerlerin Tespiti ve Düzeltilmesi

`handle_outliers()` fonksiyonu kullanılarak aykırı değerler tespit edilmiş ve sınırlandırma (capping) yöntemi ile düzeltilmiştir.

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
