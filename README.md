# İhracat Maliyet Analizi ve Anomali Tespiti Projesi

## İçindekiler
- [1.Proje Genel Bakış](#proje-genel-bakış)
- [2.Özellikler](#özellikler)
- [3.Gereksinimler](#gereksinimler)
- [4.Proje Yapısı](#proje-yapısı)
- [5.Veri Ön İşleme](#veri-ön-işleme)
- [6.Modeller](#modeller)
    - 6.1. Denetimli Öğrenme (Regresyon)
        - 5.1.1. Lineer Regresyon
        - 5.1.2. Lasso Regresyonu
        - 5.1.3. KNN Regresyonu
        - 5.1.4. Random Forest Regresyonu
        - 5.1.5. XGBoost Regresyonu
    - 6.2. Denetimsiz Öğrenme (Anomali Tespiti)
        - 5.2.1. K-Means Kümeleme
        - 5.2.2. Hiyerarşik Kümeleme
        - 5.2.3. Isolation Forest
- [7.Sonuçlar](#sonuçlar)
- [8.Kurulum ve Kullanım](#kurulum-ve-kullanım)
- [9.Katkıda Bulunma](#katkıda-bulunma)

## Proje Genel Bakış
Bu projede ATEZ YAZILIM TEKNOLOJİLERİ A.Ş'nin sağlamış olduğu gümrük/ihracat verileri kullanılmıştır. Desteklerinden ötürü kendilerine teşekkür ediyorum. Projede iki farklı learning algoritmasını kullanarak bolca pratik yapmak istedim bu yüzden hem <i>Supervised Learning<i/> hem de <i>Unsupervised Learning<i/> problemleri inceledim.

### 1. Maliyet Tahmini (Supervised Learning)
- **Amaç**: İhracat işlemlerinin toplam maliyetinin tahmin edilmesi
- **Kullanılan Veriler**:
  - Kap adedi ve ağırlık bilgileri
  - Nakliye ve sigorta maliyetleri
  - Yurt içi/dışı harcamalar
  - Döviz bilgileri
- **İş Değeri**:
  - İhracatçıların maliyet planlaması yapabilmesi
  - Fiyatlandırma stratejilerinin geliştirilmesi
  - Bütçe tahminlerinin iyileştirilmesi
  - Operasyonel verimliliğin artırılması

### 2. Anomali Tespiti (Unsupervised Learning)
- **Amaç**: Olağandışı ihracat işlemlerinin tespit edilmesi
- **Kullanım Alanları**:
  - Hatalı veri girişlerinin tespiti
  - Potansiyel dolandırıcılık vakalarının belirlenmesi
  - Operasyonel anormalliklerin saptanması
  - Risk yönetimi ve kalite kontrol
- **Metodoloji**:
  - Kümeleme analizi ile benzer işlemlerin gruplandırılması
  - İzolasyon ormanı ile aykırı değerlerin tespiti
  - Hiyerarşik kümeleme ile yapısal anomalilerin belirlenmesi

### Proje Çıktıları
1. **Tahmin Modeli**:
   - Yeni ihracat işlemlerinin maliyet tahminini yapabilen bir model
   - Model performans metrikleri ve karşılaştırma analizleri
   - Özellik önem dereceleri ve model yorumlanabilirliği

2. **Anomali Tespit Sistemi**:
   - Otomatik aykırı değer tespiti yapan bir sistem
   - Kümeleme bazlı anomali skorlaması
   - Görsel analiz araçları ve raporlama mekanizması

3. **Veri İşleme Pipeline'ı**:
   - Otomatikleştirilmiş veri temizleme süreçleri
   - Özellik mühendisliği adımları
   - Ölçeklendirme ve kodlama işlemleri

### Teknik Detaylar
- **Veri Boyutu**: 
  - Satır sayısı: [Veri setindeki toplam kayıt sayısı]
  - Değişken sayısı: [Toplam özellik sayısı]
  
- **Performans Metrikleri**:
  - Maliyet Tahmini: RMSE, R² ve MAE
  - Anomali Tespiti: Silhouette skoru ve küme kalite metrikleri

- **Kullanılan Teknolojiler**:
  - Python 3.x
  - Scikit-learn
  - XGBoost
  - Pandas & NumPy
  - Matplotlib & Seaborn

[Diğer bölümler aynı şekilde devam eder...]
