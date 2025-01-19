# -*- coding: utf-8 -*-
"""MLFinal.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/13hkTimQ8InatAgpvbdnQuxQmypObca2r
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('kasim_df.csv')

def check_df(dataframe, head=5):
    """
    Veri setinin genel özelliklerini inceleyen fonksiyon.
    """
    print("################## Shape ##################")
    print(dataframe.shape)
    print("################## Types ##################")
    print(dataframe.dtypes)
    print("################## Head ##################")
    print(dataframe.head(head))
    print("################## Tail ##################")
    print(dataframe.tail(head))
    print("################## Na ##################")
    print("Eksik Değer Sayıları ve Yüzdeleri:")
    # Calculate missing values and ratio inside the function to ensure it's always up-to-date.
    missing_values = dataframe.isnull().sum()
    missing_ratio = (missing_values / len(dataframe)) * 100
    for col, val, ratio in zip(missing_values.index, missing_values.values, missing_ratio.values):
        print(f"{col}: {val} eksik değer ({ratio:.2f}%)")
    print("################## Quantiles ##################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(data)
#data.to_csv('giden_train_missing.csv', index=False)

def delete_columns_with_high_missing_ratio(dataframe, threshold=40):
    """
    Eksik değer oranı belirli bir eşik değerin üzerinde olan sütunları siler.

    Args:
        dataframe: İşlem yapılacak Pandas DataFrame'i.
        threshold: Sütun silinmesi için gereken eksik değer oranı eşiği (yüzde olarak).

    Returns:
        Eksik değer oranı yüksek sütunları silinmiş Pandas DataFrame'i.
    """

    missing_values = dataframe.isnull().sum()
    missing_ratio = (missing_values / len(dataframe)) * 100
    columns_to_delete = missing_ratio[missing_ratio > threshold].index
    dataframe = dataframe.drop(columns_to_delete, axis=1)
    print("Deleted Columns:")
    print(columns_to_delete)
    return dataframe

# Clean the data
data = delete_columns_with_high_missing_ratio(data)

# Check again after cleaning
check_df(data)
data.to_csv('giden_dropped.csv', index=False)

#Çıktıyı analiz edince dçok daha az oranda eksik değer içeren döviz kategorik değişkenlerini gördüm
#Kategorik oldukları için eksik değerleri mod'ları ile doldurabiliriz.

# `Navlun_miktarinin_dovizi`, `Toplam_sigorta_dovizi` ve `Sigorta_miktarinin_dovizi` sütunlarındaki eksik değerleri mod ile doldur.
for column in ['Navlun_miktarinin_dovizi', 'Toplam_sigorta_dovizi', 'Sigorta_miktarinin_dovizi']:
    mode_value = data[column].mode()[0]
    data[column] = data[column].fillna(mode_value)

# Doldurulmuş veri çerçevesini `giden_filled.csv` adıyla kaydet
data.to_csv('giden_filled.csv', index=False)
check_df(data)

# check_df çıktısı kontrol edildiğinde hiç eksik değer yok :)

#GondericiID ve XMl dosyalarını parse ederken kullandığım filename sütunlarını sileim
data = data.drop(['gonderici_id', 'file_name'], axis=1)
check_df(data)

def check_zero_ratio(dataframe, as_dict=False): # Changed parameter name to as_dict
    print("################## Zero Ratio Analysis ##################")
    zero_counts = (dataframe == 0).sum()
    zero_ratio = (zero_counts / len(dataframe)) * 100

    zero_columns = []  # Sıfır oranı olan değişkenleri toplamak için bir liste

    for col, count, ratio in zip(zero_counts.index, zero_counts.values, zero_ratio.values):
        print(f"'{col}': {count} zeros ({ratio:.2f}%)")
        if count > 0:  # Eğer sıfır oranı varsa
            zero_columns.append(col)
    print("#######################################################")

    if as_dict: # Using the new parameter name
        return dict(zip(zero_columns, zero_ratio))

    # Sıfır oranı olan değişkenleri içeren yeni bir DataFrame oluştur
    zero_data = dataframe[zero_columns]
    return zero_data

# Zero Ratio Analysis yap ve sıfır oranına sahip değişkenlerden oluşan bir DataFrame oluştur
zero_data = check_zero_ratio(data)
check_df(zero_data)
zero_data.to_csv('giden_zero.csv', index=False)

#droop columns that has zero ratio higher than %65

high_zero_ratio_cols = check_zero_ratio(zero_data, as_dict=True)
print(high_zero_ratio_cols)
columns_to_delete = [col for col, ratio in high_zero_ratio_cols.items() if ratio > 63]
print(columns_to_delete)

data.drop(columns=columns_to_delete, inplace=True)
data.to_csv('giden_dropped_cleaned.csv', index=False)
check_df(data)
check_zero_ratio(data)

def identify_columns(dataframe):
    """
    Veri çerçevesindeki sütunları kategorik, numerik ve kardinal olarak tanımlar.
    Nümerik gibi görünen ancak kategorik olan sütunların tipini 'category' olarak değiştirir.

    Args:
      dataframe: Pandas DataFrame.

    Returns:
      cat_cols: Kategorik sütunların listesi.
      num_but_cat: Nümerik gibi görünen kategorik sütunların listesi.
      cat_but_car: Kardinal kategorik sütunların listesi.
      num_cols: Numerik sütunların listesi.
    """

    # Kategorik değişkenler
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]
    print("Kategorik Değişkenler:", cat_cols)

    # Nümerik gibi görünen fakat kategorik olan değişkenler
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < 10 and dataframe[col].dtypes in ["int64", "float64"]]
    print("Nümerik gibi görünen kategorik değişkenler:", num_but_cat)

    # Kategorik görünümlü fakat kardinalitesi yüksek olan değişkenler
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > 30 and str(dataframe[col].dtypes) in ["category", "object"]]
    print("Kardinal kategorik değişkenler:", cat_but_car)

    # Nümerik gibi görünen kategorik değişkenlerin tipini 'category' olarak değiştir
    for col in num_but_cat:
        dataframe[col] = dataframe[col].astype("category")

    # Final kategorik değişkenler listesi
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    print("Kategorik değişkenler (final):", cat_cols)

    # Numerik Değişkenler
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int64", "float64"]]
    num_cols = [col for col in num_cols if col not in cat_cols]
    print("Nümerik Değişkenler (final):", num_cols)

    return cat_cols, num_but_cat, cat_but_car, num_cols

cat_cols, num_but_cat, cat_but_car, num_cols = identify_columns(data)

print(num_cols)
print(cat_cols)

def cat_summary(dataframe, col):
    # Kategorik değişken özetini ekrana yazdırır
    print(pd.DataFrame({col: dataframe[col].value_counts(),
                        "Ratio": 100 * dataframe[col].value_counts() / len(dataframe)}))
    print("###############################################################")

def num_summary(dataframe, col):
    # Sayısal değişken özetini ekrana yazdırır
    quantiles = [0.25, 0.5, 0.75, 0.9, 1]
    print(dataframe[col].describe(quantiles).T)
    print("###########################################")

data.to_csv('giden_final.csv', index=False)

check_df(data)

# GÖRSELLEŞTİRME

# 2) Histogram ve Scatter Plot
def plot_histogram_scatter(dataframe, num_cols):
    """
    Sayısal sütunlar için histogram ve scatter plot çizimi.
    """
    for col in num_cols:
        plt.figure(figsize=(12, 5))

        # Histogram
        plt.subplot(1, 2, 1)
        plt.hist(dataframe[col], bins=20, color='skyblue', edgecolor='black')
        plt.title(f"{col} - Histogram")

        # Scatter Plot (Y ekseni değiştirilebilir)
        plt.subplot(1, 2, 2)
        plt.scatter(range(len(dataframe)), dataframe[col], alpha=0.5, color='orange')
        plt.title(f"{col} - Scatter Plot")

        plt.tight_layout()
        plt.show()

# Kullanımı:
plot_histogram_scatter(data, num_cols)

def plot_categorical_distributions(dataframe, cat_cols):
    """
    Kategorik değişkenlerin değer dağılımını bar grafiklerle gösterir.

    Args:
    - dataframe: pd.DataFrame - Veri seti.
    - cat_cols: list - Kategorik sütunların isimlerini içeren liste.
    - figsize: tuple - Her grafik için boyutlar (genişlik, yükseklik).

    Returns:
    - None
    """
    for col in cat_cols:
        plt.figure(figsize=(10, 5))
        sns.countplot(x=col, data=dataframe, order=dataframe[col].value_counts().index)
        plt.title(f"Distribution of {col}", fontsize=14)
        plt.xlabel(col, fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.xticks(rotation=45)
        plt.show()

plot_categorical_distributions(data, cat_cols)

# Aykırı değer analizi

def handle_outliers(dataframe, col,  method="cap"):
    """
    Aykırı değerleri tespit edip işlem uygulayan fonksiyon.

    Args:
    - dataframe: pd.DataFrame
    - col: str
    - classification: bool, sınıflandırma problemi için özel işlem gerektiğinde True yapın
    - method: str, "cap" (sınırlandırma) veya "remove" (silme) seçenekleri.
    """
    Q1 = dataframe[col].quantile(0.25)
    Q3 = dataframe[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = dataframe[(dataframe[col] < lower_bound) | (dataframe[col] > upper_bound)]
    num_outliers = len(outliers)

    print(f"{col} sütununda {num_outliers} aykırı değer bulundu.")

    if method == "remove":
        # Aykırı değerleri silme (yalnızca regresyon için önerilir)
        dataframe = dataframe[(dataframe[col] >= lower_bound) & (dataframe[col] <= upper_bound)]
    elif method == "cap":
        # Aykırı değerleri sınırlandırma
        dataframe[col] = np.where(dataframe[col] < lower_bound, lower_bound,
                                  np.where(dataframe[col] > upper_bound, upper_bound, dataframe[col]))

    return dataframe

# Tüm sayısal sütunlarda aykırı değer işlemi
for col in num_cols:
    data = handle_outliers(data, col, method="cap")

# PROBLEM 1: GÖNDERİM MAALİYETİ TAHMİNİ

'''
İş problemi olarak bir ihracatın toplam maaliyetini tahmin etmek istiyoruz.
Nümerik bir değişken tahmin edeceğimiz için bu bir regresyon problemidir.
Problem üzerinde extra analiz ve değişken mühendislikleri uygulamaya çalışalım.

Şu ana kadar veriyi toparladık, aykırı değerleri eksikleri ve anlamsız sıfır değerlerini temizledik.
Mesafe tabanlı algoritmalar için encoding ve scaling gibi işlemler gerçekleştirmeliyiz.
data_supervised üzerinden ilerleyeceğiz.
'''

data_supervised = data.copy()
#check_df(data_supervised)

# DEĞİŞKEN MÜHENDİSLİĞİ - FEATURE ENRINEERING

# Toplam maaliyet adında tahmin edeceğim yeni değişkeni oluşturalım.
data_supervised["Toplam_Maaliyet"] = data_supervised["Toplam_yurt_ici_harcamalar"] + data_supervised["Toplam_yurt_disi_harcamalar"]
#check_df(data_supervised)

# Paket başına maliyet
data_supervised['paket_basi_maliyet'] = data_supervised['Toplam_Maaliyet'] / data_supervised['Kap_adedi']

# Paket başına ağırlık
data_supervised['paket_basi_agirlik'] = data_supervised['Net_agirlik'] / data_supervised['Kap_adedi']

# Ağırlık yoğunluğu
data_supervised['agirlik_yogunlugu'] = data_supervised['Brut_agirlik'] / data_supervised['Kap_adedi']

# Nakliye ve sigorta oranları
data_supervised['nakliye_orani'] = data_supervised['Nakliye'] / data_supervised['Toplam_Maaliyet']
data_supervised['sigorta_orani'] = data_supervised['Sigorta'] / data_supervised['Toplam_Maaliyet']

#Güncellenmiş dataframe üzerinden nümerik güncelleyelim
num_cols.append(["Toplam_Maaliyet", "paket_basi_maliyet", "paket_basi_agirlik", "agirlik_yogunlugu", "nakliye_orani", "sigorta_orani"])
#

check_df(data_supervised)

def target_encode(dataframe, cat_cols, target_col):
    dataframe_encoded = dataframe.copy()

    for col in cat_cols:
        if col != target_col:  # Only encode if the column is not the target
            encoding_map = dataframe_encoded.groupby(col)[target_col].mean()
            dataframe_encoded[f'{col}_encoded'] = dataframe_encoded[col].map(encoding_map).astype('float64')
            # Orijinal sütunu kaldırmak istiyorsanız:
            dataframe_encoded.drop(columns=[col], inplace=True)

    print(f"Target encoding tamamlandı. Yeni sütun sayısı: {dataframe_encoded.shape[1]}")
    return dataframe_encoded

df_target_encoded = target_encode(data_supervised, cat_cols, 'Toplam_Maaliyet')

def correlation_heatmap(dataframe, figsize=(12, 8), annot=True, cmap='coolwarm'):
    """
    Veri seti için korelasyon matrisi oluşturur ve bir ısı haritası olarak görselleştirir.

    Args:
    - dataframe: pd.DataFrame
        Korelasyon analizi yapılacak veri seti.
    - figsize: tuple, default=(12, 8)
        Grafik boyutu.
    - annot: bool, default=True
        Her hücrede korelasyon değerlerini göstermek için.
    - cmap: str, default='coolwarm'
        Isı haritasında kullanılacak renk haritası.

    Returns:
    - None
        Korelasyon matrisi grafiğini gösterir.
    """
    # Korelasyon matrisi hesaplama
    corr_matrix = dataframe.corr()

    # Isı haritası oluşturma
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=annot, cmap=cmap, fmt=".2f", cbar=True)
    plt.title("Correlation Heatmap", fontsize=16)
    plt.show()

correlation_heatmap(df_target_encoded)

def drop_correlated_features(dataframe, target, threshold=0.9, lowest_n=3):
    """
    Bir veri setinde:
    1. Birbirleriyle yüksek korelasyona sahip sütunları threshold değerine göre çıkarır.
    2. Hedef değişkenle en düşük korelasyona sahip n sütunu çıkarır.

    Args:
    - dataframe: pd.DataFrame
        Veri seti.
    - target: str
        Hedef değişkenin adı.
    - threshold: float, default=0.9
        Yüksek korelasyon eşiği (sütunlar arası).
    - lowest_n: int, default=3
        Hedef değişkenle en düşük korelasyona sahip sütunların sayısı.

    Returns:
    - pd.DataFrame
        Güncellenmiş veri seti.
    """
    # Adım 1: Birbirleriyle olan yüksek korelasyonu kaldır
    # Exclude the target variable from correlation calculation
    features = dataframe.drop(columns=[target]).columns
    corr_matrix = dataframe[features].corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Threshold üstü korelasyona sahip sütunları bul
    to_drop_high_corr = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]

    # Adım 2: Hedef değişkenle en düşük korelasyona sahip sütunları bul
    correlations = dataframe.corr()[target].sort_values(ascending=True)
    to_drop_low_corr = correlations.head(lowest_n).index.tolist()
    print(to_drop_low_corr)

    # Sütunları birleştir ve çıkar
    to_drop = list(set(to_drop_high_corr + to_drop_low_corr))
    dataframe = dataframe.drop(columns=to_drop)

    print(f"Şu sütunlar çıkarıldı: {to_drop}")
    return dataframe

# Kullanım:
df_cleaned_supervised = drop_correlated_features(df_target_encoded, target="Toplam_Maaliyet", threshold=0.80, lowest_n=5)

from sklearn.preprocessing import StandardScaler

# Özellikler ve hedef değişkenin ayrılması
X = df_cleaned_supervised.drop(columns=['Toplam_Maaliyet'])
y = df_cleaned_supervised['Toplam_Maaliyet']

# Özellikleri scale etme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Scale edilmiş özellikleri DataFrame olarak kaydetme
df_supervised_final = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

# Hedef değişkeni tekrar ekleme
df_supervised_final['Toplam_Maaliyet'] = y



# Sonuçları kontrol edelim
check_df(df_supervised_final)

def train_evaluate_linearRegression(df, target, test_size=0.2, random_state=42):
    """
    Linear Regression modeli eğitip performansını değerlendirir.

    Args:
    - df: pd.DataFrame
        Scale edilmiş ve temizlenmiş veri seti.
    - target: str
        Hedef değişkenin adı.
    - test_size: float, default=0.2
        Test veri setinin oranı.
    - random_state: int, default=42
        Rastgelelik için kullanılan seed değeri.

    Returns:
    - None
        Performans metriklerini yazdırır.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np

    # Hedef ve özelliklerin ayrılması
    X = df.drop(columns=[target])
    y = df[target]

    # Veriyi eğitim ve test setlerine ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Model eğitimi
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Tahmin
    y_pred = model.predict(X_test)

    # Performans metrikleri
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Sonuçları yazdırma
    print("Model Performansı:")
    print(f"- MSE: {mse:.4f}")
    print(f"- RMSE: {rmse:.4f}")
    print(f"- R^2: {r2:.4f}")

    return model

train_evaluate_linearRegression(df_supervised_final, "Toplam_Maaliyet")

def train_evaluate_LassoRegression(df, target, test_size=0.2, random_state=42, alpha=1.0):
    """
    Lasso Regression modeli eğitip performansını değerlendirir ve baseline modelle karşılaştırır.

    Args:
    - df: pd.DataFrame
        Scale edilmiş ve temizlenmiş veri seti.
    - target: str
        Hedef değişkenin adı.
    - test_size: float, default=0.2
        Test veri setinin oranı.
    - random_state: int, default=42
        Rastgelelik için kullanılan seed değeri.
    - alpha: float, default=1.0
        Regularization parametresi. Daha yüksek alpha, daha güçlü regularization.

    Returns:
    - model: sklearn Lasso modeli
        Eğitilmiş Lasso Regression modeli.
    """
    from sklearn.linear_model import Lasso
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np

    # Hedef ve özelliklerin ayrılması
    X = df.drop(columns=[target])
    y = df[target]

    # Veriyi eğitim ve test setlerine ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Model tanımlama
    model = Lasso(alpha=alpha, random_state=random_state)
    model.fit(X_train, y_train)

    # Tahmin
    y_pred = model.predict(X_test)

    # Performans metrikleri
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Baseline Model Performansı
    baseline_prediction = np.mean(y_train)
    baseline_predictions = [baseline_prediction] * len(y_test)

    baseline_mse = mean_squared_error(y_test, baseline_predictions)
    baseline_rmse = np.sqrt(baseline_mse)
    baseline_r2 = r2_score(y_test, baseline_predictions)

    # Sonuçları yazdırma
    print("Lasso Regression Model Performansı:")
    print(f"- MSE: {mse:.4f}")
    print(f"- RMSE: {rmse:.4f}")
    print(f"- R^2: {r2:.4f}")

    print("\nBaseline Model Performansı:")
    print(f"- MSE: {baseline_mse:.4f}")
    print(f"- RMSE: {baseline_rmse:.4f}")
    print(f"- R^2: {baseline_r2:.4f}")

    return model


# Lasso Regression ile model eğitimi ve değerlendirme
lasso_model = train_evaluate_LassoRegression(
    df=df_supervised_final,
    target="Toplam_Maaliyet",
    test_size=0.2,
    random_state=42,
    alpha=0.1
)

def train_evaluate_KNNRegressor(df, target, test_size=0.2, random_state=42, n_neighbors=5):
    """
    K-Nearest Neighbors (KNN) Regressor modeli eğitip performansını değerlendirir ve baseline modelle karşılaştırır.

    Args:
    - df: pd.DataFrame
        Scale edilmiş ve temizlenmiş veri seti.
    - target: str
        Hedef değişkenin adı.
    - test_size: float, default=0.2
        Test veri setinin oranı.
    - random_state: int, default=42
        Rastgelelik için kullanılan seed değeri.
    - n_neighbors: int, default=5
        KNN modelinde kullanılacak komşu sayısı.

    Returns:
    - model: sklearn KNeighborsRegressor
        Eğitilmiş KNN modeli.
    """
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np

    # Hedef ve özelliklerin ayrılması
    X = df.drop(columns=[target])
    y = df[target]

    # Veriyi eğitim ve test setlerine ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Model tanımlama
    model = KNeighborsRegressor(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)

    # Tahmin
    y_pred = model.predict(X_test)

    # Performans metrikleri
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Baseline Model Performansı
    baseline_prediction = np.mean(y_train)
    baseline_predictions = [baseline_prediction] * len(y_test)

    baseline_mse = mean_squared_error(y_test, baseline_predictions)
    baseline_rmse = np.sqrt(baseline_mse)
    baseline_r2 = r2_score(y_test, baseline_predictions)

    # Sonuçları yazdırma
    print("K-Nearest Neighbors Regressor Model Performansı:")
    print(f"- MSE: {mse:.4f}")
    print(f"- RMSE: {rmse:.4f}")
    print(f"- R^2: {r2:.4f}")

    print("\nBaseline Model Performansı:")
    print(f"- MSE: {baseline_mse:.4f}")
    print(f"- RMSE: {baseline_rmse:.4f}")
    print(f"- R^2: {baseline_r2:.4f}")

    return model

# K-Nearest Neighbors Regressor ile model eğitimi ve değerlendirme
knn_model = train_evaluate_KNNRegressor(
    df=df_supervised_final,
    target="Toplam_Maaliyet",
    test_size=0.2,
    random_state=42,
    n_neighbors=5
)

def train_evaluate_RandomForest_with_CV(df, target, cv=5, random_state=42, n_estimators=50, max_depth=None):
    """
    Random Forest modeli için cross-validation ile eğitim ve değerlendirme yapar.

    Args:
    - df: pd.DataFrame
        Scale edilmiş ve temizlenmiş veri seti.
    - target: str
        Hedef değişkenin adı.
    - cv: int, default=5
        Cross-validation kat sayısı.
    - random_state: int, default=42
        Rastgelelik için kullanılan seed değeri.
    - n_estimators: int, default=100
        Random Forest'taki ağaç sayısı.
    - max_depth: int, optional
        Ağaçların maksimum derinliği.

    Returns:
    - None
        Cross-validation performans metriklerini yazdırır.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import make_scorer, mean_squared_error, r2_score
    import numpy as np

    # Hedef ve özelliklerin ayrılması
    X = df.drop(columns=[target])
    y = df[target]

    # Model tanımlama
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

    # Performans metrikleri için scorer fonksiyonları
    mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
    r2_scorer = make_scorer(r2_score)

    # Cross-validation ile MSE ve R² hesaplama
    mse_scores = cross_val_score(model, X, y, cv=cv, scoring=mse_scorer)
    r2_scores = cross_val_score(model, X, y, cv=cv, scoring=r2_scorer)

    # MSE'yi pozitif hale getirme ve RMSE hesaplama
    mse_scores = -mse_scores
    rmse_scores = np.sqrt(mse_scores)

    # Sonuçları yazdırma
    print(f"Random Forest Cross-Validation Performansı ({cv}-fold):")
    print(f"- Ortalama MSE: {np.mean(mse_scores):.4f}")
    print(f"- Ortalama RMSE: {np.mean(rmse_scores):.4f}")
    print(f"- Ortalama R²: {np.mean(r2_scores):.4f}")


    # Baseline model: Ortalama ile tahmin
    baseline_prediction = np.mean(y)
    baseline_predictions = [baseline_prediction] * len(y)

    # Baseline MSE, RMSE ve R²
    baseline_mse = mean_squared_error(y, baseline_predictions)
    baseline_rmse = np.sqrt(baseline_mse)
    baseline_r2 = r2_score(y, baseline_predictions)

    print("\nBaseline Model Performansı:")
    print(f"- MSE: {baseline_mse:.4f}")
    print(f"- RMSE: {baseline_rmse:.4f}")
    print(f"- R²: {baseline_r2:.4f}")

    return model

# Random Forest ile cross-validation kullanarak model eğitimi ve değerlendirme
rf_model = train_evaluate_RandomForest_with_CV(
    df=df_supervised_final,
    target='Toplam_Maaliyet',
    cv=5,
    random_state=42,
    n_estimators=100,
    max_depth=5
)

def train_evaluate_XGBoost(df, target, test_size=0.2, random_state=42, n_estimators=100, max_depth=3, learning_rate=0.1):
    """
    XGBoost modeli eğitip performansını değerlendirir ve baseline modelle karşılaştırır.

    Args:
    - df: pd.DataFrame
        Scale edilmiş ve temizlenmiş veri seti.
    - target: str
        Hedef değişkenin adı.
    - test_size: float, default=0.2
        Test veri setinin oranı.
    - random_state: int, default=42
        Rastgelelik için kullanılan seed değeri.
    - n_estimators: int, default=100
        Ağaç sayısı.
    - max_depth: int, default=3
        Ağaçların maksimum derinliği.
    - learning_rate: float, default=0.1
        Modelin öğrenme oranı.

    Returns:
    - model: xgboost.XGBRegressor
        Eğitilmiş XGBoost modeli.
    """
    from xgboost import XGBRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np

    # Hedef ve özelliklerin ayrılması
    X = df.drop(columns=[target])
    y = df[target]

    # Veriyi eğitim ve test setlerine ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Model tanımlama
    model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=random_state)
    model.fit(X_train, y_train)

    # Tahmin
    y_pred = model.predict(X_test)

    # Performans metrikleri
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Baseline Model Performansı
    baseline_prediction = np.mean(y_train)
    baseline_predictions = [baseline_prediction] * len(y_test)

    baseline_mse = mean_squared_error(y_test, baseline_predictions)
    baseline_rmse = np.sqrt(baseline_mse)
    baseline_r2 = r2_score(y_test, baseline_predictions)

    # Sonuçları yazdırma
    print("XGBoost Regressor Model Performansı:")
    print(f"- MSE: {mse:.4f}")
    print(f"- RMSE: {rmse:.4f}")
    print(f"- R^2: {r2:.4f}")

    print("\nBaseline Model Performansı:")
    print(f"- MSE: {baseline_mse:.4f}")
    print(f"- RMSE: {baseline_rmse:.4f}")
    print(f"- R^2: {baseline_r2:.4f}")

    return model


# XGBoost ile model eğitimi ve değerlendirme
xgb_model = train_evaluate_XGBoost(
    df=df_supervised_final,
    target="Toplam_Maaliyet",
    test_size=0.2,
    random_state=42,
    n_estimators=75,
    max_depth=4,
    learning_rate=0.1
)

# PROBLEM 2: UNSUPERVISED LEARNING = ANOMALİ TESPİTİ

'''
2.bir iş problemi olarak kümeleme gibi unsupervised learning algoritmaları ile anomali tespiti yapacağız.

Şu ana kadar veriyi toparlaamak için bayağı uğraştık , neleri yaptık:

- Eksik Değerleri temizledik
- Sıfır sütunlarını istatistiksel olarak ve işbilgisine dayanarak ayıkladık
- Aykırı değer analizi yaptık
- Encode ederek kategorik değişkenleri algoritmalar tarafından kullanılabilir hale getirdil
- Korelasyon analizi ile gereksiz değişkenlerden kurtulduk
- Scale ederek mesafe tabanlı algoritmalarda modellerin iyi çalışmasını garantileyelim
yeniden scale ediyoruz çünkü unsupervised learningte target değişkeni yok, ilk başta onu almamıştık





Temizlediğimiz veri üzerinden farklı algoritmalar ile anomali tespiti yapmaya çalışalım ve PCA ile boyutu düşürüp görselleştirelim
'''

df_unsupervised = pd.read_csv("df_final.csv")

def scale_features_unsupervised(df, features_to_scale):
    """
    Scales specified features in a DataFrame using StandardScaler for unsupervised learning.

    Args:
        df (pd.DataFrame): The DataFrame containing the features to be scaled.
        features_to_scale (list): A list of column names representing the features to scale.

    Returns:
        pd.DataFrame: The DataFrame with scaled features.
    """

    # Create a StandardScaler object
    scaler = StandardScaler()

    # Fit the scaler to the specified features and transform them
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    return df

df_unsupervised_final = scale_features_unsupervised(df_unsupervised, df_unsupervised.columns)

# Sonuçları kontrol edelim
check_df(df_supervised_final)
df_unsupervised_final.to_csv("df_unsupervised_final.csv", index=False)
#

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def train_evaluate_KMeans(df, max_clusters=10, random_state=42):
    """
    Trains and evaluates a K-Means clustering model by determining the optimal number of clusters using the Elbow method.

    Parameters:
    - df: DataFrame containing the data.
    - max_clusters: Maximum number of clusters to test for the Elbow method.
    - random_state: Random state for reproducibility.

    Returns:
    - kmeans: Trained K-Means model with the optimal number of clusters.
    - optimal_k: Optimal number of clusters determined by the Elbow method.
    - silhouette_avg: Silhouette score of the clustering.
    - cluster_centers: Coordinates of the cluster centers.
    """
    # Selecting numerical columns
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    X = df[numerical_cols]

    # Calculating distortions for the Elbow method
    distortions = []
    cluster_range = range(1, max_clusters + 1)

    for k in cluster_range:
        kmeans_temp = KMeans(n_clusters=k, random_state=random_state)
        kmeans_temp.fit(X)  # Fit on the entire data for elbow method
        distortions.append(kmeans_temp.inertia_)

    # Plotting the Elbow graph
    plt.figure(figsize=(8, 5))
    plt.plot(cluster_range, distortions, marker='o', linestyle='--')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distortion (Inertia)')
    plt.grid(True)
    plt.show()

    # Automatically determining the optimal number of clusters
    deltas = np.diff(distortions)
    second_deltas = np.diff(deltas)
    optimal_k = np.argmin(second_deltas) + 2  # +2 accounts for second derivative index offset

    print(f"Optimal number of clusters (k) determined by Elbow method: {optimal_k}")

    # Train K-Means model with the optimal number of clusters on the entire data
    kmeans = KMeans(n_clusters=optimal_k, random_state=random_state)
    kmeans.fit(X)  # Fit on entire data

    # Predict clusters on the entire data
    labels = kmeans.labels_

    # Calculate silhouette score on the entire data
    silhouette_avg = silhouette_score(X, labels)

    # Apply PCA to reduce dimensionality to 2 for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)  # Use the original data (X)

    # Plot the clusters
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.7) # Use labels for the entire data
    plt.title('K-Means Clustering Visualization with PCA')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.show()

    return kmeans, optimal_k, silhouette_avg, kmeans.cluster_centers_

kmeans_model, optimal_k, silhouette_avg, cluster_centers = train_evaluate_KMeans(
    df_unsupervised_final, max_clusters=10
)
print("Optimal Number of Clusters (k):", optimal_k)
print("Silhouette Score:", silhouette_avg)
print("Cluster Centers:", cluster_centers)


# Apply PCA to reduce dimensionality to 2 for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)  # Use the original data (X)

# Plot the clusters
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_model.labels_, cmap='viridis', alpha=0.7)
plt.title('K-Means Clustering Visualization with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA  # Import PCA

def train_evaluate_HierarchicalKMeans(df, max_clusters=10, random_state=42):
    """
    Trains and evaluates a hierarchical clustering model using the optimal number of clusters
    determined by the Elbow method and visualizes the clusters using PCA.

    Parameters:
    - df: DataFrame containing the data.
    - max_clusters: Maximum number of clusters to test for the Elbow method.
    - random_state: Random state for reproducibility.

    Returns:
    - hierarchical_model: Trained AgglomerativeClustering model.
    - optimal_k: Optimal number of clusters determined by the Elbow method.
    - silhouette_avg: Silhouette score of the clustering.
    """
    # Selecting numerical columns
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    X = df[numerical_cols]

    # Calculating distortions for the Elbow method using K-Means
    distortions = []
    cluster_range = range(1, max_clusters + 1)

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)

    # Plotting the Elbow graph
    plt.figure(figsize=(8, 5))
    plt.plot(cluster_range, distortions, marker='o', linestyle='--')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distortion (Inertia)')
    plt.grid(True)
    plt.show()

    # Automatically determine the optimal number of clusters (k)
    deltas = np.diff(distortions)
    second_deltas = np.diff(deltas)
    optimal_k = np.argmin(second_deltas) + 2  # +2 accounts for the second derivative offset

    print(f"Optimal number of clusters (k) determined by Elbow method: {optimal_k}")

    # Train Hierarchical Clustering model with the optimal number of clusters
    hierarchical_model = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
    hierarchical_labels = hierarchical_model.fit_predict(X)

    # Calculate silhouette score
    silhouette_avg = silhouette_score(X, hierarchical_labels)

    # PCA for visualization
    pca = PCA(n_components=2)  # Reduce to 2 principal components
    X_pca = pca.fit_transform(X)

    # Plot the clusters on PCA1 and PCA2
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=hierarchical_labels, cmap='viridis', alpha=0.7)
    plt.title('Hierarchical Clustering Visualization with PCA')
    plt.xlabel('Principal Component 1 (PCA1)')
    plt.ylabel('Principal Component 2 (PCA2)')
    plt.show()

    return hierarchical_model, optimal_k, silhouette_avg

# ... (Rest of the code remains the same) ...

hierarchical_model, optimal_k, silhouette_avg = train_evaluate_HierarchicalKMeans(
    df_unsupervised_final, max_clusters=5
)
print("Optimal Number of Clusters (k):", optimal_k)
print("Silhouette Score:", silhouette_avg)

from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
import numpy as np

def train_evaluate_IsolationForest(df, contamination=0.1, random_state=42):
    """
    Trains and evaluates an Isolation Forest model for anomaly detection.

    Parameters:
    - df: DataFrame containing the data.
    - contamination: The proportion of outliers in the data set.
    - random_state: Random state for reproducibility.

    Returns:
    - isolation_forest: Trained Isolation Forest model.
    - anomaly_labels: Labels predicted by the model (-1 for anomalies, 1 for normal points).
    - silhouette_avg: Silhouette score for the clustering of normal points and anomalies.
    """
    # Selecting numerical columns
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    X = df[numerical_cols]

    # Train Isolation Forest model
    isolation_forest = IsolationForest(contamination=contamination, random_state=random_state)
    isolation_forest.fit(X)

    # Predict anomaly labels
    anomaly_labels = isolation_forest.predict(X)

    # Convert anomaly labels to 0 (anomalies) and 1 (normal points) for silhouette calculation
    silhouette_labels = np.where(anomaly_labels == -1, 0, 1)

    # Calculate silhouette score if there are at least two clusters
    if len(set(silhouette_labels)) > 1:
        silhouette_avg = silhouette_score(X, silhouette_labels)
    else:
        silhouette_avg = None

    return isolation_forest, anomaly_labels, silhouette_avg

isolation_model, anomaly_labels, silhouette_avg = train_evaluate_IsolationForest(
    df_unsupervised_final, contamination=0.1
)
print("Silhouette Score:", silhouette_avg)
print("Anomalies Detected:", sum(anomaly_labels == -1))