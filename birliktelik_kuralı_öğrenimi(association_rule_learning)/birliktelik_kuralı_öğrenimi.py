############################################
# ASSOCIATION RULE LEARNING (BİRLİKTELİK KURALI ÖĞRENİMİ)
############################################
import numpy as np
# 1. Veri Ön İşleme
# 2. ARL Veri Yapısını Hazırlama (Invoice-Product Matrix)
# 3. Birliktelik Kurallarının Çıkarılması
# 4. Çalışmanın Scriptini Hazırlama
# 5. Sepet Aşamasındaki Kullanıcılara Ürün Önerisinde Bulunmak



############################################
# 1. Veri Ön İşleme
############################################

import pandas as pd
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
# çıktının tek bir satırda olmasını sağlar.
pd.set_option('display.expand_frame_repr', False)
from mlxtend.frequent_patterns import apriori, association_rules


df_ = pd.read_excel(r"C:\Users\ahmet\Desktop\miuul\ders_materyalleri\recommender_systems-221002-142715\recommender_systems\datasets\online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")

df=df_.copy()

df.head()
df.describe().T
df.isnull().sum()




def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)        #quantile() fonksiyonu çeyreklik değerini verir yani küçükten büyüğe sıralandığında yüzde bir olan değeri
    quartile3 = dataframe[variable].quantile(0.99)         #küçükten büyüğe sıralandığında yüzde doksan dokuz olan değeri
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit)

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe


df = retail_data_prep(df)
df.isnull().sum()
df.describe().T



############################################
# 2. ARL Veri Yapısını Hazırlama (Invoice-Product Matrix)
############################################

df_fr=df[df["Country"]=="France"]

df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).head(20)

df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().iloc[0:5, 0:5]
#unstack() fonksiyonu groupby da en son yazılan indexi sütuna atar(pivot eder)  satır sütun kesişimine de Quantity sum gelir

df_fr.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().fillna(0).iloc[0:5, 0:5]
#fillna(0)  boş olan hücreleri (NaN) 0 ile doldurur.


df_fr.groupby(['Invoice', 'StockCode']). \
    agg({"Quantity": "sum"}). \
    unstack(). \
    fillna(0). \
    map(lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]
#apply fonksiyonuna satır ve sütun bilgisi veriyorduk ona göre geziyordu map ise hepsini bütün satırı sütunu gezer


def create_invoice_product_df(dataframe, id=False):
    """
    fonksiyonun amacı fonksiyon eğer direk dataframe parametresi ile çağrılınca sütun isimleri ürün isimleri;
    eğer id=True ile çağırılırsa sütun isimleri ürün id'leri
    """
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            map(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            map(lambda x: 1 if x > 0 else 0)

fr_inv_pro_df = create_invoice_product_df(df_fr)

fr_inv_pro_df = create_invoice_product_df(df_fr, id=True)



def check_id(dataframe, stock_code):
    """
    bu fonksiyon amacı ürün id'sinin hangi ürün ismine denk geldiğini öğrenmek
    """
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    #tolist() numpy dizilerini python listelerine çevirmek için kullanılır
    print(product_name)

check_id(df_fr, 10120)


############################################
# 3. Birliktelik Kurallarının Çıkarılması
############################################

frequent_itemsets = apriori(fr_inv_pro_df,    #apriori() fonksiyonu bize tüm ürün birliktelikleini ve olasılıklarını verir
                            min_support=0.01,
                            use_colnames=True)   #



frequent_itemsets.sort_values("support",ascending=False)   #sort_Values sıralama yapar("support") olduğu için supporta göre ascending=false olursa azalan sırada


rules=association_rules(frequent_itemsets,metric="support",min_threshold=0.01)  #daha önemli metrikleri verir confidence lift vs
#metric=supportta support değeri min_thresholddan büyük olsun deriz başka metriklere göre threshold uygulayacaksak onu yazarız

rules[(rules["support"]>0.05) & (rules["confidence"]>0.1) & (rules["lift"]>5)]


rules[(rules["support"]>0.05) & (rules["confidence"]>0.1) & (rules["lift"]>5)]. \
sort_values("confidence", ascending=False)

############################################
# 4. Çalışmanın Scriptini Hazırlama
############################################


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1Q
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe


def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)


def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)


def create_rules(dataframe, id=True, country="France"):
    dataframe = dataframe[dataframe['Country'] == country]
    dataframe = create_invoice_product_df(dataframe, id)
    frequent_itemsets = apriori(dataframe, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
    return rules


df = retail_data_prep(df)
rules = create_rules(df)


############################################
# 5. Sepet Aşamasındaki Kullanıcılara Ürün Önerisinde Bulunmak
############################################

# Örnek:
# Kullanıcı örnek ürün id: 22492


product_id = 22492
check_id(df, product_id)


sorted_rules = rules.sort_values("lift", ascending=False)

recommendation_list = []


for i, product in enumerate(sorted_rules["antecedents"]):
    for j in list(product):
        if j == product_id:
            recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])


recommendation_list[0:3]

check_id(df, 22326)


def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]


arl_recommender(rules, 22492, 1)
arl_recommender(rules, 22492, 2)
arl_recommender(rules, 22492, 3)












