#############################
# Content Based Recommendation (İçerik Temelli Tavsiye)
#############################

#############################
# Film Overview'larına Göre Tavsiye Geliştirme
#############################

# 1. TF-IDF Matrisinin Oluşturulması
# 2. Cosine Similarity Matrisinin Oluşturulması
# 3. Benzerliklere Göre Önerilerin Yapılması
# 4. Çalışma Scriptinin Hazırlanması

#################################
# 1. TF-IDF Matrisinin Oluşturulması
#################################

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# https://www.kaggle.com/rounakbanik/the-movies-dataset
df = pd.read_csv(r"C:\Users\ahmet\Desktop\miuul\ders_materyalleri\recommender_systems-221002-142715\recommender_systems\datasets\the_movies_dataset\movies_metadata.csv",
                 low_memory=False)  # DtypeWarning kapamak icin
df.head()
df.shape


df["overview"].head()

tfidf = TfidfVectorizer(stop_words="english")

df[df['overview'].isnull()]
df['overview'] = df['overview'].fillna('')


tfidf_matrix = tfidf.fit_transform(df['overview'])  #5 adım da burada yapılır ve matrix döner

tfidf_matrix.shape

df['title'].shape

tfidf.get_feature_names_out()   #matrixteki sütun isimlerini(benzersiz kelimeleri verir)

tfidf_matrix.toarray()      #kesişimlerinde neler var


#################################
# 2. Cosine Similarity Matrisinin Oluşturulması
#################################

cosine_sim = cosine_similarity(tfidf_matrix,
                               tfidf_matrix)

cosine_sim.shape
cosine_sim[1]   #1. satırdaki filmin diğer tüm filmlerle benzerlik oranı



#################################
# 3. Benzerliklere Göre Önerilerin Yapılması
#################################

indices=pd.Series(df.index,index=df["title"])

indices.index.value_counts()

indices=indices[~indices.index.duplicated(keep="last")]
#duplicated kopya olanları bulur döndürür ~ olmayanları tutmasını sağlar keep=last ise sonuncuyu orijinal kabul et

indices["Cinderella"]

indices["Sherlock Holmes"]

movie_index = indices["Sherlock Holmes"]

cosine_sim[movie_index]

similarity_scores=pd.DataFrame(cosine_sim[movie_index],columns=["score"])


movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index


df['title'].iloc[movie_indices]



#################################
# 4. Çalışma Scriptinin Hazırlanması
#################################

def content_based_recommender(title, cosine_sim, dataframe):
    # index'leri olusturma
    indices = pd.Series(dataframe.index, index=dataframe['title'])
    indices = indices[~indices.index.duplicated(keep='last')]
    # title'ın index'ini yakalama
    movie_index = indices[title]
    # title'a gore benzerlik skorlarını hesapalama
    similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns=["score"])
    # kendisi haric ilk 10 filmi getirme
    movie_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
    return dataframe['title'].iloc[movie_indices]

content_based_recommender("Sherlock Holmes", cosine_sim, df)

content_based_recommender("The Matrix", cosine_sim, df)

content_based_recommender("The Godfather", cosine_sim, df)

content_based_recommender('The Dark Knight Rises', cosine_sim, df)


def calculate_cosine_sim(dataframe):
    tfidf = TfidfVectorizer(stop_words='english')
    dataframe['overview'] = dataframe['overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(dataframe['overview'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim


cosine_sim = calculate_cosine_sim(df)
content_based_recommender('The Dark Knight Rises', cosine_sim, df)



















