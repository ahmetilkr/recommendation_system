###########################################
# Item-Based Collaborative Filtering
###########################################

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: User Movie Df'inin Oluşturulması
# Adım 3: Item-Based Film Önerilerinin Yapılması
# Adım 4: Çalışma Scriptinin Hazırlanması


######################################
# Adım 1: Veri Setinin Hazırlanması
######################################
import pandas as pd
pd.set_option('display.max_columns', 500)
movie = pd.read_csv(r"C:\Users\ahmet\Desktop\miuul\ders_materyalleri\recommender_systems-221002-142715\recommender_systems\datasets\movie_lens_dataset\movie.csv")
rating = pd.read_csv(r"C:\Users\ahmet\Desktop\miuul\ders_materyalleri\recommender_systems-221002-142715\recommender_systems\datasets\movie_lens_dataset\rating.csv")
df = movie.merge(rating, how="left", on="movieId")
# movie solda rating sağda olacak şekilde left join ile movieıd ye göre birleştir
df.head()





######################################
# Adım 2: User Movie Df'inin Oluşturulması
######################################

df.head()
df.shape

df["title"].nunique()   #seçilen sütunda benzersiz elemanların sayısını verir number of unique kısaltmasıdır

df["title"].value_counts().head()

comment_counts = pd.DataFrame(df["title"].value_counts())
rare_movies = comment_counts[comment_counts["count"] <= 1000].index
common_movies = df[~df["title"].isin(rare_movies)]    #isin() fonksiyonu sütun içinde belirlediğin bir şeyin olup olmadığını kontrol eder
common_movies.shape
common_movies["title"].nunique()
df["title"].nunique()

user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")

user_movie_df.shape
user_movie_df.columns



######################################
# Adım 3: Item-Based Film Önerilerinin Yapılması
######################################

movie_name = "Matrix, The (1999)"
movie_name = "Ocean's Twelve (2004)"
movie_name = user_movie_df[movie_name]     #istediğim filmin sütununu bana getir
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)
#df in tüm sütunlarına bak benim verdiğim sütun ile benzerliğini (korelasyonunu) hesapla corrwith() sonra bunu çoktan aza sırala


movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]
#sütun isimlerini yani film isimlerini pandas serisi olarak tuttuk daha sonra bunun içinde rastgele bir tanesi aldık örneklem aldık(sample(1)) ile
#bunu da values[0] pandas seriesten stringe çevirdik
movie_name = user_movie_df[movie_name]
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)


def check_film(keyword, user_movie_df):
    """
     yazdığımız keywordu barındıran bütün filmleri getirme fonksiyonu
    :param keyword: aranan kelime
    :param user_movie_df: hangi dataframe içinde aranacağı
    :return: film isimleri
    """
    return [col for col in user_movie_df.columns if keyword in col]

check_film("Insomnia", user_movie_df)



######################################
# Adım 4: Çalışma Scriptinin Hazırlanması
######################################

def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv(r"C:\Users\ahmet\Desktop\miuul\ders_materyalleri\recommender_systems-221002-142715\recommender_systems\datasets\movie_lens_dataset\movie.csv")
    rating = pd.read_csv(r"C:\Users\ahmet\Desktop\miuul\ders_materyalleri\recommender_systems-221002-142715\recommender_systems\datasets\movie_lens_dataset\rating.csv")
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["count"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()


def item_based_recommender(movie_name, user_movie_df):
    movie_name = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

item_based_recommender("Matrix, The (1999)", user_movie_df)

movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]

item_based_recommender(movie_name, user_movie_df)









