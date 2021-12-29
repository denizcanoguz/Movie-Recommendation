##################### HYBRID #######################
# İş Problemi:
# ID'si verilen kullanıcı için item-based ve user-based recommender yönemlerini
# kullanarak tahmin yapınız

# Değişkenler:
# movie.csv
# movieID - eşsiz film numarası (UniqueID)
# title - Film adı

# rating.csv
# userID - Eşsiz kullanıcı numarası (UniqueID)
# movieID - Eşsiz film numarası (UniqueID)
# rating - Kullanıcı tarafından filme verilen puan
# timestamp - Değerlendirme tarihi
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
#########################################################
# Görev 1: Veri Hazırlama işlemlerini gerçekleştiriniz
#########################################################
movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
movie.head()
rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
rating.head()
df = movie.merge(rating, how="left", on="movieId")
df.head()
# comment_counts dataframe`i oluşturuluyor. Bunun içinde unique "title" bazında kaç adet olduğunu
# gösteriyor. Yani aslında film başına yorum sayılarını getiriyor:
comment_counts = pd.DataFrame(df["title"].value_counts())
comment_counts.head()
# rare_movies serisi oluşturuluyor. Bunun içinde, comment_count dataframe'de 5000'den küçük
# yorumu olan title'ları seçiyor ve kaydediyor:
rare_movies = comment_counts[comment_counts["title"] <= 5000].index
rare_movies[0:5]
# common_movies dataframe'de ise 5000'den fazla yorum alan filmler yer alıyor:
# İlk oluşturulan dataframe'den, rare_movies`da "title"`si geçen filmleri eliyoruz:
common_movies = df[~df["title"].isin(rare_movies)]
common_movies.head()
# User-Movie dataframe oluşturulması:
user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
user_movie_df.head()

#######################################################################
# Görev 2: Öneri yapılacak kullanıcının izlediği filmleri belirleyiniz
#######################################################################

# Rastgele bir izleyici alıyoruz:
random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)
# 5979 userID'li izleyici seçildi
#tüm filmler için kullanıcının verdiği oylar
random_user_df = user_movie_df[user_movie_df.index == random_user]
random_user_df.head()
# kullanıcının izlediği filmlerin seçilmesi
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
movies_watched[0:5]
# kullanıcının izlediği film sayısı :31 adet
len(movies_watched)
######################################################################################
# Görev 3: Aynı filmleri izleyen diğer kullanıcıların verisine ve ID'lerine erişiniz
#####################################################################################

# Aynı filmleri izleyen kullanıcıların seçilmesi
movies_watched_df = user_movie_df[movies_watched]
movies_watched_df.head()
movies_watched_df.shape
# Bütün kullanıcıların kaç tane film izlediği bilgisi
user_movie_count = movies_watched_df.T.notnull().sum().reset_index()
user_movie_count.head()
user_movie_count.columns = ["userId", "movie_count"]

#######################################################################
# Görev 4: Öneri yapılacak kullanıcı ile en benzer kullanıcıları belirleyiniz
#######################################################################

# Kaç kişi kullanıcının izlediği filmlerin hepsini izlemiş : 537
user_movie_count[user_movie_count["movie_count"] == 31].count()
# Kullanıcı ile en az yüzde 60 aynı filmi izlemiş kişilerin filtrelenmesi
perc_60 = (len(movies_watched)) * 60/100
users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc_60]["userId"]
users_same_movies.head()
# Kaç kişi, 5979 userID'li izleyicimiz ile en az yüzde 60 dan fazla ortak film izlemiş? 23656 kişi.
users_same_movies.count()
# Ortak, userID'ler ve izledikleri ortak filmlere verdikleri puan dataframe'si:
final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],random_user_df[movies_watched]])
final_df.head()
# Tanspozunu alıp, unstack ediyoruz, sonra sıralıyoruz ve iki defa var olan verileri siliyoruz:
corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
# Dataframe çevirip kolon isimlendirmesi yapıyoruz:
corr_df = pd.DataFrame(corr_df, columns=["corr"])
# Index isimlendirme:
corr_df.index.names = ['user_id_1', 'user_id_2']
# Yeni index atama:
corr_df = corr_df.reset_index()
corr_df.head()
# Seçtiğimiz random_user ile korelasyonu 0.65'den yüksek olan user_id_2'leri ve korelasyonlarını alıyoruz:
top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][["user_id_2", "corr"]].reset_index(drop=True)
# Korelasyona gore azalan sıralama
top_users = top_users.sort_values(by='corr', ascending=False)
# user_id_2`yi  user_id isimlendirmesi ile değiştirme
top_users.rename(columns={"user_id_2": "userId"}, inplace=True)
top_users.head()
# Filmleri getirelim
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')
top_users_ratings.head()
top_users_ratings.tail()
# Kullanıcıyı çıkaralım
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]
top_users_ratings.head()
top_users_ratings.tail()

################################################################################################
# Görev 5: Weighted Average Recommendation Score`u hesaplayınız ve ilk 5 filmi tutunuz
################################################################################################

top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']
recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()
# Önerebileceğimiz kaç film var? 6985.
recommendation_df[["movieId"]].nunique()
# Korelasyonu en yüksek olan 5 film ID'leri
movies_to_be_recommend = recommendation_df.sort_values("weighted_rating", ascending=False).head(5)
final_recommendation = movies_to_be_recommend.merge(movie[["movieId", "title"]])["title"]
# Korelasyonu en yüksek olan 5 film ismi
final_recommendation


#####################################################################
# Görev 6:
# Kullanıcının izlediği filmlerden en son en yüksek puan verdiği
# filmin adına göre item-based öneri yapınız
# 5 öneri user-based
# 5 öneri item-based olacak şekilde 10 öneri yapınız
####################################################################

# Kullanıcının en son  5.0 olarak oyladığı film
movie_id = rating[(rating["userId"] == random_user) & (rating["rating"] == 5.0)].sort_values(by="timestamp",
                                                                                             ascending=False)["movieId"][0:1].values[0]
def check_film(id, movie):
    return [col for col in movie.values if id in col]
check_film(110, movie)
# 110 id`li filmin adı  Braveheart (1995)
movie_name = "Braveheart (1995)"
# Braveheart (1995) filmini tüm userID'lerin verdiği ratingler:
movie_name = user_movie_df[movie_name]
# Braveheart (1995) filmi ile en çok benzerlik gösteren 6 film (ilki kendisi):
movies_from_item_based = user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(6)
# Braveheart (1995) filmi ile en çok benzerlik gösteren 5 film:
movies_from_item_based[1:6].index
final_recommendation = final_recommendation.reset_index()
movies_from_item_based = movies_from_item_based[1:6]
movies_from_item_based = movies_from_item_based.reset_index()
#5 use-based, 5 item-based film önerisi
recommend = pd.concat([final_recommendation, movies_from_item_based], ignore_index=True, sort=False)["title"]
