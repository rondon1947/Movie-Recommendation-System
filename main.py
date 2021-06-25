import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('white')

warnings.filterwarnings('ignore')

columns_names = ["user_id", "item_id", "rating", "timestamp"]
df = pd.read_csv("data/u.data", sep='\t', names=columns_names)
# print(df.head())
# print(df["user_id"].nunique())
# print(df["item_id"].nunique())

movies_titles = pd.read_csv("data/u.item", sep="\|", header=None)
# print(movies_titles)
# print(movies_titles.shape)

movies_titles = movies_titles[[0, 1]]
# print(movies_titles)
# print(movies_titles.shape)

movies_titles.columns = ["item_id", "title"]
# print(movies_titles)

df = pd.merge(df, movies_titles, on="item_id")
# print(df.tail())

# print(df.groupby("title").count()['rating'].sort_values(ascending=False))
ratings = pd.DataFrame(df.groupby("title").mean()['rating'])
ratings["no of ratings"] = pd.DataFrame(df.groupby('title').count()['rating'])
# print(ratings.sort_values(by='rating', ascending=True))

# plt.figure(figsize=(10, 6))
# plt.hist(ratings['no of ratings'], bins=70)
# plt.show()
#
# plt.hist(ratings['rating'], bins=70)
# plt.show()
#
# sns.jointplot(x='rating', y='no of ratings', data=ratings, alpha=0.5)
# plt.show()

# print(df.head())
movieMatrix = df.pivot_table(index='user_id', columns='title', values='rating')
# print(movieMatrix.head())

starWarsUserRatings = movieMatrix['Star Wars (1977)']
# print(starWarsUserRatings.head())

moviesSimilarToStarWars = movieMatrix.corrwith(starWarsUserRatings)
correlationWithStarWars = pd.DataFrame(moviesSimilarToStarWars, columns=['Correlation'])
# print(correlationWithStarWars)
correlationWithStarWars.dropna(inplace=True)
# print(correlationWithStarWars)
correlationWithStarWars = correlationWithStarWars.sort_values('Correlation', ascending=False)
# print(correlationWithStarWars.head(15))
correlationWithStarWars = correlationWithStarWars.join(ratings['no of ratings'])
# print(correlationWithStarWars)
# print(correlationWithStarWars[correlationWithStarWars['no of ratings'] > 100].sort_values('Correlation', ascending=False))


def predictMovies(movieName):
    movieRatings = movieMatrix[movieName]
    moviesSimilar = movieMatrix.corrwith(movieRatings)
    correlationWithMovie = pd.DataFrame(moviesSimilar, columns=['Correlation'])
    correlationWithMovie.dropna(inplace=True)
    correlationWithMovie = correlationWithMovie.join(ratings['no of ratings'])
    prediction = correlationWithMovie[correlationWithMovie['no of ratings'] > 100].sort_values('Correlation',
                                                                                               ascending=False)
    return prediction


predictions = predictMovies('Liar Liar (1997)')
print(predictions.head(15))
