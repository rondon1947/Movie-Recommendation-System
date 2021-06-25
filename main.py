import warnings

import pandas as pd
import seaborn as sns

sns.set_style('white')

warnings.filterwarnings('ignore')

columns_names = ["user_id", "item_id", "rating", "timestamp"]
df = pd.read_csv("data/u.data", sep='\t', names=columns_names)

movies_titles = pd.read_csv("data/u.item", sep="\|", header=None)

movies_titles = movies_titles[[0, 1]]

movies_titles.columns = ["item_id", "title"]

df = pd.merge(df, movies_titles, on="item_id")

ratings = pd.DataFrame(df.groupby("title").mean()['rating'])
ratings["no of ratings"] = pd.DataFrame(df.groupby('title').count()['rating'])

movieMatrix = df.pivot_table(index='user_id', columns='title', values='rating')


def predictMovies(movieName):
    movieRatings = movieMatrix[movieName]
    moviesSimilar = movieMatrix.corrwith(movieRatings)
    correlationWithMovie = pd.DataFrame(moviesSimilar, columns=['Correlation'])
    correlationWithMovie.dropna(inplace=True)
    correlationWithMovie = correlationWithMovie.join(ratings['no of ratings'])
    prediction = correlationWithMovie[correlationWithMovie['no of ratings'] > 100].sort_values('Correlation',
                                                                                               ascending=False)
    return prediction


predictions = predictMovies('Independence Day (ID4) (1996)')
print(predictions.head(15))
