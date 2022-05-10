import pandas as pd
import  requests

res = requests.get('https://api.themoviedb.org/3/movie/top_rated?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US&page=1')
data = pd.DataFrame(res.json()['results'])
print(data.shape)

df = pd.DataFrame()
for i in range(1,429):
    res = requests.get('https://api.themoviedb.org/3/movie/top_rated?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US&page={}'.format(i))
    temp_df = pd.DataFrame(res.json()['results'])[['id','title','overview','release_date','popularity','vote_average','vote_count']]
    df = df.append(temp_df,ignore_index = True)
print(df.shape)
