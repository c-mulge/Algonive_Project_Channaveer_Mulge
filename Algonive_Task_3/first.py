from google_play_scraper import reviews
import pandas as pd

result, _ = reviews(
    'com.instagram.android',
    lang='en',
    country='in',
    count=1000
)

df = pd.DataFrame(result)
df = df[['content','score','at']]
df.columns = ['review','rating','date']

print(df.head())