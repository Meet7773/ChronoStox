import pandas as pd

df = pd.read_csv("../test/ticker.csv", encoding='utf-8')
df = pd.DataFrame(df)
# df = df[df['Country'] == 'India']
print(df, len(df))
