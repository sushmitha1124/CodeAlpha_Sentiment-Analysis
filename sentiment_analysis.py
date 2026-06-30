import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
if not os.path.exists("charts"):
    os.makedirs("charts")
df = pd.read_csv("Reviews.csv")

print("FIRST 5 ROWS")
print(df.head())

print("\nDATASET INFO")
print(df.info())

print("\nDATASET SHAPE")
print(df.shape)

print("\nMISSING VALUES")
print(df.isnull().sum())

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

print("\nData Cleaning Completed!")
def sentiment(score):
    if score >= 4:
        return "Positive"
    elif score == 3:
        return "Neutral"
    else:
        return "Negative"

df["Sentiment"] = df["Score"].apply(sentiment)

print("\nFIRST 5 ROWS WITH SENTIMENT")
print(df[["Score", "Sentiment"]].head())
plt.figure(figsize=(6,5))
sns.countplot(x="Sentiment", data=df)
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.savefig("charts/sentiment_distribution.png")
plt.show()

plt.figure(figsize=(6,5))
sns.countplot(x="Score", data=df)
plt.title("Rating Distribution")
plt.xlabel("Score")
plt.ylabel("Count")
plt.savefig("charts/rating_distribution.png")
plt.show()
sentiment_counts = df["Sentiment"].value_counts()

plt.figure(figsize=(6,6))
plt.pie(
    sentiment_counts,
    labels=sentiment_counts.index,
    autopct="%1.1f%%",
    startangle=90
)
plt.title("Sentiment Percentage")
plt.savefig("charts/sentiment_pie_chart.png")
plt.show()

df["Review_Length"] = df["Text"].apply(len)

plt.figure(figsize=(8,5))
plt.hist(df["Review_Length"], bins=20)
plt.title("Review Length Distribution")
plt.xlabel("Characters")
plt.ylabel("Frequency")
plt.savefig("charts/review_length.png")
plt.show()
text = " ".join(df["Text"])

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud")
plt.savefig("charts/wordcloud.png")
plt.show()

positive_text = " ".join(df[df["Sentiment"]=="Positive"]["Text"])

positive_cloud = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(positive_text)

plt.figure(figsize=(10,5))
plt.imshow(positive_cloud, interpolation="bilinear")
plt.axis("off")
plt.title("Positive Reviews Word Cloud")
plt.savefig("charts/positive_wordcloud.png")
plt.show()

negative_text = " ".join(df[df["Sentiment"]=="Negative"]["Text"])

negative_cloud = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(negative_text)

plt.figure(figsize=(10,5))
plt.imshow(negative_cloud, interpolation="bilinear")
plt.axis("off")
plt.title("Negative Reviews Word Cloud")
plt.savefig("charts/negative_wordcloud.png")
plt.show()

print("\nProject Completed Successfully!")
print("Charts are saved in the 'charts' folder.")