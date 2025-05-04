import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load the merged dataset
df = pd.read_csv('merged_fake_true.csv')   # Make sure this file exists in the working directory

# Peek at the structure
print("DataFrame Head\n(Optional peek at the structure; Guarantees that the data is being properly read):")
print(df.head())

# Dropx unnecessary columns
df = df.drop(columns=[col for col in ['title', 'subject', 'date'] if col in df.columns], errors='ignore')

# Remove "(Reuters)" tag from true news text
df['text'] = df['text'].str.replace(r'\(Reuters\)', '', regex=True)

# Check label values
print("Unique values in 'label' column:", df['label'].unique())
print("Now preparing Classification Report and Matrix, Please wait...")

# Drop rows with missing labels just in case
df = df.dropna(subset=['label'])

# Ensure labels are integers
df['label'] = df['label'].astype(int)

# --- Exploratory Data Analysis (EDA) Outputs ---
# It may also make the "Please Wait" loading look weird in execution.
print("\nDataset Shape (rows, columns):", df.shape)
print("Dataset Columns:", df.columns.tolist())
print("Class distribution (value counts):")
print(df['label'].value_counts())

# Split features and labels
X = df['text']
y = df['label']

# Split into training/testing sets (Random state 42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Predict on test set
y_pred = model.predict(X_test_tfidf)

# Evaluate
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["FAKE", "TRUE"]))

# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["FAKE", "TRUE"])
disp.plot(cmap='Blues')
# Save Plot
plt.title("Confusion Matrix - Logistic Regression")
plt.savefig("confusion_matrix.png")
plt.close()
print("\nConfusion Matrix saved as a png to current file location!")

# --------- EDA Visuals ---------
# This is a supplemental section of code for purely my use. It has been included for posterity.
# 1. Label Distribution Plot
label_counts = df['label'].value_counts()
plt.figure(figsize=(6, 4))
sns.barplot(x=label_counts.index, y=label_counts.values, palette="Set2")
plt.xticks(ticks=[0, 1], labels=["FAKE", "TRUE"])
plt.title("Label Distribution")
plt.xlabel("Label")
plt.ylabel("Number of Articles")
plt.savefig("label_distribution.png")
plt.close()
print("Saved: label_distribution.png")

# 2. Word Cloud
all_text = ' '.join(df['text'].dropna().astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white',
                      max_words=200).generate(all_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Most Frequent Words in Articles")
plt.savefig("wordcloud.png")
plt.close()
print("Saved: wordcloud.png")

# 3. Article Length Distribution
df['text_length'] = df['text'].apply(lambda x: len(str(x).split()))
plt.figure(figsize=(8, 5))
sns.histplot(df['text_length'], bins=50, kde=True, color="skyblue")
plt.title("Distribution of Article Word Counts")
plt.xlabel("Word Count")
plt.ylabel("Frequency")
plt.savefig("text_length_distribution.png")
plt.close()
print("Saved: text_length_distribution.png")

# Optionally remove text_length column to clean up
df.drop(columns='text_length', inplace=True)
