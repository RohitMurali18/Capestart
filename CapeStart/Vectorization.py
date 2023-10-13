import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
import torch

df = pd.read_csv(r"C:\Users\Rohit\Downloads\articles.csv")
df = df.astype(str)
model_name = "paraphrase-MiniLM-L6-v2"  # You can choose a different SentenceBERT model
model = SentenceTransformer(model_name)


columns_to_encode = ['Heading', 'Article.Banner.Image', 'Outlets', 'Article.Description', 'Full_Article', 'Tonality']


embeddings_df = pd.DataFrame()

label_encoder = LabelEncoder()
df['Article_Type_encoded'] = label_encoder.fit_transform(df['Article_Type'])


for column in columns_to_encode:
    text_data = df[column]
    embeddings = model.encode(text_data, convert_to_tensor=True)
    
   
    embeddings_df[column + '_embeddings'] = embeddings.tolist()


print(embeddings_df['Heading_embeddings'][0])