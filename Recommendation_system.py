import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import messagebox

# 1. Load data
df = pd.read_csv('compressed_data.csv.gz', low_memory=False)

# 2. Check and remove null values
df.dropna(subset=['title', 'genres'], inplace=True)

# Ensure 'vote_average' column exists and fill missing ratings with 'N/A'
if 'vote_average' not in df.columns:
    df['vote_average'] = 'N/A'
else:
    df['vote_average'] = df['vote_average'].apply(lambda x: x if pd.notnull(x) else 'N/A')

# 3. Clean genres
def convert_genres(text):
    try:
        genres = ast.literal_eval(text)
        return " ".join([d['name'] for d in genres if 'name' in d])
    except:
        return ""

df['clean_genres'] = df['genres'].apply(convert_genres)
df = df[df['clean_genres'] != ""].reset_index(drop=True)

# 4. Vectorize genres
tfidf = TfidfVectorizer()
genre_matrix = tfidf.fit_transform(df['clean_genres'])

# 5. Compute similarity using sparse output to avoid memory crash
similarity = cosine_similarity(genre_matrix, dense_output=False)

# 6. Define recommendation function for GUI
def recommend_movie_gui():
    movie_title = entry.get().strip()
    listbox.delete(0, tk.END)
    if not movie_title:
        messagebox.showinfo("Input Error", "Please enter a movie title.")
        return
    matches = df[df['title'].str.lower() == movie_title.lower()]
    if matches.empty:
        messagebox.showinfo("Not Found", "Movie not found. Please check the spelling.")
        return
    idx = matches.index[0]
    sim_scores = similarity[idx].toarray().flatten()
    sim_scores = list(enumerate(sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    count = 0
    for i in sim_scores[1:]:  # Skip the first (itself)
        title = df.iloc[i[0]]['title']
        rating = df.iloc[i[0]]['vote_average'] if 'vote_average' in df.columns else 'N/A'
        if pd.isnull(rating):
            rating = 'N/A'
        if title.lower() != movie_title.lower():
            listbox.insert(tk.END, f"{title}  (Rating: {rating})")
            count += 1
        if count == 5:
            break
    if count == 0:
        listbox.insert(tk.END, "No similar movies found.")

# 7. Tkinter GUI setup
root = tk.Tk()
root.title("Movie Recommendation System")
root.geometry("500x400")

label = tk.Label(root, text="Enter a movie you've watched:", font=("Arial", 12))
label.pack(pady=10)

entry = tk.Entry(root, width=50, font=("Arial", 12))
entry.pack(pady=5)

button = tk.Button(root, text="Get Recommendations", command=recommend_movie_gui, font=("Arial", 12))
button.pack(pady=10)

listbox = tk.Listbox(root, width=60, height=10, font=("Arial", 11))
listbox.pack(pady=10)

def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()