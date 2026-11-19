from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)


try:
    df = pd.read_csv('final_games_data.csv')
    tfidf_matrix = pickle.load(open('tfidf_matrix.pkl', 'rb'))
    
    
    df.columns = df.columns.str.strip()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('').astype(str)
            
    if 'Price' in df.columns:
        df['Price_Num'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0)
    
    print("✅ System Ready: Eng. Afnan Version")
except Exception as e:
    print(f"❌ Error: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    game_name = request.form.get('game_name', '').strip().lower()
    try:
        match = df[df['Name'].str.lower().str.contains(game_name, na=False)]
        if match.empty:
            return render_template('index.html', error=f"Game '{game_name}' not found.")
        
        idx = match.index[0]
        cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
        sim_scores = sorted(list(enumerate(cosine_sim)), key=lambda x: x[1], reverse=True)[1:7]
        
        results = []
        for i in sim_scores:
            row = df.iloc[i[0]]
            results.append({
                'name': row['Name'],
                'genre': row.get('Genres', 'Action'),
                'price': row.get('Price', '0.0'),
                'website': row.get('Website', '#'),
                'pos': row.get('Positive', '0'),
                'neg': row.get('Negative', '0')
            })
        return render_template('index.html', results=results, search_name=game_name)
    except:
        return render_template('index.html', error="Similarity search failed.")

@app.route('/predict_preference', methods=['POST'])
def predict_preference():
    try:
        user_input = request.form.get('genre_input', '').strip().lower()
        max_price = float(request.form.get('price', 100))
        
        temp_df = df[df['Price_Num'] <= max_price].copy()

        if user_input:
            mask = temp_df.apply(lambda row: row.str.lower().str.contains(user_input).any(), axis=1)
            temp_df = temp_df[mask]

        top_games = temp_df.head(9)

        if top_games.empty:
            return render_template('index.html', error=f"No results for '{user_input}'.")

        pref_results = []
        for _, row in top_games.iterrows():
            pref_results.append({
                'name': row['Name'],
                'genre': row.get('Genres', 'Games'),
                'price': row.get('Price', '0.0'),
                'website': row.get('Website', '#'),
                'pos': row.get('Positive', 'N/A'),
                'neg': row.get('Negative', 'N/A')
            })
        return render_template('index.html', pref_results=pref_results, pref_mode=True)
    except:
        return render_template('index.html', error="An error occurred during search.")

if __name__ == '__main__':
    app.run(debug=True)
