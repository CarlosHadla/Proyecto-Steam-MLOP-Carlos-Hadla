#cargando librerias
from fastapi import FastAPI
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


app = FastAPI()

games_df = pd.read_csv(r'./gamesF.csv')
items_df = pd.read_csv(r'./itemsF.csv')
reviews_df = pd.read_csv(r'./reviewsF.csv')
genres_df = pd.read_csv(r'./genres.csv')
specs_df = pd.read_csv(r'./specs.csv')

# Endpoint para obtener la información de desarrollador
@app.get("/developer")
def developer(desarrollador: str):
    # Limpiar y estandarizar el nombre del desarrollador especificado
    desarrollador = desarrollador.strip().lower()
    # Verificar si el desarrollador existe en los datos
    if desarrollador not in games_df["developer"].str.strip().str.lower().values:
        return f'El Desarrollador: {desarrollador} no se encuentra en los datos.'
    # Filtra el DataFrame para obtener los juegos del desarrollador especificado y con precios mayores o iguales a 0
    developer_games = games_df[(games_df["developer"].str.strip().str.lower() == desarrollador) & (games_df["price"] >= 0)]
    # Agrupa por año y cuenta la cantidad de juegos y el porcentaje de contenido gratuito
    developer_summary = developer_games.groupby("release_date").agg(
        num_items=("app_name", "count"),  # Cuenta la cantidad de juegos
        free_content_percent=("price", lambda x: (x == 0).mean() * 100)  # Calcula el porcentaje de contenido gratuito
    ).reset_index()
    # Convierte el DataFrame de resumen a un formato JSON
    developer_summary_json = developer_summary.to_dict(orient="records")
    return developer_summary_json

@app.get("/userdata/{user_id}")
def userdata(user_id: str):
    # Filtra los DataFrames para obtener datos del usuario especificado
    user_reviews = reviews_df[reviews_df["user_id"] == user_id]
    if user_reviews.empty:
        return {"Error": f"Usuario con ID {user_id} no encontrado"}
    # Calcula la cantidad de dinero gastado
    dinero_gastado = 0
    for index, row in user_reviews.iterrows():
        game_id = row["item_id"]
        game_info = games_df[games_df["item_id"] == game_id]
        if not game_info.empty:
            precio = game_info.iloc[0]["price"]
            dinero_gastado += precio
    # Cantidad de ítems
    cantidad_items = len(user_reviews)
    # Porcentaje de recomendación
    total_recommended = user_reviews["recommend"].sum()
    total_reviews = user_reviews.shape[0]
    porcentaje_recomendacion = (total_recommended / total_reviews) * 100 if total_reviews > 0 else 0
    # Retorna los datos en un diccionario
    return {
        "Usuario": user_id,
        "Dinero gastado": f"${dinero_gastado:.2f}",
        "% de recomendación": f"{porcentaje_recomendacion:.2f}%",
        "Cantidad de ítems": cantidad_items
    }

@app.get('/UserForGenre')
def UserForGenre(genero: str):
    """
    Devuelve el usuario que ha acumulado más horas jugadas para un género dado,
    junto con una lista de la acumulación de horas jugadas por año.
    """
    if genero not in genres_df.columns:
        return {'message': 'El género solicitado no está presente en los datos'}

    # Filtrar los datos de acuerdo al género solicitado
    genre_data = genres_df[genres_df[genero] == 1]

    if genre_data.empty:
        return {'message': 'No hay juegos en el género especificado'}

    # Realizar un inner join entre genre_data y items_df usando 'item_id' como clave
    merged_df = genre_data.merge(items_df, on='item_id', how='inner')
    
    # Excluir el año 1900
    merged_df = merged_df[merged_df['release_date'] != 1900]

    if merged_df.empty:
        return {'message': 'No hay datos disponibles para el año especificado'}

    # Calcular la suma de las horas jugadas por usuario y año
    user_year_playtime = merged_df.groupby(['user_id', 'release_date'])['playtime_forever'].sum().reset_index()

    if user_year_playtime.empty:
        return {'message': 'No hay datos disponibles para el género especificado'}

    # Encontrar el usuario con más horas jugadas
    user_with_most_playtime = user_year_playtime.groupby('user_id')['playtime_forever'].sum().idxmax()

    # Filtrar los datos del usuario con más horas jugadas
    user_most_playtime_data = user_year_playtime[user_year_playtime['user_id'] == user_with_most_playtime]

    # Crear una lista de acumulación de horas jugadas por año
    hours_played_by_year = [{'Año': year, 'Horas': playtime} for year, playtime in zip(user_most_playtime_data['release_date'], user_most_playtime_data['playtime_forever'])]

    # Devolver el resultado como un diccionario
    result = {
        f'Usuario con más horas jugadas para Género {genero}': user_with_most_playtime,
        'Horas jugadas': hours_played_by_year
    }
    return result


@app.get('/best_developer_year')
def best_developer_year(anio: int):
    if anio not in reviews_df['year'].values or anio == 1900:
        return "No hay reviews para el año especificado"
    # Filtrar los juegos por el año especificado
    games_by_year = games_df[games_df['release_date'] == anio]
    # Filtrar las reseñas por el año especificado
    reviews_by_year = reviews_df[reviews_df['year'] == anio]
    # Filtrar las reseñas con recommend igual a True
    recommended_reviews = reviews_by_year[reviews_by_year['recommend']]
    # Vincular las reseñas con los juegos filtrados
    merged_reviews = recommended_reviews.merge(games_by_year, on='item_id', how='inner')
    # Agrupar las reseñas por desarrollador
    developer_reviews = merged_reviews.groupby('developer')
    # Contar la cantidad de reseñas recomendadas por desarrollador
    developer_review_counts = developer_reviews.size().reset_index(name='count')
    # Clasificar los desarrolladores por la cantidad de reseñas recomendadas en orden descendente
    top_developers = developer_review_counts.sort_values(by='count', ascending=False)
    # Seleccionar los 3 primeros desarrolladores
    top3_developers = top_developers.head(3)
    # Formatear el resultado en la estructura requerida
    result = [{"Puesto 1": top3_developers.iloc[0]['developer']},
              {"Puesto 2": top3_developers.iloc[1]['developer']},
              {"Puesto 3": top3_developers.iloc[2]['developer']}]
    return result

@app.get("/developer_reviews_analysis")
def developer_reviews_analysis(desarrolladora: str):
    if desarrolladora not in games_df['developer'].values:
        return f"No esta la desarroladora {desarrolladora} en los datos"
    # Realizar un inner join entre reviews_df y games_df utilizando 'item_id' como clave
    merged_reviews = reviews_df.merge(games_df, on='item_id', how='inner')
    # Filtrar las reseñas por la desarrolladora especificada
    developer_reviews = merged_reviews[merged_reviews['developer'] == desarrolladora]
    # Contar la cantidad de reseñas con análisis de sentimiento positivo (sentiment_analysis = 2)
    positive_reviews_count = len(developer_reviews[developer_reviews['sentiment_analysis'] == 2])
    # Contar la cantidad de reseñas con análisis de sentimiento negativo (sentiment_analysis = 0)
    negative_reviews_count = len(developer_reviews[developer_reviews['sentiment_analysis'] == 0])
    # Preparar el resultado en un diccionario
    result = {
        desarrolladora: {"Positive": positive_reviews_count, "Negative": negative_reviews_count}
    }
    return result

@app.get("/recomendacion_juego/{id_producto}")
async def recomendacion_juego(id_producto:int):
    # Verificar si el ID existe en el DataFrame
    if id_producto not in games_df['item_id'].values:
        return "El item_id solicitado no pertenece a ningún juego."
    #armar el df con generos y specs
    df_gs = genres_df.merge(specs_df, on='item_id', how='inner')
    #armar el df completo
    df = games_df.merge(df_gs, on='item_id', how='inner')
    df.fillna(0, inplace=True)
    # Obtener el vector de géneros del juego de entrada
    juego_vector = df[df['item_id'] == id_producto].iloc[:, 4:].values.reshape(1, -1)
    # Calcular la similitud del coseno entre el juego de entrada y todos los demás juegos
    similarity_scores = cosine_similarity(df.iloc[:, 4:], juego_vector)
    # Obtener los índices de los juegos más similares
    similar_indices = similarity_scores.argsort(axis=0)[::-1][:5]
    # Obtener los nombres de los juegos recomendados
    recomendaciones = df.iloc[similar_indices.ravel(), :]['app_name'].values.tolist()
    return recomendaciones

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)