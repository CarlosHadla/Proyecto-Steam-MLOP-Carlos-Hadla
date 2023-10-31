# Sistema de Recomendación de Videojuegos para Steam

Este README proporciona una descripción detallada de un proyecto de Machine Learning Operations (MLOps) para crear un sistema de recomendación de videojuegos en la plataforma Steam. El proyecto se realizó asumiendo el rol de un MLOps Engineer.

## Descripción del Proyecto

En este proyecto, se abordó la tarea de desarrollar un sistema de recomendación de videojuegos para usuarios de Steam, una plataforma multinacional de videojuegos. El objetivo principal fue crear un MVP (Producto Mínimo Viable) que proporcionara recomendaciones de juegos a los usuarios en función de su historial de juego y reseñas.

### Contexto
Steam proporcionó datos de juego, reseñas de usuarios y otros detalles, pero los datos iniciales presentaban desafíos significativos. La calidad de los datos era baja, con datos en bruto y falta de automatización en la actualización de nuevos productos.

### Rol a Desarrollar
El rol asumido fue el de un Data Scientist convertido en MLOps Engineer. Se abordaron diversas tareas, desde la limpieza y transformación de datos hasta la implementación de una API RESTful y el desarrollo de un modelo de recomendación.

# Acciones realizadas
### Transformaciones y Feature Engineering
- Se limpiaron los datos y se realizaron transformaciones necesarias.
- Se aplicó análisis de sentimiento con NLP para crear una columna 'sentiment_analysis' en el conjunto de datos de reseñas de usuarios.

### Desarrollo de la API
- Se implementó una API RESTful utilizando el framework FastAPI.
- Se crearon varios endpoints para consultar datos, incluyendo developer, userdata, UserForGenre, best_developer_year y developer_reviews_analysis.

### Deployment
- Se aseguró que la API fuera accesible en línea, utilizando Render

### Análisis Exploratorio de Datos (EDA)
- Se realizó un análisis exploratorio de datos para comprender mejor las relaciones y patrones en los datos.

### Modelo de Aprendizaje Automático

- Se implementó un sistema de recomendación ítem-ítem y se expuso como un endpoint en la API.

## Autor
Carlos Hadla
## Enlaces Importantes

- [Segundo repositorio con los jupyter notebook y funciones](https://github.com/CarlosHadla/ExtraMLOP-STEAM)
- [Video del proyecto](https://drive.google.com/drive/u/1/folders/1o4x9YurGVyBrH2XKTDl0ijj0n6yxuu4z)
- [Api en Render](https://steam-mlop-carlos-hadla.onrender.com/docs)
