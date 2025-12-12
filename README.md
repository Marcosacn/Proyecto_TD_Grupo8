# Proyecto_TD_Grupo8
Alejandro Fernández-Vegue García-Caro (100472719), Marcos Amigo Cantón (100451629), Marco Puche Insua (100429690), Santiago Montes Jiménez (100571840)

# Análisis de la polarización ideológica en redes sociales a partir de contenidos desinformativos

Tratamiento de Datos  
Máster en Ingeniería de Telecomunicación  

Realizado por:  
Alejandro Fernández-Vegue García-Caro  
Santiago Montes Jiménez  
Marcos Amigo Cantón  
Marco Puche Insua  

## Detección de desinformación mediante técnicas de NLP y Machine Learning

## 1. Descripción del problema

El proyecto aborda un problema de clasificación binaria supervisada. Dado un texto correspondiente a una noticia o publicación, el objetivo es determinar si se trata de una noticia real (label = 0) o de una fake news (label = 1).

Además del objetivo predictivo, el trabajo busca analizar si existen diferencias lingüísticas y estilísticas entre ambos tipos de contenido, especialmente en lo relativo a carga emocional, polarización del lenguaje y uso de recursos expresivos.

## 2. Conjunto de datos

El conjunto de datos ha sido procesado a partir de una base original de gran tamaño que contenía información no relevante para el proyecto. Tras un proceso de limpieza y filtrado, se seleccionó un subconjunto de aproximadamente 10.000 ejemplos, priorizando los textos con mayor longitud para disponer de más información textual.

La base de datos final utilizada es BASEDATOS_REDUCIDA.xlsx. En el código se emplean, entre otras, las siguientes variables:

- Texto y metadatos principales:
    - title
    - text
    - label
- Longitudes del contenido:
    - title_len
    - text_len
    - total_len
- Variables numéricas relacionadas con el estilo del texto:
    - punctuation_count
    - uppercase_ratio
    - numerical_count
    - sentiment_polarity
- Texto preprocesado generado durante el código:
    - clean_text

El preprocesado se realiza en inglés, utilizando stopwords y lematización, por lo que el análisis está orientado a textos en este idioma.

## 3. Metodologías utilizadas

### 3.1 Análisis exploratorio e hipótesis iniciales

En primer lugar se realiza un análisis exploratorio del conjunto de datos. Este análisis incluye la distribución de clases, el estudio de las longitudes de los textos, el análisis de palabras frecuentes y ejemplos representativos por clase, así como el cálculo de medias de distintos rasgos estilísticos.

A partir de este análisis se plantean las siguientes hipótesis:

1. Las fake news presentan una mayor carga emocional, reflejada en valores más extremos de polaridad.
2. Las noticias falsas hacen un uso más intensivo de signos de puntuación y mayúsculas como recurso enfático.
3. Las fake news tienden a mostrar longitudes de texto extremas, siendo muy cortas o muy largas.
4. La presencia de números es mayor en fake news como posible estrategia de credibilidad.

### 3.2 Representación vectorial del texto

Antes de la vectorización se construye una versión limpia del texto aplicando conversión a minúsculas, tokenización, eliminación de stopwords, lematización y filtrado de tokens no alfanuméricos.

Se emplean tres estrategias principales de representación:

**TF-IDF**  
El dataset se divide de forma estratificada en conjuntos de entrenamiento, validación y test. El texto se vectoriza utilizando TF-IDF con unigramas y bigramas, limitando el tamaño del vocabulario y filtrando términos poco frecuentes. A esta representación se le añaden variables numéricas relacionadas con el estilo del texto.

**Word2Vec**  
Se entrena un modelo Word2Vec sobre los textos del conjunto de entrenamiento. Cada documento se representa mediante el promedio de los embeddings de sus palabras, al que se concatenan las mismas variables numéricas adicionales.

**Embeddings contextuales (DistilBERT)**  
Se utilizan embeddings contextuales obtenidos con DistilBERT. La representación de cada documento se obtiene a partir del vector CLS y se combina con las variables numéricas.

### 3.3 Modelado y evaluación

Se entrenan tres tipos de modelos, tal y como exige el enunciado del proyecto.

Por un lado, se emplean redes neuronales implementadas en PyTorch, utilizando una arquitectura MLP común para las tres representaciones. Además, se entrenan modelos tradicionales de scikit-learn como Logistic Regression, Random Forest, SVM y KNN sobre las distintas representaciones vectoriales.

Finalmente, se realiza un ajuste fino de un modelo DistilBERT para clasificación binaria utilizando la librería Hugging Face Transformers.

## 4. Resultados experimentales y discusión

Los modelos se evalúan sobre el conjunto de test utilizando métricas estándar como:
- Accuracy (TP + TN) / (TP + TN + FP + FN)
- Precision (TP) / (TP + FP)
- Recall (TP) / (TP + FN)
- F1-score (2 · (Precision · Recall) / (Precision + Recall))
- ROC-AUC 

Los resultados obtenidos muestran que las representaciones basadas en TF-IDF siguen siendo extremadamente eficaces para la detección de desinformación. En particular, el modelo Red Neuronal + TF-IDF alcanza el mejor rendimiento global en el conjunto de test, con un F1-score de 0.9391, una accuracy de 0.9305 y un ROC-AUC de 0.9793, lo que lo convierte en la mejor solución evaluada. Los modelos tradicionales, especialmente Logistic Regression + TF-IDF, también presentan resultados muy competitivos, actuando como un baseline sólido, eficiente e interpretable, con un rendimiento cercano al de la red neuronal. Esto pone de manifiesto que gran parte de la información discriminativa está contenida en la frecuencia y distribución léxica del texto. Por otro lado, las representaciones basadas en Word2Vec y en embeddings contextuales de DistilBERT funcionan adecuadamente en combinación con modelos más complejos, como redes neuronales o Random Forest. Sin embargo, los modelos más simples, como KNN y SVM, muestran un rendimiento claramente inferior cuando se combinan con Word2Vec o DistilBERT, obteniendo los peores resultados globales del estudio. Esto sugiere que estos clasificadores no son capaces de explotar eficazmente la información contenida en representaciones densas y de alta dimensión sin un proceso adicional de adaptación o fine-tuning. Finalmente, el fine-tuning de DistilBERT logra resultados elevados y estables, aunque la mejora frente a los modelos clásicos basados en TF-IDF es moderada, lo que resulta razonable teniendo en cuenta el tamaño del dataset y el mayor coste computacional asociado a este tipo de modelos.

## 5. Conclusiones

En este proyecto se ha abordado el análisis de la desinformación desde una perspectiva integral, combinando análisis exploratorio, representación vectorial del texto y modelado mediante técnicas de Machine Learning y Deep Learning. Los resultados obtenidos demuestran que la detección automática de contenido desinformativo es viable y eficaz cuando se utilizan representaciones adecuadas del lenguaje y modelos correctamente ajustados, destacando por encima del resto el uso de Red Neuronal + TF-IDF. En cuanto a las hipótesis iniciales planteadas relacionadas con estilo y carga emocional, estas resultan coherentes con los resultados observados, demostrando una mayor carga emocional en contenido desinformativo, un mayor uso de signos de puntuación y letras en mayúsculas en textos falsos y un exceso de número de palabras y de datos numéricos en estos textos falsos.

## 6. Proyecto de extensión

Como ampliación del proyecto, se incluye un bloque adicional de análisis orientado a profundizar en la relación entre desinformación y polarización desde un punto de vista exploratorio.

### 6.1 Análisis temático mediante clustering

Se aplica clustering no supervisado (K-Means) sobre los embeddings contextuales obtenidos con DistilBERT para identificar agrupaciones temáticas de noticias. Para cada grupo se analiza la proporción de fake news y la media de distintos rasgos lingüísticos.

Este análisis permite observar que ciertos temas, especialmente relacionados con política o asuntos sociales, presentan una mayor concentración de contenido desinformativo.

### 6.2 Análisis de incertidumbre del modelo

Se analiza la confianza de las predicciones del modelo basado en DistilBERT, definida a partir de la cercanía de la probabilidad al umbral de decisión. Los textos con menor confianza muestran un lenguaje más emocional y una mayor presencia de fake news, lo que apunta a la existencia de casos ambiguos difíciles de clasificar.

### 6.3 Relación entre desinformación y polarización lingüística

Por último, se estudia la relación entre la etiqueta real/fake y distintos indicadores lingüísticos asociados a polarización, como el sentimiento, el uso de mayúsculas, los signos de puntuación y la presencia de números. Los resultados refuerzan la relación entre polarización lingüística y desinformación.




