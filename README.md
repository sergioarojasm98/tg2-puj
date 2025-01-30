# Estegoanálisis de Imágenes basado en Técnicas de Inteligencia Artificial

## Descripción

Este repositorio contiene la implementación del proyecto de grado para la Maestría en Inteligencia Artificial, el cual aborda la detección de esteganografía en imágenes RGB mediante técnicas avanzadas de Inteligencia Artificial. El proyecto evalúa el desempeño de tres arquitecturas distintas de Redes Neuronales Convolucionales (CNN) y una arquitectura de Vision Transformer (ViT) en la tarea de estegoanálisis.

## Estructura del Repositorio

# Estegoanálisis de Imágenes basado en Técnicas de Inteligencia Artificial

## Descripción

Este repositorio contiene la implementación del proyecto de grado para la Maestría en Inteligencia Artificial, el cual aborda la detección de esteganografía en imágenes RGB mediante técnicas avanzadas de Inteligencia Artificial. El proyecto evalúa el desempeño de tres arquitecturas distintas de Redes Neuronales Convolucionales (CNN) y una arquitectura de Vision Transformer (ViT) en la tarea de estegoanálisis.

## Estructura del Repositorio

```
|-- README.md
|-- cnn.py  # Implementación de las arquitecturas de CNN
|-- vit.py  # Implementación de la arquitectura Vision Transformer
```

## Métodos de Esteganografía Evaluados

El proyecto abarca la detección de esteganografía en imágenes utilizando los siguientes métodos:

- **LSB (Least Significant Bit)**: Modificación de los bits menos significativos en la imagen.
- **DCT (Discrete Cosine Transform)**: Modificación de coeficientes de frecuencia en la transformada de coseno discreta.
- **DWT (Discrete Wavelet Transform)**: Ocultamiento de información en los coeficientes de transformada wavelet discreta.
- **ViT (Visual Transformers)**: 

## Resultados

El rendimiento de las arquitecturas se evaluó mediante diversas métricas, incluyendo:

- Precisión (Accuracy)
- Área Bajo la Curva (AUC-ROC)
- Coeficiente de Correlación de Matthews (MCC)

