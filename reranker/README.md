# Кросс-энкодер для переранжирования документов

В данной директории находится все необходимое для обучения модели, 
которая получает на вход вопрос и текст потенциально релевантного документа, 
и предсказывает вероятность того, насколько текст документа релевантен заданному вопросу.


## Данные

В `dataset.py` содержатся функции для подготовки данных как для претрейна на mMARCO, 
так и для SFT этапа обучения модели на сгенерированных вопросах разного уровня релевантности.

## Обучение

Конфигурация обучения производится из файла `train_config.yaml`, для запуска обучения достаточно выполнить `python3 train.py`. В результате обучения получатся pythorch-чекпоинты, которые нужно подгружать в инстанциированную HF-модель.

## Использование

Весь пайплайн ранжирования чанков данных выполняется классом `handler.py`