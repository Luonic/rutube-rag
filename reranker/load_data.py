from datasets import load_dataset

# Загружаем датасет
dataset = load_dataset('unicamp-dl/mmarco', 'russian', split='train[:1000]')
print(dataset[500])

# # Функция для преобразования записей в формат query, answer, label
# def process_example(example):
#     # Создаем две записи: одну для положительного, другую для отрицательного примера
#     return {
#         'query': [example['query'], example['query']],
#         'answer': [example['positive'], example['negative']],
#         'label': [1, 0]
#     }

# # Применяем функцию через map, и flatten для объединения вложенных списков
# tidy_dataset = dataset['train'].map(process_example, batched=True, remove_columns=['positive', 'negative'])

# # Просматриваем результат
# print(tidy_dataset[0])
# print(tidy_dataset[1])

# print(len(dataset['train']))