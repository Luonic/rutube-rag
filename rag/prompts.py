RAG_PROMPT = """Используйте следующие части полученного контекста для ответа на поставленный вопрос. 
     Ответ должен быть максимально точным, правильным и релевантным вопросу и контексту. 
     Не совершайте фактологических и логических ошибок относительно контекста и вопроса.
     Если в контексте нет ответа на поставленный вопрос, скажите, что не знаете, либо ведите диалог с пользователем в режиме болталки, если он задает общие вопросы.
     Вопрос: {question}
     Контекст: {context}"""

MULTIQUERY_PROMPT = """Вы помощник языковой модели искусственного интеллекта для ответов на вопросы пользователей видеохостинга Rutube. 
                  Ваша задача — сгенерировать 3 разных, но близких по смыслу версий заданного вопроса для извлечения соответствующих документов из векторной базы данных. 
                  Можно использовать парафразы или синонимы к понятиям из вопроса, но с высокой степенью релевантности к первоначальному вопросу. 
                  Укажите исходный вопрос и 3 альтернативных вопросов в JSON следующего формата: {"queries": ["Исходный вопрос", "Альтернативный вопрос 1", ..., "Альтернативный вопрос 3"]}."""