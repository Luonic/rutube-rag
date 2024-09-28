SIMILAR_QUESTIONS_PROMPT = """Тебе надо сгенерировать 10 потенциальных вопросов пользователей к документу, который будет представлен ниже.
    Вопросы должны различаться друг от друга, как будто их писали абсолютно разные люди, разного достатка и интеллекта, пола, расы и возраста.
    Запросы должны отличаться длиной, формальностью, стилем написания, грамотностью. Но они должны быть полностью релевантны первоначальному вопросу из базы знаний и ответу, выдумывать какие-то факты запрещено.
    Поровну для всех вопросов используй релевантные синонимы к понятиям из вопроса, парафразы, в каких-то вопросах можешь писать с ошибками.
    Четко следуй инструкциям, даю 200$ за успешное выполнение задания.
    Ответ должен быть только JSON, следующего формата:

    Пример:
    Вопрос:
    Как выйти из аккаунта в приложении Студия RUTUBE?
    Ответ:
    Для выхода из аккаунта кликните на фото своего профиля в правом верхнем углу и нажмите на кнопку "Выйти" внизу экрана.
    JSON:
    {
        "user_queries": [
            {
            "query": "Здравствуйте!Как выйти из чужого аккаунта стдии на анроиде, чтобы зайти под моим аккаунтом?С уважением,Дмитрий"
            },
            {
            "query": "Как выйти из акаунта в приложении студии?"
            },
            ...
            {
            "query": "Как можно выйти со всех устройств в пиложении студии?"
            },
            {
            "query": "Как выйти из аккаунта в приложении Студия RUTUBE?"
            },
            {
            "query": "напишите пошагово действия как выйти из канала в приложении Студия RUTUBE"
            }
        ]
    }

"""