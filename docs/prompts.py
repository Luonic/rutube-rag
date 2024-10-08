DOCX_QUESTIONS_PROMPT = """
Перед тобой фрагмент 'Генеральное пользовательское соглашение RUTUBE' на видехостинге RUTUBE. Извлеки из него набор качественных вопросов и кратких ответов. Вопросы должны быть информативными и фокусироваться на ключевых фактах текста. Ответы должны быть чёткими, конкретными и взяты напрямую из текста. Убедись, что каждый вопрос относится только к информации, содержащейся в тексте, и избегай предположений.

Если в тексте упоминаются конкретные пункты соглашений, политик, договоров или оферт (например, "п. 4.17", "п. 9.3.2"), эти упоминания должны быть включены только в ответе, но не в вопросе.
Упоминай наименование документа 'Генеральное пользовательское соглашение RUTUBE' или его синонимы, когда это необходимо при упоминании пункта.

Количество вопросов-ответов должно варьироваться от 2 до 12 в зависимости от длины ответа:
- Чем длиннее ответы, тем больше должно быть вопросов (до максимума в 12).
- Чем короче ответы, тем меньше должно быть вопросов (до минимума в 2).

Результат верни в формате JSON, где каждая пара вопрос-ответ находится в следующей структуре:

{
  "questions": [
    {
      "question": "Текст вопроса",
      "answer": "Текст ответа"
    }
  ]
}

Требования к JSON:
1. Структура включает массив вопросов и ответов.
2. Вопросы точные, без лишней информации.
3. Ответы краткие и содержат только факты из текста.
4. Упоминания пунктов соглашений или договоров включаются только в ответах.
5. Минимизируй дублирование информации в вопросах.
6. Количество вопросов-ответов должно быть от 2 до 12, исходя из длины ответа.
7. Формулировки вопросов могут требовать интерпретации ключевых идей текста.

Пример:

Текст: "Категория вашего видео была изменена для соответствия содержанию ролика. Согласно п. 4.17. Генерального пользовательского соглашения (https://rutube.ru/info/agreement/), мы можем изменить категорию, если считаем, что ролик стоит отнести к другой тематике. После публикации видео вы не сможете изменить его категорию самостоятельно."

JSON:

{
  "questions": [
    {
      "question": "Почему категория видео может быть изменена на RUTUBE?",
      "answer": "Категория может быть изменена для соответствия содержанию ролика согласно п. 4.17. Генерального пользовательского соглашения."
    },
    {
      "question": "Могу ли я изменить категорию видео после его публикации?",
      "answer": "Нет, вы не сможете изменить категорию после публикации видео."
    }
  ]
}

Пример:

Текст: "При выявлении такой ситуации администрация RUTUBE уведомит вас о необходимости самостоятельно удалить дублирующее видео, и до его удаления монетизация канала/конкретного видео приостанавливается. Если нарушение не устраняется, Компания вправе расторгнуть Договор в одностороннем порядке согласно п. 9.3.2 Оферты/Договора (https://rutube.ru/info/adv_oferta)."

JSON:

{
  "questions": [
    {
      "question": "Что произойдёт, если RUTUBE обнаружит дублирующее видео?",
      "answer": "Администрация уведомит о необходимости удалить дублирующее видео, и до удаления монетизация приостанавливается. Если нарушение не устраняется, компания вправе расторгнуть Договор согласно п. 9.3.2 Оферты/Договора."
    }
  ]
}

"""

CLASSIFIER_PROMPT = """
Классифицируй приведённые ниже вопросы и ответы по следующим заранее предопределённым категориям:

    БЛАГОТВОРИТЕЛЬНОСТЬ ДОНАТЫ | Подключение/отключение донатов
    ВИДЕО | Воспроизведение видео
    ВИДЕО | Встраивание видео
    ВИДЕО | Загрузка видео
    ВИДЕО | Комментарии
    ВИДЕО | Недоступность видео
    ВИДЕО | Перенос видео с Youtube
    ВИДЕО | Система рекомендаций
    ВИДЕО | Управление плеером
    ДОСТУП К RUTUBE | Приложение
    МОДЕРАЦИЯ | Блокировка канала
    МОДЕРАЦИЯ | Долгая модерация
    МОДЕРАЦИЯ | Запрещенный контент
    МОДЕРАЦИЯ | Нарушение авторских прав
    МОДЕРАЦИЯ | Отклонение/блокировка видео
    МОДЕРАЦИЯ | Смена категории/возрастные ограничения
    МОНЕТИЗАЦИЯ | Отключение/подключение монетизации
    МОНЕТИЗАЦИЯ | Подключение/отключение рекламы
    МОНЕТИЗАЦИЯ | Статистика по монетизации
    ОТСУТСТВУЕТ | Отсутствует
    ПОИСК | История поиска
    ПОИСК | Текстовый поиск
    ПРЕДЛОЖЕНИЯ | Монетизация
    ПРЕДЛОЖЕНИЯ | Навигация
    ПРЕДЛОЖЕНИЯ | Персонализация 0
    ПРЕДЛОЖЕНИЯ | Плеер
    ПРЕДЛОЖЕНИЯ | Студия RUTUBE
    ПРЕДЛОЖЕНИЯ | Трансляция
    СОТРУДНИЧЕСТВО ПРОДВИЖЕНИЕ РЕКЛАМА | Продвижение канала
    ТРАНСЛЯЦИЯ | Просмотр трансляции
    ТРАНСЛЯЦИЯ | ТВ-эфиры
    ТРАНСЛЯЦИЯ | Управление трансляцией
    ТРАНСЛЯЦИЯ | Чат/Комментарии
    УПРАВЛЕНИЕ АККАУНТОМ | Аналитика
    УПРАВЛЕНИЕ АККАУНТОМ | Верификация
    УПРАВЛЕНИЕ АККАУНТОМ | Персонализация
    УПРАВЛЕНИЕ АККАУНТОМ | Платный контент
    УПРАВЛЕНИЕ АККАУНТОМ | Регистрация/Авторизация
    УПРАВЛЕНИЕ АККАУНТОМ | Удаление аккаунта

Требования к классификации:
1. Тщательно анализируй каждый вопрос-ответ перед назначением категории. Убедись, что выбранная категория точно соответствует основному смыслу пары.
2. Если вопрос-ответ касается загрузки, воспроизведения, рекомендаций или других аспектов видео, отнеси его к соответствующим подкатегориям раздела "ВИДЕО".
3. Если вопрос-ответ касается монетизации, статистики доходов, или рекламы, отнеси его к соответствующим категориям раздела "МОНЕТИЗАЦИЯ".
4. Вопросы, касающиеся модерации контента, блокировки каналов, авторских прав, или запрещённого контента, должны классифицироваться в категории раздела "МОДЕРАЦИЯ".
5. Вопросы, касающиеся управления аккаунтом, регистрации, персонализации, или платного контента, следует отнести к категориям раздела "УПРАВЛЕНИЕ АККАУНТОМ".
6. Если вопрос не относится ни к одной из категорий, используй категорию "ОТСУТСТВУЕТ".

Пример:

Вход:

{
  "questions": [
    {
      "question": "Как я могу включить монетизацию на своём канале?",
      "answer": "Чтобы включить монетизацию, необходимо подключить аккаунт к партнёрской программе.",
    },
    {
      "question": "Почему мой комментарий к видео был удалён?",
      "answer": "Ваш комментарий мог быть удалён за нарушение правил платформы."
    }
  ]
}

Выход JSON:

{
  "questions": [
    {
      "question": "Как я могу включить монетизацию на своём канале?",
      "answer": "Чтобы включить монетизацию, необходимо подключить аккаунт к партнёрской программе.",
      "clf": "МОНЕТИЗАЦИЯ | Отключение/подключение монетизации"
    },
    {
      "question": "Почему мой комментарий к видео был удалён?",
      "answer": "Ваш комментарий мог быть удалён за нарушение правил платформы.",
      "clf": "ВИДЕО | Комментарии"
    }
  ]
}

Требования:
1. Каждый вопрос и ответ должен быть отнесён к одной из категорий.
2. Анализируй контекст и содержание перед присвоением категории.
3. Классифицируй пары точно и строго в соответствии с темой вопроса-ответа.
4. Точно указывай классифицируемые категории вместе с основной категорией и подкатегорией 'КАТЕГОРИЯ / Подкатегория'
"""