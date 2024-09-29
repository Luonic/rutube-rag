import os
import telebot
import requests


if __name__ == "__main__":
    token = os.environ["TG_BOT_TOKEN"]
    url = "http://95.165.128.223:31000/get_answer"

    bot=telebot.TeleBot(token)
    
    @bot.message_handler(commands=['start'])
    def start_message(message):
        bot.send_message(message.chat.id,"Привет! Какой у вас вопрос?")
        
    @bot.message_handler(func=lambda m: True)
    def echo_all(message):
        data_json = {"question": message.text}
        resp = requests.post(url, json=data_json)
        resp.raise_for_status()
        answer_json = resp.json()        
        answer = answer_json['answer'] 
        cls_1 = answer_json['class_1']
        cls_2 = answer_json['class_2']
        
        combined_answer = f"{answer}\nКласс1: {cls_1}\nКласс2: {cls_2}"
        
        bot.reply_to(message, combined_answer)
     
    bot.infinity_polling()
