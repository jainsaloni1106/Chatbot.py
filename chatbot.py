#import the required libraries
import io
import random
import string 
import warnings
import numpy as np
from termcolor import*
from colorama  import*
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True) 



#Reading the file as chatbot.txt data taken from kaggle...
with open('chatbot.txt','r', encoding='utf8', errors ='ignore') as file_input:
    file_open = file_input.read().lower()

#Tokenization
sent_the_tokens = nltk.sent_tokenize(file_open)# It converts to list of sentences 
word_the_tokens = nltk.word_tokenize(file_open)# It converts to list of words

# Lemmatization pre-processing , tokens, text and punctuations...
lemmatizer_word = WordNetLemmatizer()
def Lemmatize_Tokens(tokens):
    return [lemmatizer_word.lemmatize(t) for t in tokens]
delete_punctuation_dictionary = dict((ord(punctute), None) for punctute in string.punctuation)
def Lemmatize_Normalize(text):
    return Lemmatize_Tokens(nltk.word_tokenize(text.lower().translate(delete_punctuation_dictionary)))


# Keyword Matching input greeting
user_input_greeting = ("hello", "hi","Hey", "hy there", "what's up","hey","wow")
bot_reply_greeting = ["hi", "hey", "hi there", "hello", "Hy, I am glad! You are talking to me","yes dear"]

def greetings_pass(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in user_input_greeting:
            return random.choice(bot_reply_greeting)


# Generating response by the bot 
def response_function(user_response):
    bot_response=''
    sent_the_tokens.append(user_response)
    Tfidf_vector = TfidfVectorizer(tokenizer=Lemmatize_Normalize, stop_words='english')
    tf = Tfidf_vector.fit_transform(sent_the_tokens)
    values = cosine_similarity(tf[-1], tf)
    index=values.argsort()[0][-2]
    flat_values = values.flatten()
    flat_values.sort()
    request_tf = flat_values[-2]
    if(request_tf==0):
        bot_response=bot_response+"I am sorry! I can't understand , what are you asking.."
        return bot_response
    else:
        bot_response = bot_response+sent_the_tokens[index]
        return bot_response


flag=True
print("\n\n")
cprint("                                ***** Hello, WELCOME TO CHATBOT *****              \n",'red')
cprint("BOT :  I am Bot, You can talk to me about anything; If you want to exit, then type Bye, see you soon !!\n",'yellow')
while(flag==True):
    user_response = input(Fore.RED  + "YOU : ")
    user_response=user_response.lower()
    if(user_response!='bye'):
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            cprint("Bot: You are welcome..",'yellow')
        else:
            if(greetings_pass(user_response)!=None):
                print("BOT: "+greetings_pass(user_response))
            else:
                print("BOT: ",end="")
                print(response_function(user_response))
                sent_the_tokens.remove(user_response)
    else:
        flag=False
        cprint("BOT: Bye! take care..",'green') 
