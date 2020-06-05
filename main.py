from flask import Flask,redirect,url_for,render_template,request
from newspaper import Article
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# getting the corpus that'll be used to train the bot
article = Article('https://ngcareers.com/course/171/accounting')
article.download()
article.parse()
article.nlp()
corpus1 = article.text

article = Article('https://ngcareers.com/course/171/physics')
article.download()
article.parse()
article.nlp()
corpus2 = article.text

article = Article('https://universitycompass.com/courses/computer-science.php#:~:text=With%20the%20exponential%20increase%20in,courses%20offered%20by%20Nigerian%20Universities.&text=It%20is%20a%20Four%20year,Network%20and%20System%20analysts%20etc.')
article.download()
article.parse()
article.nlp()
corpus3 = article.text

corpus = corpus1+corpus2+corpus3
#tokenizing the corpus
text = corpus
sent_tokens = nltk.sent_tokenize(text)

#removing punctuation
remove_punct_dict = dict( (ord(punct),None) for punct in string.punctuation)

# function to return a list of lemmatized low case words after removing punctuation
def LemNormalize(text):
    return nltk.word_tokenize(text.lower().translate(remove_punct_dict))

#Keyword Matching
GREETING_INPUTS = ["hi",'hello','hola','greetings','wassup','hey','hwfa']
GREETINGS_RESPONSE =["howdy","hi","hey","what's good","hello","hey there"]
#function to return greetings
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETINGS_RESPONSE)

def response(user_response):
    #user_response = "what is chronic kidney disease"
    user_response_response = user_response.lower()

    robo_response = ""

    sent_tokens.append(user_response)
    #print(sent_tokens)
    TfidfVec = TfidfVectorizer(tokenizer = LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    #print(tfidf)
    vals = cosine_similarity(tfidf[-1],tfidf)
    #print(vals)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    score = flat[-2]
    #print(score)
    if(score == 0):
        robo_response = robo_response + "I apologize, i don't understand"
    elif score<0.5:
        New_query = 'https://www.google.com/search?q='+user_response
        article = Article(New_query)
        article.download()
        article.parse()
        article.nlp()
        qcorpus = article.text
        robo_response = qcorpus
    else:
        robo_response = robo_response+sent_tokens[idx]
    #print(robo_response)
    sent_tokens.remove(user_response)
    
    return robo_response

app=Flask(__name__)
@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/process',methods = ['POST'])
def process():
    user_input= request.form["user_input"]
    user_response = user_input.lower()
    if(user_response != "bye"):
        if(user_response == 'thanks' or user_response == 'thank you'):
            flag = False
            bot_response = "you are welcome"
        else:
            if(greeting(user_response)!= None):
                bot_response = greeting(user_response)
            else:
                bot_response = response(user_response)
    else:
        flag = False
        bot_response = "chat with you later ! "
    return render_template('index.html',user_input=user_input,bot_response=bot_response)    

if __name__ == '__main__':
    #DEBUG is SET to TRUE. CHANGE FOR PROD
    app.run(port=5000,debug=True)