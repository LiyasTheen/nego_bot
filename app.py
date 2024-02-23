from flask import Flask, request, jsonify, render_template,render_template_string

import gym
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete

import numpy as np
import random

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os

import random  
import json
import pickle
import numpy as np
import tensorflow as tf

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')


class PriceEnv(Env):
  def __init__(self,limit,sp):
    self.limit = limit
    self.sp = sp
    self.bp = self.sp//2
    self.action_space = Discrete(sp-limit)
    self.observation_space = Box(low=0, high=self.sp, shape=(2,)) # Modified observation space
    self.state = sp
    self.round = 7
    self.prev_state = sp

  def step(self, action, bp=-1):

    if bp==-1:
      self.bp = int(np.min(Box(self.bp,self.state,(5,)).sample()))
    else:
      self.bp = bp

    self.state -=action
    self.round-=1

    if self.state>=1:
      ratio = self.state/self.bp
    else:
      ratio = 0
    sbl = ((self.state-self.bp)/self.limit)

    reward = 0

    done = False
    if self.round == 1:
      if self.bp>=self.limit and self.state>self.bp:
        reward -= sbl+ratio+10
      done = True

    if self.bp>=self.limit:
      if self.state>self.bp:
        reward += ((sbl+ratio)*self.round)
      if self.bp>=self.state:
        self.state = self.bp
        reward += (((self.state/self.limit)+ratio)*7/self.round)
        done = True
    else:
      if self.bp>=self.state:
        reward += sbl*self.round
        self.state = self.bp

        done = True

    if self.sp<=self.limit:
      reward -= 10
      done = True


    obs = np.array([self.state, self.bp]) # Modified observation
    info = {}

    return obs, reward, done, info

  def render(self):
    pass

  def reset(self):
    self.bp = self.sp//2
    self.state = self.sp
    self.round = 7
    return np.array([self.state, self.bp]) # Modified observation
  
class ChatBot():
  def __init__(self):
    self.lemmatizer = WordNetLemmatizer()
    self.intents = json.loads(open('intents.json').read())

    self.words = pickle.load(open('words.pkl','rb'))
    self.classes = pickle.load(open('classes.pkl','rb'))
    self.model = tf.keras.models.load_model('chatbot_model.h5')

  def clean_up_sentence(self,sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [self.lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


  def bag_of_words(self,sentence):
    sentence_words = self.clean_up_sentence(sentence)
    bag = [0] * len(self.words)
    for w in sentence_words:
      for i, word in enumerate(self.words):
        if word == w:
          bag[i]=1
    return np.array(bag)

  def predict_class(self,sentence):
    bow = self.bag_of_words(sentence)
    res = self.model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x:x[1], reverse=True)
    return_list = []
    for r in results:
      return_list.append({'intent':self.classes[r[0]],'probability':str(r[1])})
    return return_list

  def get_response(self,intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
      if i['tag'] ==tag:
        result = random.choice(i['responses'])
        break
    return result


app = Flask(__name__)

@app.route("/learn/<seller_id>", methods=['GET'])
def learn_model(seller_id):
    details = json.loads(open('seller_details.json').read())
    limit = int(request.args.get('limit'))
    price = int(request.args.get('price'))
    seller_id = int(seller_id)
    
    found=0
    for i in details["sellers"]:
        if i['id'] == seller_id:
            i['limit'] = limit
            i['price'] = price
            found = 1
            break
    if found==0:
        details["sellers"].append({"id":seller_id,"limit":limit,"price":price})
    with open('seller_details.json', 'w') as f:
        json.dump(details, f, indent=4)
        
    env = PriceEnv(limit,price)
    log_path = os.path.join('Training','Logs')  
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

    model.learn(total_timesteps=100000)

    shower_path = os.path.join('Training','Saved Models',f'Price_Model_{seller_id}')
    model.save(shower_path)

    return 'Seller ID: {} Limit: {} Price: {}'.format(seller_id,limit, price)


messages = [{'sender': 'ai', 'content': "Hi there!I am Seller's assistant. How can I assist you?"}]

bot = ChatBot()
details = json.loads(open('seller_details.json').read())
limit=price=0
done = False
total_reward = 0
isCompleted = isSuccess = 0
env=model=obs=None
nego = 0
nego_res = 0


@app.route("/nego_bot/<seller_id>",methods=['GET'])
def nego_bot(seller_id):
  
  global messages,limit, price, env, model ,done,total_reward,nego,isCompleted,isSuccess,obs,ints,nego_res

  if env is None and model is None:
    seller_id = int(seller_id)
    found = 0
    for i in details["sellers"]:
        if i['id'] == seller_id:
            limit = int(i['limit'])
            price = int(i['price'])
            found = 1
            break
    if found == 0:
        return f"Error: No seller found {seller_id}"
    
    env = PriceEnv(limit, price)
    shower_path = os.path.join('Training', 'Saved Models', f'Price_Model_{seller_id}')
    model = PPO.load(shower_path, env)
    obs = env.reset()
  
      
  if nego==0: 
    message = request.args.get('msg')
    ints = bot.predict_class(message)

  
  if nego==1:
    if not done:
      bp = int(request.args.get('msg'))
      messages.append({'sender':'user','content':bp})
      
      action, _ = model.predict(obs, deterministic=False)

      next_obs, reward, done, info = env.step(action,bp)
      total_reward += reward
      obs = next_obs

    if done:
      isCompleted=1
      if obs[0]==obs[1]:
        print("hi")
        nego_res = bot.get_response([{'intent':"nego_success"}], bot.intents)
        isSuccess = 1 
        nego=0
        
      
  else:
    res = bot.get_response(ints, bot.intents)
    messages.append({'sender':'user','content':message})
    messages.append({'sender':'ai','content':res})

  if(ints[0]['intent']=="negotiation"):
    
    nego=1
    if isCompleted==0:
      ai = f"My price is: {env.state}"
      messages.append({'sender':'ai','content':ai})
    elif isSuccess!=1:
      res = bot.get_response([{'intent':"nego_doubt"}], bot.intents)
      ai = f"{res}{obs[0]}. Type 'ok' to confirm or 'cancel' to Cancel"
      messages.append({'sender':'ai','content':ai})
      msg = request.args.get('msg')
      if msg.lower() == "ok":
        messages.append({'sender':'user','content':msg})
        nego_res = bot.get_response([{'intent':"nego_success"}], bot.intents)
        isSuccess = 1
      elif msg.lower() == "cancel":
        messages.append({'sender':'user','content':msg})
        nego_res = bot.get_response([{'intent':"nego_failure"}], bot.intents)
        isSuccess=-1

    if isSuccess == 1:
      nego=0
      nego_res = f"{nego_res}{obs[0]}."
      messages.append({'sender':'ai','content':nego_res})
    elif isSuccess == -1:
      nego=0
      nego_res = f"{nego_res}."
      messages.append({'sender':'ai','content':nego_res})

    if isCompleted==1 and isSuccess==1:
      messages.append({'sender':'ai','content':nego_res})
      # return render_template('index.html', messages=messages)


  return render_template('index.html', messages=messages)

@app.route('/')
def index():
    global messages,limit, price, env, model ,done,total_reward,nego,isCompleted,isSuccess,obs,ints,nego_res
    messages = [
        {'sender': 'ai', 'content': "Hi there!I am Seller's assistant. How can I assist you?"},
    ]
    limit=price=0
    done = False
    total_reward = 0
    isCompleted = isSuccess = 0
    env=model=obs=None
    nego = 0
    nego_res = 0
    return render_template('index.html', messages=messages)

if __name__ == "__main__":
    app.run(debug=True)