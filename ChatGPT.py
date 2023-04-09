# https://www.parlimen.gov.my/files/hindex/pdf/DR-13032018.pdf

# u are a chatbot that faham malaysia hansard, so all your knowledge related to malaysia hansard, and always reply in standard malay, if out of the context of malaysia hansard, reply `ini keluar konteks`

!pip3 install openai

import os

os.environ['OPENAI_API_KEY'] = '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'

import openai
import numpy as np
import os

!pip3 install PyPDF2

pdf_reader = PyPDF2.PdfReader('2. KAEDAH-KAEDAH UNIVERSITI MALAYA (TATATERTIB PELAJAR-PELAJAR) 1999.pdf')
len(pdf_reader.pages)
pdf_reader.pages[0].extract_text().strip()

# !pip3 install unidecode

import re
from unidecode import unidecode

def cleaning(string):
  return re.sub(r'[ ]+', ' ', unidecode(string).replace('\n', ' ')).strip()
  
def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input = [text], model=model)['data'][0]['embedding']
   
v = get_embedding(cleaning(pdf_reader.pages[0].extract_text()))
len(v)


%%time

from tqdm import tqdm

vs = []
for i in tqdm(range(len(pdf_reader.pages))):
  c = cleaning(pdf_reader.pages[i].extract_text())
  v = get_embedding(c)
  vs.append(v)
  
from sklearn.metrics.pairwise import cosine_similarity

q = 'saya pelajar baru, apa yang patut saya tahu'
q_v = get_embedding(q)

score = cosine_similarity(vs_np, np.array([q_v]))[:,0]
c_best_doc = cleaning(pdf_reader.pages[int(np.argmax(score))].extract_text())

prompting = f'teks `{c_best_doc}`, berdasarkan teks, jawab soalan berikut, `{q}`, jawab dalam standard malay'

response = openai.Completion.create(
  model="text-davinci-003",
  prompt=prompting,
  temperature=0.7,
  max_tokens=384,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0,
)

response.choices[0].text


r = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "user", "content": prompting},
    ]
)





