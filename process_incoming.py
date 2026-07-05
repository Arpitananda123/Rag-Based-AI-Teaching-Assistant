import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import requests
import numpy as np
import joblib



def create_embedding(text_list):

    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        }
    )

    data = r.json()

    if "embeddings" not in data:
        print("ERROR RESPONSE:")
        print(data)
        return []

    return data["embeddings"]

def inference(prompt):
    r = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    }
)
    response=r.json()
    print(response)
    return response




    

df=joblib.load('embeddings.joblib')
incoming_query=input("Ask a Question: ")
question_embedding=create_embedding([incoming_query])[0]
print(question_embedding)

#Find similarities of question_embedding with other embeddings
similarities=cosine_similarity(np.vstack(df['embedding']),[question_embedding]).flatten() 
top_results=5
max_idx=similarities.argsort()[::-1][0:top_results]   
print(max_idx) 
new_df=df.loc[max_idx]

#prompt=f''' I am tecahing web development in my Sigma web development course.Here are video subtitle chunks containing video title,video number,start time in seconds,end time in seconds,the text at that time:

#{new_df[["title","number","start","end","text"]].to_json()}
#---------------------------------------------------
#"{incoming_query}"
#User asked this question related to the video chunks,you have to answer in a human way(don't mention the above format,its just for your reference) where and how much content is taught in which video(in which video and at what timestamp) and guide the user to go to that particular video.If user asks unrelated question,tell him that you can only answer questions related to the course'''

prompt = f"""
You are an AI Teaching Assistant for the Sigma Web Development Course.

Your job is to help students find exactly where a topic is taught in the course videos.

=========================
COURSE CONTENT
=========================

{new_df[["title","number","start","end","text"]].to_json()}

=========================
STUDENT QUESTION
=========================

{incoming_query}

=========================
INSTRUCTIONS
=========================

1. Analyze the course content carefully.
2. Answer ONLY using the provided course content.
3. Mention:
   - Video Number
   - Video Title
   - Relevant Timestamp(s)
4. Explain briefly what is taught in that section.
5. Guide the student to watch the most relevant video first.
6. If information is spread across multiple videos, mention all relevant videos in order.
7. If the answer is not available in the provided content, respond:
   "I could not find enough information in the course material to answer this question."
8. If the question is unrelated to web development or the course, respond:
   "I can only answer questions related to the Sigma Web Development Course."

=========================
RESPONSE FORMAT
=========================

Answer naturally like a teacher.

Example:

You can find this topic in Video 45: CSS Flexbox Basics.

Relevant timestamps:
• 12:30 - 18:45
• 20:10 - 24:50

In this section, Flexbox properties such as justify-content and align-items are explained with examples.

I recommend watching Video 45 first and then Video 46 for advanced layouts.

Now answer the student's question.
"""

with open("prompt.txt","w") as f:
    f.write(prompt)

#response=inference(prompt)
response=inference(prompt)['response']
print(response)

with open("response.txt","w",encoding="utf-8") as f:
    f.write(response)

#for index,item in new_df.iterrows():
    #print(index,item["title"],item["number"],item["text"],item["start"],item["end"])