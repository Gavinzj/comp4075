import re, math
from collections import Counter

WORD = re.compile(r'\w+')

def get_cosine(vec1, vec2):
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator

def text_to_vector(text):
     words = WORD.findall(text)
     return Counter(words)

result=[]
topic=0
content1=""
for line in open("sport.txt"):
    content1 += line
vector1 = text_to_vector(content1)

content2=""
for i in range(0,19):
    for line in open("topic"+str(i)+".txt"):
        content2 += line
    
    vector2 = text_to_vector(content2)

    cosine = get_cosine(vector1, vector2)
    result.append(cosine)

for i in range(0,19):
    max=0
    if(result[i]>max):
        max = result[i]
        topic=i
print "topic:",topic," The cosine is",max