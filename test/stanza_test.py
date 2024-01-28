import stanza
import spacy

# 加载Spacy的英文模型
nlp = spacy.load('en_core_web_sm')

# 定义一个函数来提取名词短语
def extract_noun_phrases(text):
    doc = nlp(text)
    for chunk in doc.noun_chunks:
        print(chunk.text)

# 调用函数来提取名词短语
text = "What is the person in the black suit in the image warning people against?"
extract_noun_phrases(text)