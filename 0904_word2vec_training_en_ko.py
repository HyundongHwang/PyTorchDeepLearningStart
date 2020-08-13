import myutil as mu
import os.path

################################################################################
# - 영어/한국어 Word2Vec 훈련시키기
#     - 영어와 한국어 훈련 데이터에 대해서 Word2Vec을 학습해보겠습니다.
#     - gensim 패키지에서 Word2Vec은 이미 구현되어져 있으므로,
#     - 별도로 Word2Vec을 구현할 필요없이 손쉽게 훈련시킬 수 있습니다

################################################################################
# - 영어 Word2Vec 만들기
#   - 영어 데이터를 다운로드 받아 직접 Word2Vec 작업을 진행해보도록 하겠습니다.
#   - 영어로 된 코퍼스를 다운받아 전처리를 수행하고,
#   - 전처리한 데이터를 바탕으로 Word2Vec 작업을 진행하겠습니다.

################################################################################
# - 훈련 데이터 이해하기
#   - zip 파일의 압축을 풀면 ted_en-20160408.xml이라는 파일을 얻을 수 있습니다.

import nltk

nltk.download("punkt")

import urllib.request
import zipfile
from lxml import etree
import re
from nltk.tokenize import word_tokenize, sent_tokenize

if not os.path.isfile(".ted_en-20160408.xml"):
    urllib.request.urlretrieve(
        url="https://raw.githubusercontent.com/GaoleMeng/RNN-and-FFNN-textClassification/master/ted_en-20160408.xml",
        filename=".ted_en-20160408.xml"
    )

################################################################################
# - 훈련 데이터 전처리하기

targetXML = open(
    file=".ted_en-20160408.xml",
    mode="r",
    encoding="UTF8")

target_text = etree.parse(targetXML)
target_text_xpath = target_text.xpath('//content/text()')
mu.log("len(target_text_xpath)", len(target_text_xpath))
mu.log("target_text_xpath[:5]", target_text_xpath[:5])

parse_text = "\n".join(target_text_xpath)
mu.log("len(parse_text)", len(parse_text))
mu.log("parse_text", parse_text[:300])

# 정규 표현식의 sub 모듈을 통해
# content 중간에 등장하는 (Audio), (Laughter) 등의 배경음 부분을 제거.
content_text = re.sub(r'\([^)]*\)', '', parse_text)
mu.log("len(content_text)", len(content_text))
mu.log("content_text", content_text[:300])

# 입력 코퍼스에 대해서 NLTK를 이용하여 문장 토큰화를 수행.
sent_text = sent_tokenize(content_text)
mu.log("len(sent_text)", len(sent_text))
mu.log("sent_text", sent_text[:5])

# 각 문장에 대해서 구두점을 제거하고, 대문자를 소문자로 변환.
normalized_text = []

for string in sent_text:
    tokens = re.sub(r"[^a-z0-9]+", " ", string.lower())
    normalized_text.append(tokens)

mu.log("len(normalized_text)", len(normalized_text))
mu.log("normalized_text", normalized_text[:5])

# 각 문장에 대해서 NLTK를 이용하여 단어 토큰화를 수행.
result = []

result = [
    word_tokenize(sentence)
    for sentence in normalized_text
]

################################################################################
# - 상위 3개 문장만 출력해보았는데 토큰화가 수행되었음을 볼 수 있습니다.
# - 이제 Word2Vec 모델에 텍스트 데이터를 훈련시킵니다.

mu.log("len(result)", len(result))
mu.log("result", result[:5])

################################################################################
# - Word2Vec 훈련시키기
# ```
# pip install gensim
# ```

from gensim.models import Word2Vec

# size
#   워드 벡터의 특징 값.
#   즉, 임베딩 된 벡터의 차원.
#   밀집벡터의 길이
# window
#   컨텍스트 윈도우 크기,
#   함께 훈련한 연관단어 갯수
# min_count
#   단어 최소 빈도 수 제한
#   (빈도가 적은 단어들은 학습하지 않는다.)
# workers
#   학습을 위한 프로세스 수
# sg =
#   0은 CBOW, 주변단어로 가운데 단어 맞추기,
#   1은 Skip-gram, 가운데 단어로 주변단어 맞추기
model = Word2Vec(
    sentences=result,
    size=100,
    window=5,
    min_count=5,
    workers=4,
    sg=0
)

################################################################################
# - Word2Vec는 입력한 단어에 대해서 가장 유사한 단어들을 출력하는
# - model.wv.most_similar을 지원합니다.

model_result = model.wv.most_similar("man")
mu.log("model_result", model_result)

################################################################################
# - Word2Vec 모델 저장하고 로드하기
#   - 공들여 학습한 모델을 언제든 나중에 다시 사용할 수 있도록 컴퓨터 파일로 저장하고 다시 로드해보겠습니다.

from gensim.models import KeyedVectors

model.wv.save_word2vec_format(".eng_w2v")
loaded_model = KeyedVectors.load_word2vec_format(".eng_w2v")
loaded_model_result = loaded_model.wv.most_similar("man")
mu.log("loaded_model_result", loaded_model_result)

################################################################################
# - 한국어 Word2Vec 만들기
#     - 위키피디아 한국어 덤프 파일을 다운받아서 한국어로 Word2Vec을 직접 진행해보도록 하겠습니다.
#     - 영어와 크게 다른 점은 없지만 한국어는 형태소 토큰화를 해야만 좋은 성능을 얻을 수 있습니다.
#     - 간단히 말해 형태소 분석기를 사용합니다.
# ```
# curl https://dumps.wikimedia.org/kowiki/latest/kowiki-latest-pages-articles.xml.bz2 -o .kowiki-latest-pages-articles.xml.bz2
# pip install wikiextractor
# python -m wikiextractor.WikiExtractor .kowiki-latest-pages-articles.xml.bz2
# pwsh
# Get-ChildItem text/wiki_* -Recurse | ForEach-Object { Get-Content $_.FullName | Add-Content .wiki_data.txt }
# ```

################################################################################
# - 훈련 데이터 전처리 하기
#   - 파일이 정상적으로 저장되었는지 5개의 줄만 출력해보겠습니다.

f = open(".wiki_data.txt", mode="r", encoding="utf8")

i = 0

while True:
    line = f.readline()
    if line != "\n":
        i = i + 1
        mu.log("i", i)
        mu.log("line", line)
    if i == 5:
        break

f.close()

################################################################################
# - 이제 본격적으로 Word2Vec을 위한 학습 데이터를 만들어보겠습니다.
#     - 여기서는 형태소 분석기로 KoNLPy의 Okt를 사용하여
#     - 명사만을 추출하여 훈련 데이터를 구성하겠습니다.
#     - 위 작업은 시간이 꽤 걸립니다.
#     - 훈련 데이터를 모두 만들었다면,
#     - 훈련 데이터의 길이를 확인해보겠습니다.

from konlpy.tag import Okt

okt = Okt()
fread = open(".wiki_data.txt", mode="r", encoding="utf8")
n = 0
result = []

while True:
    line = fread.readline()

    if not line:
        break

    n = n + 1

    if n % 1000 == 0:
        mu.log("n", n)

    tokenlist = okt.pos(line, stem=True, norm=True)
    temp = []

    for word in tokenlist:
        if word[1] in ["Noun"]:
            temp.append(word[0])

    if temp:
        result.append(temp)

fread.close()

################################################################################
# - 약 240만여개의 줄(line)이 명사 토큰화가 되어 저장되어 있는 상태입니다.
# - 이제 이를 Word2Vec으로 학습시킵니다.


mu.log("len(result)", len(result))

################################################################################
# - Word2Vec 훈련시키기

from gensim.models import Word2Vec

model = Word2Vec(result, size=100, min_count=5, workers=4, sg=0)

################################################################################
# - 학습을 다했다면 이제 임의의 입력 단어로부터 유사한 단어들을 구해봅시다.

model_result = model.most_similar("대한민국")
mu.log("model_result", model_result)

model_result = model.most_similar("어벤져스")
mu.log("model_result", model_result)

model_result = model.most_similar("반도")
mu.log("model_result", model_result)
체

################################################################################
# - 사전 훈련된 Word2Vec 임베딩(Pre-trained Word2Vec embedding) 소개
#   - 위키피디아 등의 방대한 데이터로 사전에 훈련된 워드 임베딩(pre-trained word embedding vector)를 가지고 와서
#   - 해당 벡터들의 값을 원하는 작업에 사용 할 수도 있습니다.
#   - 사전 훈련된 워드 임베딩을 가져와서 간단히 단어들의 유사도를 구해보는 실습을 해보겠습니다.


################################################################################
# - 영어
#     - 구글이 제공하는 사전 훈련된(미리 학습되어져 있는) Word2Vec 모델을 사용하는 방법에 대해서 알아보도록 하겠습니다.
#     - 구글은 사전 훈련된 3백만 개의 Word2Vec 단어 벡터들을 제공합니다.
#     - 각 임베딩 벡터의 차원은 300입니다.
#     - gensim을 통해서 이 모델을 불러오는 건 매우 간단합니다.
#     - 이 모델을 다운로드하고 파일 경로를 기재하면 됩니다.
# ```
# www https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
# gzip -d ~/다운로드/GoogleNews-vectors-negative300.bin.gz
# mv ~/다운로드/GoogleNews-vectors-negative300.bin .GoogleNews-vectors-negative300.bin
# ```


import gensim

model = gensim.models.KeyedVectors.load_word2vec_format(".GoogleNews-vectors-negative300.bin", binary=True)
mu.log("model.vectors.shape", model.vectors.shape)

mu.log("model.similarity('this', 'is')", model.similarity("this", "is"))
mu.log("model.similarity('book', 'post')", model.similarity("book", "post"))
mu.log("model['book']", model['book'])

################################################################################
# - 한국어
#     - 한국어의 미리 학습된 Word2Vec 모델은 박규병님의 깃허브 에 공개되어져 있습니다.
#
# ```
# www https://drive.google.com/file/d/0B0ZXk88koS2KbDhXdWg1Q2RydlU/view
# unzip ~/다운로드/ko.zip -d .ko
# mv ~/다운로드/ko.bin .ko.bin
# ```


import gensim

model = gensim.models.Word2Vec.load(".ko/ko.bin")
result = model.mv.most_similar("강아지")
mu.log("result", result)
