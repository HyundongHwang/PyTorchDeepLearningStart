import myutil as mu

################################################################################
# - NLP에서의 원-핫 인코딩(One-hot encoding)
#   - 우선, 한국어 자연어 처리를 위해 코엔엘파이 패키지를 설치합니다.
#
# ```
# pip install konlpy
# ```
#
#   - 코엔엘파이의 Okt 형태소 분석기를 통해서 우선 문장에 대해서 토큰화를 수행하였습니다.

from konlpy.tag import Okt

okt = Okt()
token = okt.morphs("나는 자연어 처리를 배운다")
mu.log("token", token)
word2index = {}

for voca in token:
    if voca not in word2index.keys():
        word2index[voca] = len(word2index)

mu.log("word2index", word2index)


################################################################################
# - 토큰을 입력하면 해당 토큰에 대한 원-핫 벡터를 만들어내는 함수를 만들었습니다.


def one_hot_encoding(word, word2index):
    one_hot_vector = [0] * len(word2index)
    index = word2index[word]
    one_hot_vector[index] = 1
    return one_hot_vector


mu.log("one_hot_encoding('자연어', word2index)", one_hot_encoding("자연어", word2index))

################################################################################
# - 원-핫 인코딩(One-hot encoding)의 한계
#   - 이러한 표현 방식은 단어의 개수가 늘어날 수록, 벡터를 저장하기 위해 필요한 공간이 계속 늘어난다는 단점이 있습니다.
#   - 또한 원-핫 벡터는 단어의 유사도를 표현하지 못한다는 단점이 있습니다.
#   - 단어 간 유사성을 알 수 없다는 단점은 검색 시스템 등에서 심각한 문제입니다.