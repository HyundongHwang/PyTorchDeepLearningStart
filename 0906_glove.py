################################################################################
# - 글로브(GloVe)
#   - 글로브(Global Vectors for Word Representation, GloVe)는
#       - 카운트 기반과 예측 기반을 모두 사용하는 방법론으로
#       - 2014년에 미국 스탠포드대학에서 개발한 단어 임베딩 방법론입니다.
#   - 앞서 학습하였던 기존의 카운트 기반의 LSA(Latent Semantic Analysis)와
#       - 예측 기반의 Word2Vec의 단점을 지적하며 이를 보완한다는 목적으로 나왔고,
#       - 실제로도 Word2Vec만큼 뛰어난 성능을 보여줍니다.
#   - LSA는 카운트 기반으로 코퍼스의 전체적인 통계 정보를 고려하기는 하지만,
#       - 왕:남자 = 여왕:? (정답은 여자)와 같은
#       - 단어 의미의 유추 작업(Analogy task)에는 성능이 떨어집니다.
#   - Word2Vec는 예측 기반으로 단어 간 유추 작업에는 LSA보다 뛰어나지만,
#       - 임베딩 벡터가 윈도우 크기 내에서만 주변 단어를 고려하기 때문에
#       - 코퍼스의 전체적인 통계 정보를 반영하지 못합니다.
#   - GloVe는 이러한 기존 방법론들의 각각의 한계를 지적하며,
#       - LSA의 메커니즘이었던 카운트 기반의 방법과
#       - Word2Vec의 메커니즘이었던 예측 기반의 방법론 두 가지를 모두 사용합니다.