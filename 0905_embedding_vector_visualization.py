################################################################################
#
# - 임베딩 벡터의 시각화(Embedding Visualization)
#     - 구글은 임베딩 프로젝터(embedding projector)라는 데이터 시각화 도구를 지원합니다.
#     - 이번 챕터에서는 임베딩 프로젝터를 사용하여 학습한 임베딩 벡터들을 시각화해보겠습니다.
#
# - 시각화를 위해서는 이미 모델을 학습하고,
#     - 파일로 저장되어져 있어야 합니다.
#     - 모델이 저장되어져 있다면 아래 커맨드를 통해 시각화에 필요한 파일들을 생성할 수 있습니다.
#
# ```
# python -m gensim.scripts.word2vec2tensor --input .eng_w2v --output .eng_w2v
# mv .eng_w2v_metadata.tsv dot_eng_w2v_metadata.tsv
# mv .eng_w2v_tensor.tsv dot_eng_w2v_tensor.tsv
# ```
#
# - 임베딩 프로젝터를 사용하여 시각화하기
#     - https://projector.tensorflow.org/ 오픈
#     - 로드 버튼클릭
#     - dot_eng_w2v_metadata.tsv, dot_eng_w2v_tensor.tsv 로드