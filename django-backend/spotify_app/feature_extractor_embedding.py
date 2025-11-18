# spotify_app/feature_extractor_embedding.py

from sentence_transformers import SentenceTransformer
from .feature_extractor import extract_features

# 384차원 MiniLM 임베딩 모델
EMBED_MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def extract_features_with_embedding(metadata):
    """ Mega Extractor + 텍스트 임베딩 포함 """

    base = extract_features(metadata)

    # 텍스트 기반 문장 생성
    text = (
        metadata.get("name", "") + " " +
        " ".join(metadata.get("artists", [])) + " " +
        ", ".join(metadata.get("genres", []))
    )

    embedding = EMBED_MODEL.encode(text).tolist()

    base["text_embedding"] = embedding
    base["text_embedding_dim"] = len(embedding)
    base["version"] = "mega-embedding"

    return base
