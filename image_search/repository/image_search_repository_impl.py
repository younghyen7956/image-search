import os
import torch
from transformers import AutoProcessor, AutoModel
from pymilvus import connections, Collection, utility
from dotenv import load_dotenv
import numpy as np
from PIL import Image
from io import BytesIO
from fastapi import UploadFile
from image_search.repository.image_search_repository import ImageSearchRepository
import sys

class ImageSearchRepositoryImpl(ImageSearchRepository):
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            # .env 로딩은 인스턴스 최초 생성 시 1회만
            load_dotenv()
            cls.__instance = super().__new__(cls)
        return cls.__instance

    @classmethod
    def getInstance(cls):
        if cls.__instance is None:
            cls.__instance = cls()
        return cls.__instance

    def __init__(self):
        # 환경변수에서 설정 읽기
        self.host = os.getenv("MILVUS_HOST")
        self.port = int(os.getenv("MILVUS_PORT"))
        self.checkpoint = os.getenv("SIGLIP_CHECKPOINT")
        self.collection = os.getenv("MILVUS_COLLECTION")
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        # 모델과 DB 연결을 미리 초기화해 둘 수도 있습니다
        self._model = None
        self._processor = None
        self._collection = None
        self.connectDB()

    def get_model(self):
        if self._model is None:

            # checkpoint 도 환경변수에서
            processor = AutoProcessor.from_pretrained(
                self.checkpoint, trust_remote_code=True, use_fast=True
            )
            model = AutoModel.from_pretrained(
                self.checkpoint, trust_remote_code=True
            ).to(self.device)
            model.eval()

            self._processor = processor.image_processor
            self._model = model
        # print('success get model')
        return self._model, self._processor

    def connectDB(self):
        if self._collection is None:
            connections.connect(alias="default", host=self.host, port=self.port)
            self._collection = Collection(self.collection)
        # print('success connect db')
        return self._collection

    async def image_embedding(self, upload_file: UploadFile) -> np.ndarray:
        # startup 시 이미 loaded 된 model, processor 사용
        model, processor = self._model, self._processor
        device = next(model.parameters()).device

        contents = await upload_file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            feats = model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.squeeze(0).cpu().numpy()

    def search(self, query_vec, top_k):
        # startup 시 미리 connectDB()로 세팅된 self._collection 사용
        results = self._collection.search(
            data=[query_vec.tolist()],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=top_k,
            output_fields=["filename"]
        )
        return [
            {"filename": hit.entity.get("filename"), "score": hit.distance}
            for hits in results for hit in hits
        ]

