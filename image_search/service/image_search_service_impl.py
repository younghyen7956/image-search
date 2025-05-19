import os

from image_search.repository.image_search_repository_impl import ImageSearchRepositoryImpl
from image_search.service.image_search_service import ImageSearchService
from fastapi import UploadFile

class ImageSearchServiceImpl(ImageSearchService):
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
            cls.__instance.__ImageSearchRepository = ImageSearchRepositoryImpl.getInstance()

        return cls.__instance

    @classmethod
    def getInstance(cls):
        if cls.__instance is None:
            cls.__instance = cls()

        return cls.__instance

    async def imageSearch(self, upload_file: UploadFile, top_k: int):
        # 1) 업로드 받은 파일 → 벡터
        query_vec = await self.__ImageSearchRepository.image_embedding(upload_file)
        # 2) Milvus 검색
        results = self.__ImageSearchRepository.search(query_vec, top_k)
        # 3) URL만 생성하여 반환
        return [
            {
                "filename": r["filename"],
                "score": round(r["score"],4),
                "image": f"/images/{r['filename']}"
            }
            for r in results
        ]

    async def search(self, upload_file: UploadFile):
        query_vec = await self.__ImageSearchRepository.image_embedding(upload_file)
        # 2) Milvus 검색
        results = self.__ImageSearchRepository.search(query_vec, 1)
        # 3) URL만 생성하여 반환
        return [
            {
                "filename": r["filename"],
                "score": round(r["score"], 4),
                "image": f"/images/{r['filename']}"
            }
            for r in results
        ]

