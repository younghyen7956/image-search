from fastapi import APIRouter, Depends, HTTPException,UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel

from vector_db_crud.service.milvus_db_service_impl import MilvusDBServiceImpl
from typing import List, Optional
MilvusDBRouter = APIRouter()

def injectMilvusDBService() -> MilvusDBServiceImpl:
    # 싱글톤 인스턴스를 반환
    return MilvusDBServiceImpl.getInstance()

@MilvusDBRouter.post('/create')
def create_collection(
    name: str = None,
    dim: int = None,
    fields: list = None,
    overwrite: bool = False,
    service: MilvusDBServiceImpl = Depends(injectMilvusDBService)
):
    try:
        service.create_collection(
            name=name,
            dim=dim,
            fields=fields,
            overwrite=overwrite
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"message": "컬렉션 생성 성공", "collection": name}

@MilvusDBRouter.delete("/drop_collection")
def drop_collection(
    collection_name: str,
    service: MilvusDBServiceImpl = Depends(injectMilvusDBService)
):
    try:
        service.drop_collection(name=collection_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"message": f"컬렉션 '{collection_name}' 삭제 완료"}

@MilvusDBRouter.post("/upsert")
def upsert_entities(
    collection_name: str,
    data: list[dict],
    service: MilvusDBServiceImpl = Depends(injectMilvusDBService)
):
    try:
        service.upsert(data=data,collection_name=collection_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"message": f"{len(data)}개 엔티티 upsert 완료 in '{collection_name}'"}

@MilvusDBRouter.post("/search")
def search_vectors(
        self,
        query_vectors: List[List[float]],
        top_k: int = 3,
        metric: str = "COSINE",
        nprobe: int = 10,
        output_fields: Optional[List[str]] = None,
        collection_name: str = None,
        service : MilvusDBServiceImpl = Depends(injectMilvusDBService)
):

    try:
        results = service.search(
            query_vectors=query_vectors,
            top_k=top_k,
            metric=metric,
            nprobe=nprobe,
            output_fields=output_fields,
            collection_name=collection_name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"results": results}

class DeleteRequest(BaseModel):
    ids: List[int]

@MilvusDBRouter.delete("/delete")
def delete(
    collection_name: str,
    req: DeleteRequest,
    service: MilvusDBServiceImpl = Depends(injectMilvusDBService)
):
    try:
        service.delete_by_ids(ids=req.ids, collection_name=collection_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"message": f"IDs {req.ids} 삭제 완료 from '{collection_name}'"}

@MilvusDBRouter.post("/collections_list")
def list_collections(
    service: MilvusDBServiceImpl = Depends(injectMilvusDBService)
):
    try:
        cols = service.list_collections()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"collections": cols}

@MilvusDBRouter.post("/describe")
def describe_collection(
    collection_name: str,
    service: MilvusDBServiceImpl = Depends(injectMilvusDBService)
):
    try:
        info = service.describe_collection(name=collection_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"schema": info}

@MilvusDBRouter.post("/query_collection")
def query_entities(
    collection_name: str,
    top_k: int = 5,
    ids: int = 1,
    service: MilvusDBServiceImpl = Depends(injectMilvusDBService)
):
    try:
        data = service.query_entities(collection_name=collection_name,top_k=top_k,ids=ids)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"entities": data}
