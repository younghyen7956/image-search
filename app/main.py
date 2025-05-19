import os
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from image_search.controller.image_search_controller import ImageSearchRouter
from image_search.repository.image_search_repository_impl import ImageSearchRepositoryImpl
from vector_db_crud.controller.milvus_db_controller import MilvusDBRouter

# 1) 프로젝트 루트 디렉터리 계산
BASE_DIR = Path(__file__).resolve().parent.parent

# 2) StaticFiles 마운트
IMAGE_DIR  = BASE_DIR / "images"
UPLOAD_DIR = BASE_DIR / "photo"
IMAGE_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(debug=True)
# 기존 저장 이미지
app.mount("/images",  StaticFiles(directory=str(IMAGE_DIR)),  name="images")
# # 업로드된 쿼리 이미지 (필요시)
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

app.include_router(ImageSearchRouter)
app.include_router(MilvusDBRouter)

origins = os.getenv("ALLOWED_ORIGINS", "").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def on_startup():
    """
    fastapi 시작시 모델, 프로세서, db 연결
    """
    repo = ImageSearchRepositoryImpl.getInstance()
    # 모델과 프로세서 로드
    repo.get_model()
    # Milvus 연결 및 컬렉션 준비
    repo.connectDB()
    # (원한다면 임베딩 캐시나 인덱스 로드도 이곳에서)
    print("✅ Startup: model loaded and DB connected.")

if __name__ == "__main__":
    host = os.getenv("FASTAPI_HOST", "0.0.0.0")
    port = int(os.getenv("FASTAPI_PORT", 8000))
    uvicorn.run(app, host=host, port=port,log_level="debug")
