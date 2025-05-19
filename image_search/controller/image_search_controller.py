from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException,UploadFile, File
from fastapi.responses import FileResponse
from image_search.service.image_search_service_impl import ImageSearchServiceImpl

ImageSearchRouter = APIRouter()

BASE_DIR = Path(__file__).resolve().parents[2]
IMAGE_DIR = BASE_DIR / "images"

def injectSearchService() -> ImageSearchServiceImpl:
    # 싱글톤 인스턴스를 반환
    return ImageSearchServiceImpl.getInstance()

@ImageSearchRouter.post('/image-search')
async def request_image_search_result(
    file: UploadFile = File(..., description="검색할 이미지를 업로드하세요"),
    service: ImageSearchServiceImpl = Depends(injectSearchService),
):
    try:
        results = await service.imageSearch(file, top_k=1)
        if not results:
            raise HTTPException(status_code=404, detail="검색 결과가 없습니다.")

        first = results[0]
        filename = first["filename"]  # ex. "abc.png"

        # 절대경로 조합
        file_path = IMAGE_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="서버에 이미지 파일이 없습니다.")

        ext = file_path.suffix.lower().lstrip(".")
        media_type = f"image/{'jpeg' if ext == 'jpg' else ext}"

        return FileResponse(path=str(file_path), media_type=media_type)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류: {e}")