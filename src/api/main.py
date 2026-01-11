"""FastAPI 메인 애플리케이션"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from .routes import router, init_pipeline, shutdown_pipeline
from ..inference.pipeline import PipelineConfig


def load_config() -> PipelineConfig:
    """설정 로드"""
    config_path = os.getenv("CONFIG_PATH", "configs/inference.yaml")

    if Path(config_path).exists():
        with open(config_path, encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        model_config = config_dict.get("model", {})
        preprocessing_config = config_dict.get("preprocessing", {})

        return PipelineConfig(
            model_path=os.getenv("MODEL_PATH", model_config.get("path")),
            ontology_config_path=os.getenv("ONTOLOGY_CONFIG", "configs/ontology.yaml"),
            use_neo4j=os.getenv("USE_NEO4J", "false").lower() == "true",
            quantize=os.getenv("QUANTIZE", "false").lower() == "true",
            target_size=(
                preprocessing_config.get("resize", {}).get("width", 1024),
                preprocessing_config.get("resize", {}).get("height", 1024),
            ),
            normalize=preprocessing_config.get("normalize", True),
        )

    return PipelineConfig(
        model_path=os.getenv("MODEL_PATH"),
        ontology_config_path=os.getenv("ONTOLOGY_CONFIG", "configs/ontology.yaml"),
        use_neo4j=os.getenv("USE_NEO4J", "false").lower() == "true",
        quantize=os.getenv("QUANTIZE", "false").lower() == "true",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """앱 라이프사이클 관리"""
    # 시작 시
    config = load_config()
    init_pipeline(config)
    yield
    # 종료 시
    shutdown_pipeline()


# FastAPI 앱 생성
app = FastAPI(
    title="SDC Display Defect Analyzer",
    description="Cosmos Reason VLM 기반 디스플레이 결함 분석 API",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(router, prefix="/api/v1")


@app.get("/", include_in_schema=False)
async def root():
    """루트 -> 문서로 리다이렉트"""
    return RedirectResponse(url="/docs")


@app.get("/version")
async def version():
    """버전 정보"""
    return {
        "name": "SDC Display Defect Analyzer",
        "version": "0.1.0",
        "model": "Cosmos-Reason1-7B",
    }


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 1,
    reload: bool = False,
):
    """서버 실행"""
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
    )


if __name__ == "__main__":
    run_server()
