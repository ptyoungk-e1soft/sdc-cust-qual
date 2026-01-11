"""추론 파이프라인"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from .analyzer import DefectAnalyzer
from .types import AnalysisResult
from ..data.preprocessing import ImagePreprocessor


@dataclass
class PipelineConfig:
    """파이프라인 설정"""

    model_path: str | None = None
    ontology_config_path: str | None = None
    use_neo4j: bool = False
    quantize: bool = False

    # 전처리 설정
    target_size: tuple[int, int] = (1024, 1024)
    normalize: bool = True

    # 배치 처리 설정
    max_batch_size: int = 8
    batch_timeout: float = 5.0

    # 캐싱
    enable_cache: bool = True
    cache_ttl: int = 3600


class InferencePipeline:
    """추론 파이프라인

    이미지 전처리 → VLM 추론 → GraphRAG 추론 → 결과 반환
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.preprocessor = ImagePreprocessor(
            target_size=self.config.target_size,
            normalize=self.config.normalize,
        )
        self.analyzer: DefectAnalyzer | None = None
        self._cache: dict[str, AnalysisResult] = {}

    def initialize(self) -> None:
        """파이프라인 초기화"""
        self.analyzer = DefectAnalyzer(
            model_path=self.config.model_path,
            ontology_config_path=self.config.ontology_config_path,
            use_neo4j=self.config.use_neo4j,
            quantize=self.config.quantize,
        )
        self.analyzer.initialize()

    def shutdown(self) -> None:
        """파이프라인 종료"""
        if self.analyzer:
            self.analyzer.shutdown()
        self._cache.clear()

    def process(
        self,
        image: Image.Image | str | Path,
        additional_context: str | None = None,
        use_cache: bool = True,
    ) -> AnalysisResult:
        """단일 이미지 처리"""
        if self.analyzer is None:
            self.initialize()

        # 이미지 로드 및 전처리
        if isinstance(image, (str, Path)):
            image_path = str(image)
            image = self.preprocessor.process(image_path)

            # 캐시 확인
            if use_cache and self.config.enable_cache and image_path in self._cache:
                return self._cache[image_path]
        else:
            image_path = None

        # 분석 수행
        result = self.analyzer.analyze(
            image=image,
            additional_context=additional_context,
        )

        # 캐시 저장
        if image_path and self.config.enable_cache:
            self._cache[image_path] = result

        return result

    def process_batch(
        self,
        images: list[Image.Image | str | Path],
        additional_contexts: list[str] | None = None,
    ) -> list[AnalysisResult]:
        """배치 처리"""
        if additional_contexts is None:
            additional_contexts = [None] * len(images)

        results = []
        for image, context in zip(images, additional_contexts):
            result = self.process(image, context)
            results.append(result)

        return results

    async def process_async(
        self,
        image: Image.Image | str | Path,
        additional_context: str | None = None,
    ) -> AnalysisResult:
        """비동기 처리"""
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(
                executor,
                self.process,
                image,
                additional_context,
            )
        return result

    async def process_batch_async(
        self,
        images: list[Image.Image | str | Path],
        additional_contexts: list[str] | None = None,
        max_concurrent: int = 4,
    ) -> list[AnalysisResult]:
        """비동기 배치 처리"""
        if additional_contexts is None:
            additional_contexts = [None] * len(images)

        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_with_semaphore(img, ctx):
            async with semaphore:
                return await self.process_async(img, ctx)

        tasks = [
            process_with_semaphore(img, ctx)
            for img, ctx in zip(images, additional_contexts)
        ]

        return await asyncio.gather(*tasks)

    def clear_cache(self) -> None:
        """캐시 클리어"""
        self._cache.clear()

    def get_cache_stats(self) -> dict[str, Any]:
        """캐시 통계"""
        return {
            "size": len(self._cache),
            "enabled": self.config.enable_cache,
        }


class StreamingPipeline:
    """스트리밍 추론 파이프라인

    대용량 이미지 폴더 처리를 위한 스트리밍 파이프라인
    """

    def __init__(
        self,
        config: PipelineConfig | None = None,
        callback: Callable[[AnalysisResult], None] | None = None,
    ):
        self.pipeline = InferencePipeline(config)
        self.callback = callback

    def process_directory(
        self,
        directory: str | Path,
        pattern: str = "*.png",
        recursive: bool = False,
    ) -> list[AnalysisResult]:
        """디렉토리 내 이미지 처리"""
        directory = Path(directory)

        if recursive:
            image_files = list(directory.rglob(pattern))
        else:
            image_files = list(directory.glob(pattern))

        results = []

        for image_path in image_files:
            try:
                result = self.pipeline.process(image_path)
                results.append(result)

                if self.callback:
                    self.callback(result)

            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue

        return results

    async def process_directory_async(
        self,
        directory: str | Path,
        pattern: str = "*.png",
        recursive: bool = False,
        max_concurrent: int = 4,
    ) -> list[AnalysisResult]:
        """비동기 디렉토리 처리"""
        directory = Path(directory)

        if recursive:
            image_files = list(directory.rglob(pattern))
        else:
            image_files = list(directory.glob(pattern))

        return await self.pipeline.process_batch_async(
            image_files,
            max_concurrent=max_concurrent,
        )


def create_pipeline(
    model_path: str | None = None,
    ontology_config: str | None = None,
    use_neo4j: bool = False,
    quantize: bool = False,
) -> InferencePipeline:
    """파이프라인 생성 헬퍼"""
    config = PipelineConfig(
        model_path=model_path,
        ontology_config_path=ontology_config,
        use_neo4j=use_neo4j,
        quantize=quantize,
    )

    pipeline = InferencePipeline(config)
    pipeline.initialize()

    return pipeline
