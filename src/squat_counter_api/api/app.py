from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

from squat_counter_api.api.errors import http_exception_handler, unhandled_exception_handler
from squat_counter_api.api.middleware import request_id_middleware
from squat_counter_api.api.schemas import HealthResponse, ModelInfoResponse, SquatPredictionResponse
from squat_counter_api.api.validation import read_valid_mp4
from squat_counter_api.core.artifacts import validate_artifacts
from squat_counter_api.core.logging import configure_logging
from squat_counter_api.core.settings import get_settings
from squat_counter_api.ml.analyzer import SquatAnalyzer

settings = get_settings()


class SquatAnalyzerService:
    def __init__(self) -> None:
        self.status = validate_artifacts(settings.model_root)
        self._analyzer: SquatAnalyzer | None = None

    @property
    def ready(self) -> bool:
        return self.status.ready

    @property
    def model_loaded(self) -> bool:
        return self._analyzer is not None and self.status.ready

    @property
    def version(self) -> str | None:
        return self.status.version

    def load(self) -> None:
        if self.status.ready and self._analyzer is None:
            self._analyzer = SquatAnalyzer(settings.model_root, self.status)

    def info(self) -> ModelInfoResponse:
        return ModelInfoResponse(
            model_loaded=self.model_loaded,
            version=self.version,
            architecture=self.status.architecture,
            training_date=self.status.metadata.get("training_date"),
            metrics=self.status.metadata.get("metrics", {}),
        )

    def analyze(self, video_bytes: bytes) -> SquatPredictionResponse:
        if not self.ready:
            raise RuntimeError(self.status.message)
        self.load()
        if self._analyzer is None:
            raise RuntimeError("Analyzer failed to load.")
        return self._analyzer.analyze(video_bytes)


@lru_cache
def get_analyzer_service() -> SquatAnalyzerService:
    return SquatAnalyzerService()


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(
        title="Squat Counter API",
        version="1.0.0",
        docs_url=None,
        redoc_url=None,
    )
    app.middleware("http")(request_id_middleware)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "x-request-id"],
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        service = get_analyzer_service()
        status_value = "ok" if service.model_loaded else "warming"
        return HealthResponse(
            status=status_value,
            model_loaded=service.model_loaded,
            model_version=service.version,
            message=service.status.message,
        )

    @app.get("/health/live", response_model=HealthResponse)
    def live() -> HealthResponse:
        service = get_analyzer_service()
        return HealthResponse(status="ok", model_loaded=service.model_loaded, model_version=service.version)

    @app.get("/health/ready", response_model=HealthResponse)
    def ready() -> HealthResponse:
        service = get_analyzer_service()
        if not service.ready:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=service.status.message)
        service.load()
        return HealthResponse(status="ok", model_loaded=service.model_loaded, model_version=service.version)

    @app.get("/model-info", response_model=ModelInfoResponse)
    def model_info() -> ModelInfoResponse:
        return get_analyzer_service().info()

    @app.post("/predict", response_model=SquatPredictionResponse)
    async def predict(file: UploadFile) -> SquatPredictionResponse:
        video_bytes = await read_valid_mp4(file, settings.max_video_bytes)
        service = get_analyzer_service()
        if not service.ready:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=service.status.message)
        return service.analyze(video_bytes)

    return app


app = create_app()
