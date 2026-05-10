from pydantic import BaseModel, Field


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorBody


class RepMetric(BaseModel):
    rep: int
    depth_ratio: float
    torso_lean_degrees: float
    z_depth: float
    z_lean: float
    flagged: bool


class SquatPredictionResponse(BaseModel):
    model_version: str
    rep_count: int
    degradation_start_rep: int | None
    per_rep_metrics: list[RepMetric]
    annotated_frames: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
    message: str | None = None


class ModelInfoResponse(BaseModel):
    model_loaded: bool
    version: str | None = None
    architecture: str | None = None
    training_date: str | None = None
    metrics: dict[str, float | int | str] = Field(default_factory=dict)

