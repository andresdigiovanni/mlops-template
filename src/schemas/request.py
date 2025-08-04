from pydantic import BaseModel


class PredictionRequest(BaseModel):
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float
    mean_compactness: float
    mean_concavity: float
    mean_concave_points: float
    mean_symmetry: float
    mean_fractal_dimension: float
    radius_error: float
    texture_error: float
    perimeter_error: float
    area_error: float
    smoothness_error: float
    compactness_error: float
    concavity_error: float
    concave_points_error: float
    symmetry_error: float
    fractal_dimension_error: float
    worst_radius: float
    worst_texture: float
    worst_perimeter: float
    worst_area: float
    worst_smoothness: float
    worst_compactness: float
    worst_concavity: float
    worst_concave_points: float
    worst_symmetry: float
    worst_fractal_dimension: float

    class Config:
        schema_extra = {
            "example": {
                "mean_radius": 14.5,
                "mean_texture": 20.5,
                "mean_perimeter": 96.2,
                "mean_area": 644.1,
                "mean_smoothness": 0.105,
                "mean_compactness": 0.15,
                "mean_concavity": 0.12,
                "mean_concave_points": 0.075,
                "mean_symmetry": 0.18,
                "mean_fractal_dimension": 0.06,
                "radius_error": 0.55,
                "texture_error": 1.0,
                "perimeter_error": 3.3,
                "area_error": 40.0,
                "smoothness_error": 0.006,
                "compactness_error": 0.015,
                "concavity_error": 0.02,
                "concave_points_error": 0.01,
                "symmetry_error": 0.015,
                "fractal_dimension_error": 0.003,
                "worst_radius": 16.5,
                "worst_texture": 28.0,
                "worst_perimeter": 105.0,
                "worst_area": 800.0,
                "worst_smoothness": 0.13,
                "worst_compactness": 0.2,
                "worst_concavity": 0.25,
                "worst_concave_points": 0.15,
                "worst_symmetry": 0.25,
                "worst_fractal_dimension": 0.08,
            }
        }
