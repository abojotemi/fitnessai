from dataclasses import dataclass
from pydantic import BaseModel, Field
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class AppConfig:
    VALID_IMAGE_TYPES: tuple = ("png", "jpg", "jpeg", "gif", "webp")
    MAX_IMAGE_SIZE: int = 10 * 1024 * 1024 
    DEFAULT_WEIGHT: float = 70.0
    DEFAULT_HEIGHT: float = 178.0
    DEFAULT_AGE: int = 30
    MIN_AGE: int = 10
    MIN_HEIGHT: float = 120.0
    MIN_WEIGHT: float = 30.0

class UserInfo(BaseModel):
    """User profile data model with validation"""
    name: str = Field(min_length=3, max_length=30)
    age: int = Field(gt=10)
    sex: str = Field(min_length=3)
    weight: float = Field(ge=30)
    height: float = Field(ge=120)
    goals: str = Field(min_length=5)
    country: str = Field(min_length=3)