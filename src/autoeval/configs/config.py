from typing import Optional
from pydantic import BaseModel


class ModelConfig(BaseModel):
    fp16: bool
    cuda: bool
    gpus: int
    seed: Optional[int]
    deterministic: bool
    top_p: float
    temperature: float
    gen_max_len: int
    batch_size: int
    model_name: str
    model_type: str  # Can be "hf", "openai", etc
    description: str
    preamble: str
    entity: str
