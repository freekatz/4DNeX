from typing import Any

from pydantic import BaseModel


class Components(BaseModel):
    pipeline_cls: Any = None

    tokenizer: Any = None
    text_encoder: Any = None
    vae: Any = None
    transformer: Any = None
    scheduler: Any = None
    image_encoder: Any = None
    image_processor: Any = None
