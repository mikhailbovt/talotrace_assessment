from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", ".env.local"), extra="ignore")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # A host's unrelated global API key must not override this project's session key.
        # The Docker image contains no env files, so injected container env remains effective.
        return init_settings, dotenv_settings, env_settings, file_secret_settings

    app_env: str = "development"
    database_url: SecretStr = SecretStr("")
    redis_url: SecretStr = SecretStr("redis://127.0.0.1:16379/0")
    materials_dir: Path = Path("../material")
    artifacts_dir: Path = Path("data/tests")
    style_anchor_path: Path = Path("assets/references/style_anchor.png")
    hand_reference_path: Path = Path("assets/references/hand_reference.png")
    openai_api_key: SecretStr = SecretStr("")
    openai_director_model: str = "gpt-5.6-luna"
    openai_image_model: str = "gpt-image-2"
    openai_embedding_model: str = "text-embedding-3-large"
    comfyui_url: str = "http://127.0.0.1:18188"
    kokoro_url: str = "http://127.0.0.1:18081"
    kokoro_voice: str = "af_heart"

    video_worker_enabled: bool = False
    video_renderer: Literal["ltx", "keyframe-composite-v1"] = "ltx"
    video_comfy_timeout_seconds: int = 21600
    video_poll_seconds: float = 5.0
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"
