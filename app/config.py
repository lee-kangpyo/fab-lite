import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    postgres_host: str = ""
    postgres_port: int = 5432
    postgres_user: str = ""
    postgres_password: str = ""
    postgres_db: str = ""

    llm_api_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""

    redis_urls: str = "redis://localhost:6379/0"

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = ""

    token_limit_per_session: int = 100000

    lock_namespace_task: int = 1
    lock_namespace_schedule: int = 2

    @property
    def database_url(self) -> str:
        if os.environ.get("TESTING"):
            return "sqlite+aiosqlite:///file::memory:?cache=shared&uri=true"
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def checkpointer_url(self) -> str:
        # LangGraph PostgresSaver는 psycopg 드라이버를 기대함
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url_list(self) -> list[str]:
        return [url.strip() for url in self.redis_urls.split(",")]

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()