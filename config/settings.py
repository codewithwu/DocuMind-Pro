from pathlib import Path
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from typing import Optional

# 项目根目录

class Settings(BaseSettings):
    """主要配置"""

    # 路径配置
    Project_Root: Path = Path(__file__).parent.parent
    Data_Dir:Path = Project_Root / "data"
    Documents_Dir:Path = Data_Dir / "documents"
    Vector_Store_Dir:Path = Data_Dir / "vector_store"

    # 模型配置
    Ollama_Base_Url: str = Field(..., alias="OLLAMA_BASE_URL")
    Ollama_Model: str = Field(..., alias="OLLAMA_MODEL")

    # 文档处理配置
    Chunk_Size: int = 1000
    Chunk_Overlap: int = 200

    # 日志配置
    Log_Dir: Path = Project_Root / "logs"
    Log_Level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra="ignore"
    )

    @model_validator(mode='after')
    def set_paths_and_create_dirs(self):
        """模型验证后设置路径并创建目录"""
        for directory in [self.Documents_Dir, self.Vector_Store_Dir, self.Log_Dir]:
            directory.mkdir(parents=True, exist_ok=True)
        return self


settings = Settings()

print(settings.Ollama_Base_Url)
print(settings.Ollama_Model)



