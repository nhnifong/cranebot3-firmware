from dataclasses import dataclass
from lerobot.teleoperators.config import TeleoperatorConfig

@TeleoperatorConfig.register_subclass("wand_config")
@dataclass
class WandConfig(TeleoperatorConfig):
    # adress to connect to observer
    grpc_addr: str