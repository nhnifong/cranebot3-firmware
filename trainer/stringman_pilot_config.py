from dataclasses import dataclass, field

from lerobot.robots import RobotConfig


@RobotConfig.register_subclass("stringman")
@dataclass
class StringmanConfig(RobotConfig):
    port: str