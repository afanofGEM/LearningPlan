"""供不同学习阶段复用的公共模块。"""

from .dataset import TicketDataset, build_char_to_idx, encode_text

__all__ = ["TicketDataset", "build_char_to_idx", "encode_text"]
