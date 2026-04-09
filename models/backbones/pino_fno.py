from __future__ import annotations

from models.backbones.fno import FNO2d


class PINOFNO2d(FNO2d):
    """
    PINO backbone variant reusing FNO architecture.
    Physics-informed behavior is injected via training losses.
    """

    pass
