import numpy as np
from typing import Callable, Optional, List

def create_ashare_hard_mask(
    data, prefixes: Optional[List[str]] = None, restrict_tomorrow_open: bool = False
) -> np.ndarray:
    meets_price = data.close.rolling(60).min() > 3
    meets_volume = data.volume.rolling(20).mean() > 10_000_000
    meets_no_ban = (data.high == data.low).rolling(60).sum() == 0
    if restrict_tomorrow_open:
        meets_neg_gap = ((data.open.shift(-1) / data.close) < 0.99) & (
            (data.open.shift(-1) / data.close) > 0.97
        )
        hard_mask = (meets_price & meets_volume & meets_no_ban & meets_neg_gap).values
    else:
        hard_mask = (meets_price & meets_volume & meets_no_ban).values

    if prefixes:
        stock_mask = np.array(
            [any(col.startswith(p) for p in prefixes) for col in data.close.columns]
        )
    else:
        stock_mask = np.array(
            [col.startswith("60") or col.startswith("00") for col in data.close.columns]
        )
    hard_mask = hard_mask.copy()
    hard_mask[:, ~stock_mask] = False

    return hard_mask

HARD_MASK_FACTORIES = {
    "ashare": create_ashare_hard_mask,
}

def get_hard_mask_factory(
    market_type: str, prefix: Optional[str] = None, restrict_tomorrow_open: bool = False
) -> Callable:
    if market_type != "ashare":
        raise ValueError(
            f"Unsupported market type: {market_type}. Only 'ashare' is supported."
        )

    prefixes = prefix.split(",") if prefix else None

    def factory(data):
        return HARD_MASK_FACTORIES[market_type](
            data, prefixes=prefixes, restrict_tomorrow_open=restrict_tomorrow_open
        )

    return factory
