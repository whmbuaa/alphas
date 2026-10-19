import pandas as pd
from xtquant import xtdata


def get_hs300_stocks(time):
    """按日期取沪深300成分股。返回 (6位代码列表, 含 code 列的 DataFrame)。"""
    day = pd.Timestamp(time).strftime("%Y%m%d")
    codes = xtdata.get_stock_list_in_sector("沪深300", day) or []
    if not codes:
        xtdata.download_sector_data()
        codes = xtdata.get_stock_list_in_sector("沪深300", day) or []
    if not codes:
        raise RuntimeError(f"QMT 未返回 {day} 的沪深300成分股")
    symbols = [str(code).split(".")[0].zfill(6) for code in codes]
    return symbols, pd.DataFrame({"code": symbols})
