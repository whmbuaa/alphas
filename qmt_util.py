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


# 取当前沪深 A 股代码。板块数据为空时先从 QMT 下载板块。
# 参数:
#     无
# 返回:
#     list[str]: 6 位股票代码，如 "000001"。
# 异常:
#     RuntimeError: QMT 未返回沪深 A 股列表。
def get_a_share_codes():
    codes = xtdata.get_stock_list_in_sector("沪深A股") or []
    if not codes:
        xtdata.download_sector_data()
        codes = xtdata.get_stock_list_in_sector("沪深A股") or []
    if not codes:
        raise RuntimeError("QMT 未返回沪深A股列表")
    return [str(code).split(".")[0].zfill(6) for code in codes]
