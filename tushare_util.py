import os
from pathlib import Path

import pandas as pd
import tushare as ts

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


# 从本仓库 .env 或环境变量里取 token。
# 参数:
#     无
# 返回:
#     str: Tushare Pro token。
# 异常:
#     RuntimeError: 没有找到 TUSHARE_TOKEN。
def _token():
    if load_dotenv is not None:
        load_dotenv(Path(__file__).resolve().parent / ".env")

    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("请在环境变量或 .env 中设置 TUSHARE_TOKEN")
    return token


# 创建 Tushare Pro 接口。
# 参数:
#     无
# 返回:
#     tushare.pro.client.DataApi: Pro 接口。
# 异常:
#     RuntimeError: 没有找到 TUSHARE_TOKEN。
def _pro():
    ts.set_token(_token())
    return ts.pro_api()


# 按日期取沪深300成分股。权重按月公布，取该日及之前最近一次公布的成分。
# 参数:
#     time (str): 日期，如 "2019-01-01"。
# 返回:
#     tuple[list[str], pandas.DataFrame]: 6 位代码列表，以及含 code 列的表。
# 异常:
#     RuntimeError: 未取到成分股，或没有 TUSHARE_TOKEN。
def get_hs300_stocks(time):
    day = pd.Timestamp(time).strftime("%Y%m%d")
    start_day = (pd.Timestamp(time) - pd.Timedelta(days=60)).strftime("%Y%m%d")

    weight_df = _pro().index_weight(index_code="000300.SH", start_date=start_day, end_date=day)
    if weight_df is None or weight_df.empty:
        raise RuntimeError(f"Tushare 未返回 {day} 的沪深300成分股")

    latest_day = weight_df["trade_date"].max()
    con_codes = weight_df.loc[weight_df["trade_date"] == latest_day, "con_code"]
    symbols = list(dict.fromkeys(str(code).split(".")[0].zfill(6) for code in con_codes))
    return symbols, pd.DataFrame({"code": symbols})


# 取当前上市的沪深 A 股代码，不含北交所和 B 股。
# 参数:
#     无
# 返回:
#     list[str]: 6 位股票代码，如 "000001"。
# 异常:
#     RuntimeError: 未取到股票列表，或没有 TUSHARE_TOKEN。
def get_a_share_codes():
    pro = _pro()
    symbols = []
    # exchange 一次只能传一个市场。60/68、00/30 是 A 股，用来去掉同一市场里的 B 股。
    for exchange, prefixes in (("SSE", ("60", "68")), ("SZSE", ("00", "30"))):
        basic_df = pro.stock_basic(exchange=exchange, list_status="L", fields="symbol")
        if basic_df is None or basic_df.empty:
            continue
        symbols.extend(
            str(symbol).zfill(6)
            for symbol in basic_df["symbol"]
            if str(symbol).startswith(prefixes)
        )

    symbols = list(dict.fromkeys(symbols))
    if not symbols:
        raise RuntimeError("Tushare 未返回沪深A股列表")
    return symbols
