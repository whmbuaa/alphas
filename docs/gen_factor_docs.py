"""为已计算的因子批量生成 Alphalens 文档。

目录：docs/<alpha191|alpha101>/<年份>/<alpha编号>/README.md
"""
import argparse
import inspect
import json
import os
import re
import traceback
import warnings
from io import StringIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

from alphalens import performance as perf
from alphalens import plotting
from alphalens import utils
from alphalens.utils import MaxLossExceededError, get_clean_factor_and_forward_returns

FIELDS = [
    ("OPEN", "open", "开盘价"),
    ("HIGH", "high", "最高价"),
    ("LOW", "low", "最低价"),
    ("CLOSE", "close", "收盘价"),
    ("VWAP", "vwap", "成交均价"),
    ("VOLUME", "volume", "成交量"),
    ("AMOUNT", "amount", "成交额"),
    ("RETURNS", "pctChg", "日收益率"),
]
OPS = [
    ("COVIANCE", "协方差"),
    ("COVARIANCE", "协方差"),
    ("DECAYLINEAR", "线性衰减加权"),
    ("REGBETA", "回归斜率"),
    ("TSRANK", "时序排名"),
    ("TSMAX", "窗口内最大值"),
    ("TSMIN", "窗口内最小值"),
    ("HIGHDAY", "最高价距今天数"),
    ("LOWDAY", "最低价距今天数"),
    ("SUMIF", "条件求和"),
    ("RANK", "截面排名"),
    ("DELTA", "差分"),
    ("DELAY", "滞后"),
    ("TS_ARGMAX", "窗口内最大值出现的位置"),
    ("TS_ARGMIN", "窗口内最小值出现的位置"),
    ("SIGNEDPOWER", "带符号的幂"),
    ("DECAY_LINEAR", "线性衰减加权"),
    ("INDNEUTRALIZE", "行业中性"),
    ("TS_STDDEV", "滚动标准差"),
    ("STDDEV", "滚动标准差"),
    ("CORRELATION", "滚动相关"),
    ("TS_RANK", "时序排名"),
    ("TS_MAX", "窗口内最大值"),
    ("TS_MIN", "窗口内最小值"),
    ("TS_SUM", "滚动求和"),
    ("TS_PRODUCT", "滚动乘积"),
    ("SCALE", "截面缩放"),
    ("CORR", "滚动相关"),
    ("COUNT", "条件计数"),
    ("MEAN", "滚动均值"),
    ("STD", "滚动标准差"),
    ("SMA", "指数平滑"),
    ("WMA", "加权移动平均"),
    ("SIGN", "符号"),
    ("LOG", "对数"),
    ("ABS", "绝对值"),
    ("SUM", "滚动求和"),
    ("PROD", "滚动乘积"),
    ("MAX", "较大值"),
    ("MIN", "较小值"),
]


def load_formulas(module_name):
    import importlib

    mod = importlib.import_module(module_name)
    cls = mod.Alphas191 if module_name == "alphas191" else mod.Alphas101
    out = {}
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        m = re.fullmatch(r"alpha(\d+)", name)
        if not m:
            continue
        src = inspect.getsource(method)
        formula = ""
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith("####"):
                formula = stripped.strip("#").strip()
                break
        ret = re.search(r"^\s*return .+$", src, re.M)
        out[int(m.group(1))] = {
            "formula": formula,
            "code": ret.group(0).strip() if ret else "",
        }
    class_src = inspect.getsource(cls)
    for hit in re.finditer(r"(?m)^[ \t]*#\s*Alpha#(\d+)\s+(.+)$", class_src):
        num = int(hit.group(1))
        formula = hit.group(2).strip()
        slot = out.setdefault(num, {"formula": "", "code": ""})
        if formula:
            slot["formula"] = formula
    return out


def load_prices(codes, start, end):
    frames = []
    missing = []
    usecols = ["日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "涨跌幅"]
    for code in codes:
        path = ROOT / "data_bfq" / f"{code}.csv"
        if not path.exists():
            missing.append(code)
            continue
        df = pd.read_csv(path, usecols=usecols)
        df["asset"] = code
        df = df[(df["日期"] >= start) & (df["日期"] <= end)]
        if df.empty:
            missing.append(code)
            continue
        frames.append(df)
    if not frames:
        raise RuntimeError("本地不复权数据为空")
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "涨跌幅": "pctChg",
    })
    df["date"] = pd.to_datetime(df["date"])
    df["vwap"] = df["amount"] / df["volume"] / 100
    return df, missing


def factor_universe(factor_dir):
    codes = []
    for path in sorted(factor_dir.glob("alpha*.csv")):
        cols = pd.read_csv(path, nrows=0).columns.tolist()
        for col in cols:
            if col == "date":
                continue
            code = str(col).split(".")[0].zfill(6)
            if code not in codes:
                codes.append(code)
        break
    return codes


def read_factor(path, year):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    start = pd.Timestamp(f"{year}-01-01")
    end = pd.Timestamp(f"{year + 1}-01-01")
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    df = df.melt(id_vars=["date"], var_name="asset", value_name="factor")
    df["asset"] = df["asset"].astype(str).str.split(".").str[0].str.zfill(6)
    df = df.dropna(subset=["factor"])
    df = df.set_index(["date", "asset"]).sort_index()
    return df["factor"]


def present_fields(formula):
    text = formula.upper()
    return [(col, cn) for token, col, cn in FIELDS if re.search(rf"\b{token}\b", text)]


def present_ops(formula):
    text = formula.upper()
    found = []
    for token, cn in OPS:
        if re.search(rf"\b{token}\b", text) and cn not in found:
            found.append(cn)
    return found


def windows_of(formula):
    nums = [int(x) for x in re.findall(r",\s*(\d+)\s*(?:\)|,)", formula)]
    return sorted(set(n for n in nums if 1 <= n <= 250))


def idea_paragraphs(formula):
    text = formula.upper().replace("COVIANCE", "COVARIANCE")
    fields = present_fields(formula)
    ops = present_ops(formula)
    windows = windows_of(formula)
    paras = []
    if fields:
        paras.append("公式直接用到的行情是" + "、".join(cn for _, cn in fields) + "。")
    if ops:
        paras.append("里面的运算有" + "、".join(ops) + "。")
    if windows:
        shown = "、".join(str(n) for n in windows)
        paras.append(f"出现的窗口是 {shown} 个交易日。窗口短，名次几天就会重排；窗口长，排序会稳一些。")
    raw_price = any(token in text for token in ("HIGH", "LOW", "OPEN", "CLOSE", "VWAP"))
    has_change = any(token in text for token in ("DELTA", "RETURNS")) or bool(
        re.search(r"CLOSE\s*/\s*DELAY\(\s*CLOSE", text)
    )
    if "RANK" in text and raw_price and not has_change:
        paras.append(
            "截面排名用的是价格或成交量的水平，不是区间涨幅。贵的股票、平时成交就大的股票会经常站在排名一端。"
            "如果后面还做了相关、协方差或差分，最后把股票分开的是这些名次在窗口里怎么变，不是谁更贵。"
            "名次几乎不动的股票，相关和协方差接近 0，会落在中间档。"
        )
    elif has_change:
        paras.append("公式里有差分、收益率，或收盘价除以滞后收盘价，这部分看的是涨跌，不是名义价格高低。")
    if re.search(r"\bADV\d*\b", text):
        paras.append("ADV 是一段窗口里的平均成交额。它和成交量一样，大市值股票往往长期偏大。")
    if "COVARIANCE" in text or re.search(r"\bCORR(ELATION)?\b", text):
        paras.append(
            "相关或协方差看窗口里两个序列是否同向变化。同向为正，反向为负。"
            "只在其中一天又贵又放量，占不满整个窗口，协方差不会很大。"
        )
    if "STD" in text or "标准差" in "".join(ops):
        paras.append("标准差大，表示这段行情抖得厉害；标准差小，表示这段比较平。")
    if "TS_ARGMAX" in text or "TS_ARGMIN" in text:
        paras.append("Ts_ArgMax、Ts_ArgMin 看的是窗口里最高或最低出现在第几天，不是价格本身有多高。")
    if "TSMAX" in text or "TSMIN" in text or "TS_MAX" in text or "TS_MIN" in text:
        paras.append(
            "窗口最高、最低用来给当前价格一个相对位置。收盘靠近窗口高点，通常是这段已经涨上去；靠近低点，通常是这段已经跌下去。"
        )
    if text.strip().startswith("-1") or text.strip().startswith("(-1"):
        paras.append(
            "公式以 -1 乘在外层，因子方向和括号里的结果相反。数值越小，越符合括号里那个条件；"
            "数值越大，越不符合。后面若最高档收益高于最低档，就和这个负号的方向一致。"
        )
    if re.search(r"-1\s*\*\s*RANK", text) or re.search(r"RANK\([^)]*\)\s*\)\s*$", text):
        paras.append("最外层如果是排名，因子会被压到 0 到 1，或乘上 -1 之后压到 -1 到 0，五档均值会近似均匀排开。")
    if not paras:
        paras.append("这个因子把公式里的行情收成一个可以在当天横截面上比较的数。数值越大，排进越高的档。")
    return paras


def fmt_value(col, value):
    if pd.isna(value):
        return "缺失"
    if col == "volume":
        return f"{value / 10000:.2f} 万手"
    if col == "amount":
        return f"{value / 1e8:.2f} 亿元"
    if col == "pctChg":
        return f"{value:.2f}%"
    return f"{value:.2f}"


def move_word(prev, now):
    if pd.isna(prev) or pd.isna(now) or prev == 0:
        return "缺失"
    ratio = now / prev - 1
    if ratio > 0.01:
        return "上升"
    if ratio < -0.01:
        return "下降"
    return "基本持平"


def example_block(factor_data, prices, formula, year):
    fields = present_fields(formula)
    if not fields:
        fields = [("close", "收盘价"), ("volume", "成交量")]
    windows = [n for n in windows_of(formula) if n >= 2] or windows_of(formula)
    window = min(windows) if windows else 5
    window = min(window, 20)
    dates = factor_data.index.get_level_values("date").unique().sort_values()
    target = pd.Timestamp(f"{year}-06-14")
    if target not in dates:
        later = dates[dates >= target]
        target = later[0] if len(later) else dates[len(dates) // 2]
    snap = factor_data.xs(target, level="date")
    if snap.empty:
        return ""
    qmin = snap["factor_quantile"].min()
    qmax = snap["factor_quantile"].max()
    low = snap[snap["factor_quantile"] == qmin].nsmallest(3, "factor")
    high = snap[snap["factor_quantile"] == qmax].nlargest(3, "factor")
    cal = prices["date"].drop_duplicates().sort_values().tolist()
    if target not in cal:
        return ""
    pos = cal.index(target)
    prev_day = cal[max(0, pos - window)]

    def bullets(frame, label):
        lines = [f"{target.date()} 当天，{label}：", ""]
        for asset, row in frame.iterrows():
            now = prices[(prices["date"] == target) & (prices["asset"] == asset)]
            prev = prices[(prices["date"] == prev_day) & (prices["asset"] == asset)]
            bits = []
            moves = []
            for col, cn in fields:
                if now.empty or prev.empty or col not in now.columns:
                    continue
                a = prev.iloc[0][col]
                b = now.iloc[0][col]
                bits.append(f"{cn} {fmt_value(col, a)} → {fmt_value(col, b)}")
                moves.append(move_word(a, b))
            joined = "，".join(bits)
            ret = ""
            if not now.empty and not prev.empty and prev.iloc[0]["close"]:
                chg = now.iloc[0]["close"] / prev.iloc[0]["close"] - 1
                ret = f"这 {window} 个交易日收盘涨跌 {chg * 100:.2f}%。"
            same = ""
            if len(moves) >= 2 and all(m in ("上升", "下降", "基本持平") for m in moves[:2]):
                if moves[0] == moves[1]:
                    same = f"前两个输入{moves[0]}，方向相同。"
                else:
                    same = f"前两个输入一个{moves[0]}、一个{moves[1]}。"
            lines.append(
                f"- `{asset}`，因子 {row['factor']:.3f}，第 {int(row['factor_quantile'])} 档。"
                + (joined + "。" if bits else "")
                + ret
                + same
            )
        lines.append("")
        return lines

    lines = [
        f"下面用 {target.date()}、因子处在两端的股票看公式里的行情。价格是不复权日线，除权造成的缺口也会算进价格变化。对比区间是往前 {window} 个交易日（{prev_day.date()} 到 {target.date()}）。",
        "",
    ]
    lines += bullets(low, f"因子最低的一档（第 {int(qmin)} 档）")
    lines += bullets(high, f"因子最高的一档（第 {int(qmax)} 档）")
    lines.append("最低档是公式算出来最小的那一批，最高档是最大的那一批。两端都可能既有高价股也有低价股，价格水平本身不决定档位。")
    return "\n".join(lines)


def period_label(period):
    text = str(period)
    if text.endswith("D"):
        return f"{text[:-1]} 日"
    return text


def fnum(value, digits=3):
    if pd.isna(value):
        return "—"
    return f"{value:.{digits}f}"


def bps(value):
    if pd.isna(value):
        return "—"
    return f"{value * 10000:.1f}"


def strength(ic):
    if pd.isna(ic):
        return "算不出来"
    mag = abs(ic)
    if mag < 0.02:
        word = "很弱"
    elif mag < 0.05:
        word = "偏弱"
    else:
        word = "比较明显"
    direction = "正" if ic > 0 else "负" if ic < 0 else "接近零的"
    return f"{word}的{direction}相关"


def sign_paragraph(factor_data, beta, nav_end):
    values = factor_data["factor"]
    neg = float((values < 0).mean())
    pos = float((values > 0).mean())
    if neg > 0.95:
        return (
            f"因子值几乎全是负数（占比 {neg:.0%}）。`long_short=False` 时，权重等于因子值除以绝对值之和，符号保留，"
            f"于是组合里每只股票的权重都是负的，相当于按因子强弱做空一篮子股票。beta {fnum(beta)}。"
            f"这一年股票如果总体上涨，空头组合净值会往下走，期末净值 {fnum(nav_end)}。这不是因子把方向预测反了。"
        )
    if pos > 0.95:
        return (
            f"因子值几乎全是正数（占比 {pos:.0%}）。权重也都是正的，组合是按因子大小做多一篮子股票，会跟着市场同向走。"
            f"beta {fnum(beta)}，期末净值 {fnum(nav_end)}。净值上升主要反映这一年股票在涨，还要看下面各档是否拉开。"
        )
    return (
        f"因子有正有负（正值占比 {pos:.0%}，负值占比 {neg:.0%}）。权重大于 0 的是因子为正的股票，小于 0 的是因子为负的股票。"
        f"beta {fnum(beta)}，因子加权组合期末净值 {fnum(nav_end)}。这条净值要和下面按档画的累计收益分开看。"
    )


def monotonic_sentence(means):
    ordered = [means[q] for q in sorted(means)]
    labels = list(sorted(means))
    if len(ordered) < 2 or any(pd.isna(x) for x in ordered):
        return "分档收益不完整。"
    increasing = all(ordered[i] <= ordered[i + 1] + 1e-12 for i in range(len(ordered) - 1))
    decreasing = all(ordered[i] >= ordered[i + 1] - 1e-12 for i in range(len(ordered) - 1))
    best = max(means, key=lambda q: means[q])
    worst = min(means, key=lambda q: means[q])
    if increasing:
        return f"1 日收益从低档到高档递增，最高是第 {int(best)} 档（{bps(means[best])} 个基点）。和“因子越大、后面越好”一致。"
    if decreasing:
        return f"1 日收益从低档到高档递减，最好的是第 {int(worst)} 档（{bps(means[worst])} 个基点）。因子越大，后面这天反而越差。"
    return (
        f"1 日收益不是从第 {int(labels[0])} 档到第 {int(labels[-1])} 档单调变化。"
        f"最好的是第 {int(best)} 档，约 {bps(means[best])} 个基点；最差的是第 {int(worst)} 档，约 {bps(means[worst])} 个基点。"
    )


def hold_sentence(ac):
    a1 = ac.get("1D", np.nan)
    a5 = ac.get("5D", np.nan)
    a10 = ac.get("10D", np.nan)
    if pd.isna(a1):
        return "自相关没有算出来。"
    if a1 >= 0.5 and (pd.isna(a5) or a5 < 0.3):
        return f"隔 1 天的平均自相关是 {a1:.3f}，排序还在；隔 5 天 {fnum(a5)}，隔 10 天 {fnum(a10)}。信号主要活在几天里，持有期按天更合适。"
    if not pd.isna(a5) and a5 >= 0.5:
        return f"隔 5 天自相关还有 {a5:.3f}，隔 10 天 {fnum(a10)}。排序可以维持大约一周，不必每天都换。"
    return f"隔 1 天自相关 {a1:.3f}，隔 5 天 {fnum(a5)}，隔 10 天 {fnum(a10)}。名次换得比较快。"


def save_ax(ax, path):
    ax.figure.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(ax.figure)


def new_ax():
    fig, ax = plt.subplots(1, 1, figsize=(18, 6))
    return fig, ax


def render_figures(factor_data, ctx, fig_dir):
    fig_dir.mkdir(parents=True, exist_ok=True)
    periods = list(ctx["periods"])

    _, ax = new_ax()
    plotting.plot_quantile_returns_bar(ctx["mean_quant_rateret"], by_group=False, ylim_percentiles=None, ax=ax)
    save_ax(ax, fig_dir / "01-mean-return-by-quantile.png")

    _, ax = new_ax()
    plotting.plot_quantile_returns_violin(ctx["mean_quant_rateret_bydate"], ylim_percentiles=(1, 99), ax=ax)
    save_ax(ax, fig_dir / "02-return-violin.png")

    if "1D" in ctx["factor_returns"]:
        _, ax = new_ax()
        plotting.plot_cumulative_returns(
            ctx["factor_returns"]["1D"],
            period="1D",
            title="Factor Weighted Portfolio Cumulative Return (1D Period)",
            ax=ax,
        )
        save_ax(ax, fig_dir / "03-portfolio-cumulative.png")

        _, ax = new_ax()
        plotting.plot_cumulative_returns_by_quantile(ctx["mean_quant_ret_bydate"]["1D"], period="1D", ax=ax)
        save_ax(ax, fig_dir / "04-cumulative-by-quantile.png")

    for period in periods:
        _, ax = new_ax()
        plotting.plot_mean_quantile_returns_spread_time_series(
            ctx["mean_ret_spread_quant"][period],
            std_err=ctx["std_spread_quant"][period],
            bandwidth=0.5,
            ax=ax,
        )
        save_ax(ax, fig_dir / f"05-spread-{period}.png")

    ic = ctx["ic"]
    for period in ic.columns:
        _, ax = new_ax()
        plotting.plot_ic_ts(ic[[period]], ax=[ax])
        save_ax(ax, fig_dir / f"06-ic-ts-{period}.png")
        _, ax = new_ax()
        plotting.plot_ic_hist(ic[[period]], ax=[ax])
        save_ax(ax, fig_dir / f"07-ic-hist-{period}.png")
        _, ax = new_ax()
        plotting.plot_ic_qq(ic[[period]], ax=[ax])
        save_ax(ax, fig_dir / f"08-ic-qq-{period}.png")

    monthly = ctx["mean_monthly_ic"]
    axes = []
    figs = []
    for _ in monthly.columns:
        fig, ax = plt.subplots(1, 1, figsize=(8, 3))
        figs.append(fig)
        axes.append(ax)
    plotting.plot_monthly_ic_heatmap(monthly, ax=axes)
    for fig, period in zip(figs, monthly.columns):
        fig.savefig(fig_dir / f"09-monthly-ic-{period}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)

    for period, frame in ctx["quantile_turnover"].items():
        _, ax = new_ax()
        plotting.plot_top_bottom_quantile_turnover(frame, period=period, ax=ax)
        save_ax(ax, fig_dir / f"10-turnover-{period}d.png")
    for period in ctx["autocorr"].columns:
        _, ax = new_ax()
        plotting.plot_factor_rank_auto_correlation(ctx["autocorr"][period], period=int(period[:-1]), ax=ax)
        save_ax(ax, fig_dir / f"11-autocorr-{period.lower()}.png")


def analyze(factor_data):
    long_short = False
    factor_returns = perf.factor_returns(factor_data, long_short, False)
    mean_quant_ret, _ = perf.mean_return_by_quantile(
        factor_data, by_group=False, demeaned=long_short, group_adjust=False
    )
    base = mean_quant_ret.columns[0]
    mean_quant_rateret = mean_quant_ret.apply(utils.rate_of_return, axis=0, base_period=base)
    mean_quant_ret_bydate, std_quant_daily = perf.mean_return_by_quantile(
        factor_data, by_date=True, by_group=False, demeaned=long_short, group_adjust=False
    )
    mean_quant_rateret_bydate = mean_quant_ret_bydate.apply(
        utils.rate_of_return, axis=0, base_period=mean_quant_ret_bydate.columns[0]
    )
    compstd = std_quant_daily.apply(utils.std_conversion, axis=0, base_period=std_quant_daily.columns[0])
    alpha_beta = perf.factor_alpha_beta(factor_data, factor_returns, long_short, False)
    qmax = factor_data["factor_quantile"].max()
    qmin = factor_data["factor_quantile"].min()
    mean_ret_spread_quant, std_spread_quant = perf.compute_mean_returns_spread(
        mean_quant_rateret_bydate, qmax, qmin, std_err=compstd
    )
    ic = perf.factor_information_coefficient(factor_data, False)
    monthly = perf.mean_information_coefficient(factor_data, group_adjust=False, by_group=False, by_time="ME")
    periods = utils.get_forward_returns_columns(factor_data.columns, require_exact_day_multiple=True).tolist()
    turnover_periods = utils.timedelta_strings_to_integers(periods)
    quantile_factor = factor_data["factor_quantile"]
    quantile_turnover = {
        p: pd.concat(
            [perf.quantile_turnover(quantile_factor, q, p) for q in sorted(quantile_factor.unique())],
            axis=1,
        )
        for p in turnover_periods
    }
    autocorr = pd.concat(
        [perf.factor_rank_autocorrelation(factor_data, period) for period in turnover_periods],
        axis=1,
    )
    autocorr.columns = [f"{p}D" for p in turnover_periods]
    return {
        "factor_returns": factor_returns,
        "mean_quant_rateret": mean_quant_rateret,
        "mean_quant_ret_bydate": mean_quant_ret_bydate,
        "mean_quant_rateret_bydate": mean_quant_rateret_bydate,
        "alpha_beta": alpha_beta,
        "mean_ret_spread_quant": mean_ret_spread_quant,
        "std_spread_quant": std_spread_quant,
        "ic": ic,
        "mean_monthly_ic": monthly,
        "periods": periods,
        "quantile_turnover": quantile_turnover,
        "autocorr": autocorr,
        "qmin": qmin,
        "qmax": qmax,
    }


def markdown_doc(num, meta, formula, year, factor_data, ctx, prices, drop_line, n_universe, n_missing):
    stats = factor_data.groupby("factor_quantile")["factor"].agg(["min", "max", "mean", "std", "count"])
    periods = ctx["periods"]
    ic = ctx["ic"]
    ab = ctx["alpha_beta"]
    ret = ctx["mean_quant_rateret"]
    name = f"alpha{num:03d}"
    library = "国泰君安 191 " if meta["library"] == "alpha191" else "WorldQuant 101 "
    title_formula = formula or "（源码里没有单独的公式注释）"
    ideas = idea_paragraphs(formula or ctx.get("code", ""))
    qmeans = {q: ret.loc[q, "1D"] for q in ret.index} if "1D" in ret.columns else {}
    spread_mean = {}
    spread_pos = {}
    for period in periods:
        series = ctx["mean_ret_spread_quant"][period].dropna()
        spread_mean[period] = float(series.mean()) if len(series) else np.nan
        spread_pos[period] = float((series > 0).mean()) if len(series) else np.nan
    ic_mean = {c: float(ic[c].mean()) for c in ic.columns}
    ic_std = {c: float(ic[c].std()) for c in ic.columns}
    ra = {c: ic_mean[c] / ic_std[c] if ic_std[c] else np.nan for c in ic.columns}
    monthly = ctx["mean_monthly_ic"]
    month_pos = {}
    for c in monthly.columns:
        month_pos[c] = int((monthly[c] > 0).sum())
    ac = {c: float(ctx["autocorr"][c].mean()) for c in ctx["autocorr"].columns}
    turn = {}
    for period, frame in ctx["quantile_turnover"].items():
        key = f"{period}D"
        turn[f"{period}D"] = (
            float(frame[frame.columns.max()].mean()),
            float(frame[frame.columns.min()].mean()),
        )
    nav = perf.cumulative_returns(ctx["factor_returns"]["1D"]) if "1D" in ctx["factor_returns"] else None
    nav_end = float(nav.iloc[-1]) if nav is not None and len(nav) else np.nan
    q_end = {}
    if "1D" in ctx["mean_quant_ret_bydate"]:
        wide = ctx["mean_quant_ret_bydate"]["1D"].unstack("factor_quantile")
        cum = wide.apply(perf.cumulative_returns)
        q_end = {q: float(cum[q].iloc[-1]) for q in cum.columns}
    best_end = max(q_end, key=q_end.get) if q_end else None
    worst_end = min(q_end, key=q_end.get) if q_end else None
    daily = ctx["mean_quant_rateret_bydate"]["1D"] if "1D" in ctx["mean_quant_rateret_bydate"] else None
    within = float(daily.groupby(level="factor_quantile").std().median()) if daily is not None else np.nan
    beta = ab.loc["beta", "1D"] if "1D" in ab.columns else np.nan

    def row(values, digits=3):
        return " | ".join(fnum(values.get(p, np.nan), digits) for p in periods)

    def row_bps(values):
        return " | ".join(bps(values.get(p, np.nan)) for p in periods)

    lines = [
        f"# {meta['class_name']} {name}，{year} 年 Alphalens 结果",
        "",
        f"因子是{library}公式里的第 {num} 号：",
        "",
        f"`{title_formula}`",
        "",
    ]
    lines.extend(ideas)
    lines.append("")
    missing_txt = f"其中 {n_missing} 只在本地不复权数据里没有对上。" if n_missing else ""
    last_day = factor_data.index.get_level_values("date").max()
    partial = ""
    if last_day < pd.Timestamp(f"{year}-12-20"):
        partial = f"本地行情到 {last_day.date()} 为止，这一年不是完整的自然年。"
    lines.append(
        f"样本是 {year} 年沪深 300。成分股用因子文件里的 {n_universe} 只代码，和当时计算因子时写入 CSV 的名单一致。"
        f"{missing_txt}价格是本地不复权日线。{partial}"
        f"Alphalens 默认看未来 1、5、10 个交易日的收益，分成 {int(factor_data['factor_quantile'].nunique())} 档。"
        "`long_short=False`，收益没有扣掉当天全体股票的平均涨跌。"
    )
    lines.append("")
    lines.append(drop_line or "清洗记录没有打印出来。")
    lines.append("")
    lines.append("## 因子详解")
    lines.append("")
    detail = example_block(factor_data, prices, formula or "", year)
    if detail:
        lines.append(detail)
        lines.append("")
    lines += [
        "## 分档统计",
        "",
        "打印标题：**Quantiles Statistics**（分位数统计）",
        "",
        "这是一张表，不是图。档位数字越小，因子越低。",
        "",
        "| 档 | 因子最小值 | 因子最大值 | 均值 | 标准差 | 条数 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for q, rowv in stats.iterrows():
        lines.append(
            f"| {int(q)} | {rowv['min']:.3f} | {rowv['max']:.3f} | {rowv['mean']:.3f} | {rowv['std']:.3f} | {int(rowv['count'])} |"
        )
    gap = stats["mean"].max() - stats["mean"].min() if len(stats) else np.nan
    lines.append("")
    if len(stats) >= 2 and gap < 0.05:
        lines.append("各档均值挤在一起，因子在横截面上拉不开，分档的意义很弱。")
    else:
        lines.append("各档均值分开，说明当天的因子没有都堆在同一个数上。排名类因子通常会切得很均匀。")
    lines.append("")
    lines += [
        "## 收益",
        "",
        "打印标题：**Returns Analysis**（收益分析）",
        "",
        "|  | " + " | ".join(period_label(p) for p in periods) + " |",
        "| --- | " + " | ".join("---:" for _ in periods) + " |",
        f"| Ann. alpha 年化超额 | {row(ab.loc['Ann. alpha'])} |",
        f"| beta 相对全体股票平均收益的斜率 | {row(ab.loc['beta'])} |",
        f"| 最高档平均收益（基点） | {row_bps({p: ret.loc[ctx['qmax'], p] for p in periods})} |",
        f"| 最低档平均收益（基点） | {row_bps({p: ret.loc[ctx['qmin'], p] for p in periods})} |",
        "",
        sign_paragraph(factor_data, beta, nav_end),
        "",
        "### Mean Period Wise Return By Factor Quantile",
        "",
        "按因子分位数的平均收益。横轴是档位，蓝、橙、绿分别是未来 1 日、5 日、10 日，纵轴单位是基点。",
        "",
        "![Mean Period Wise Return By Factor Quantile](figures/01-mean-return-by-quantile.png)",
        "",
        monotonic_sentence(qmeans),
        f"最高档减最低档，1 日平均只差 {bps(spread_mean.get('1D', np.nan))} 个基点。"
        f"各档每天收益的波动中位数约 {bps(within)} 个基点，比档与档的平均差距大得多。单看一天、一只股票，分档解释不了大部分涨跌。",
        "",
        "### Period Wise Return By Factor Quantile",
        "",
        "按因子分位数的收益分布。小提琴图看每天收益散得有多开，中间的横线是四分位数。",
        "",
        "![Period Wise Return By Factor Quantile](figures/02-return-violin.png)",
        "",
        "每一档的小提琴都拉得很长。档与档的平均值只差几个到几十个基点，落在每天自己的波动里面。",
        "",
        "### Factor Weighted Portfolio Cumulative Return (1D Period)",
        "",
        "因子加权组合的累计收益，持有期 1 日。起点是 1。",
        "",
        "![Factor Weighted Portfolio Cumulative Return (1D Period)](figures/03-portfolio-cumulative.png)",
        "",
        f"曲线期末在 {fnum(nav_end)}。读这条线之前先看因子符号：全负就是空头组合，全正就是多头组合，有正有负才是多空都有。",
        "",
        "### Cumulative Return by Quantile (1D Period Forward Return)",
        "",
        "按分位数的累计收益，未来 1 日。纵轴是对数坐标下的累计净值。颜色从高档到低档排列。",
        "",
        "![Cumulative Return by Quantile (1D Period Forward Return)](figures/04-cumulative-by-quantile.png)",
        "",
    ]
    if best_end is not None:
        end_txt = "、".join(f"第 {int(q)} 档 {q_end[q]:.3f}" for q in sorted(q_end))
        lines.append(
            f"年末各档净值：{end_txt}。最高的是第 {int(best_end)} 档，最低的是第 {int(worst_end)} 档。"
            "这里没有扣市场，所以上涨年份里这几条线往往一起往上。"
        )
    lines.append("")
    lines += [
        "### Top Minus Bottom Quantile Mean Return (... Period Forward Return)",
        "",
        "最高档减最低档的平均收益。这里的最高档、最低档是因子最大的一档减最小的一档，不是收益最好的那一档减最差的那一档。绿线是当天的差，橙线是 22 个交易日的移动平均，蓝色带子是波动范围。",
        "",
    ]
    for period in periods:
        lines.append(f"![Top Minus Bottom Quantile Mean Return ({period} Period Forward Return)](figures/05-spread-{period}.png)")
        lines.append("")
    pos1 = spread_pos.get("1D", np.nan)
    lines.append(
        f"1 日的差有 {pos1:.0%} 的交易日大于 0，全年平均 {bps(spread_mean.get('1D', np.nan))} 个基点。"
        "移动平均如果经常穿过 0，最高档相对最低档就没有稳定的正超额。"
    )
    lines.append("")
    lines += [
        "## 信息系数",
        "",
        "打印标题：**Information Analysis**（信息分析）",
        "",
        "IC 是当天因子值和随后收益的秩相关，每天一个数。大于 0 表示因子大的股票后面更容易涨。",
        "",
        "|  | " + " | ".join(period_label(p) for p in ic.columns) + " |",
        "| --- | " + " | ".join("---:" for _ in ic.columns) + " |",
        f"| IC Mean | {row(ic_mean)} |",
        f"| IC Std. | {row(ic_std)} |",
        f"| Risk-Adjusted IC | {row(ra)} |",
        "",
        f"1 日 IC 均值 {fnum(ic_mean.get('1D', np.nan))}，是{strength(ic_mean.get('1D', np.nan))}。"
        f"标准差约 {fnum(ic_std.get('1D', np.nan))}，单日噪声比均值大。风险调整后的 IC（均值除以标准差）1 日是 {fnum(ra.get('1D', np.nan))}。",
        "",
        "### 1D / 5D / 10D Period Forward Return Information Coefficient (IC)",
        "",
        "对应持有期的信息系数。蓝线是每天的 IC，绿线是一个月移动平均。",
        "",
    ]
    for period in ic.columns:
        lines.append(f"![{period} Period Forward Return Information Coefficient (IC)](figures/06-ic-ts-{period}.png)")
        lines.append("")
    lines.append("蓝线摆动大、绿线贴着 0，表示预测方向不稳定，不能用全年一个均值代表每一天。")
    lines.append("")
    lines += ["### 1D / 5D / 10D Period IC", "", "IC 的直方图。", ""]
    for period in ic.columns:
        lines.append(f"![{period} Period IC](figures/07-ic-hist-{period}.png)")
        lines.append("")
    lines.append("分布若以 0 为中心、只是略偏一侧，就和 IC 均值很小一致。")
    lines.append("")
    lines += [
        "### 1D / 5D / 10D Period IC Normal Dist. Q-Q",
        "",
        "IC 和正态分布的 Q-Q 图。点贴着斜线，说明接近正态。纵轴 Observed Quantile 是实际分位点，横轴 Normal Distribution Quantile 是正态分布分位点。",
        "",
    ]
    for period in ic.columns:
        lines.append(f"![{period} Period IC Normal Dist. Q-Q](figures/08-ic-qq-{period}.png)")
        lines.append("")
    lines.append("中间贴着直线、两头偶尔翘起，只说明个别交易日的 IC 特别大或特别小。")
    lines.append("")
    lines += [
        "### Monthly Mean 1D / 5D / 10D Period IC",
        "",
        "月均 IC。每一格是某个月的平均信息系数，暖色为正，冷色为负。",
        "",
    ]
    for period in monthly.columns:
        n = month_pos.get(period, 0)
        lines.append(f"![Monthly Mean {period} Period IC](figures/09-monthly-ic-{period}.png)")
        lines.append("")
        lines.append(f"{period} 有 {n} 个月的平均 IC 为正，其余为负或接近 0。")
        lines.append("")
    lines += [
        "## 换手",
        "",
        "打印标题：**Turnover Analysis**（换手分析）",
        "",
        "### 1D / 5D / 10D Period Top and Bottom Quantile Turnover",
        "",
        "最高档和最低档的换手。纵轴 Proportion Of Names New To Quantile 是新进入这一档的股票比例。蓝线是因子最高档，橙线是因子最低档。",
        "",
    ]
    for key, (top, bot) in turn.items():
        lines.append(f"![{key} Period Top and Bottom Quantile Turnover](figures/10-turnover-{key.lower()}.png)")
        lines.append("")
        lines.append(f"隔 {key[:-1]} 天，最高档平均有 {top:.0%} 的名字是新的，最低档平均 {bot:.0%}。")
        lines.append("")
    lines += [
        "### 1D / 5D / 10D Period Factor Rank Autocorrelation",
        "",
        "因子排名自相关。纵轴 Autocorrelation Coefficient 是相关系数。图内白框里的 Mean 是全年平均。",
        "",
    ]
    for period in ctx["autocorr"].columns:
        lines.append(f"![{period} Period Factor Rank Autocorrelation](figures/11-autocorr-{period.lower()}.png)")
        lines.append("")
    lines.append(hold_sentence(ac))
    lines.append("")
    lines += [
        "## 怎么读这一份结果",
        "",
        f"{year} 年，{name} 的 1 日 IC 是{strength(ic_mean.get('1D', np.nan))}（均值 {fnum(ic_mean.get('1D', np.nan))}）。"
        + monotonic_sentence(qmeans),
        "",
        sign_paragraph(factor_data, beta, nav_end),
        "",
        hold_sentence(ac)
        + f" 1 日最高档减最低档平均 {bps(spread_mean.get('1D', np.nan))} 个基点，和各档内部每天几十个基点的波动比，这个差距不大。"
        "换手高的时候，这点差距覆盖不了来回交易。这只说明这一年、这一批沪深 300 上的截面表现，不能推广到别的年份或其他因子。",
        "",
    ]
    return "\n".join(lines), {
        "ic_1d": ic_mean.get("1D", np.nan),
        "ic_5d": ic_mean.get("5D", np.nan),
        "ic_10d": ic_mean.get("10D", np.nan),
        "alpha_1d": float(ab.loc["Ann. alpha", "1D"]) if "1D" in ab.columns else np.nan,
        "beta_1d": float(beta) if not pd.isna(beta) else np.nan,
        "spread_1d_bps": None if pd.isna(spread_mean.get("1D", np.nan)) else spread_mean["1D"] * 10000,
        "nav_end": nav_end,
        "ac_1d": ac.get("1D", np.nan),
    }


def failure_doc(num, meta, formula, year, message):
    library = "国泰君安 191 " if meta["library"] == "alpha191" else "WorldQuant 101 "
    text = "\n".join([
        f"# {meta['class_name']} alpha{num:03d}，{year} 年",
        "",
        f"因子是{library}公式里的第 {num} 号：",
        "",
        f"`{formula or '（没有公式注释）'}`",
        "",
        "这一份没有生成图表。Alphalens 在分档或计算收益时失败，常见原因是因子几乎是常数、有效样本太少，或和行情对不齐。",
        "",
        "```",
        message.strip()[:2000],
        "```",
        "",
    ])
    return text


def write_index(out_dir, year, library, records, missing_nums):
    ok = [r for r in records if r.get("status") == "ok"]
    bad = [r for r in records if r.get("status") != "ok"]
    ok_sorted = sorted(ok, key=lambda r: r["num"])
    lines = [
        f"# {library} {year} 年因子文档",
        "",
        "每份文档在对应因子目录的 `README.md`，图在同目录的 `figures/`。请用「在侧边打开预览」（`Ctrl+K` 然后 `V`）看图。右上角 Preview 不显示本地图片。",
        "",
        f"已生成 {len(ok_sorted)} 份，失败 {len(bad)} 份。样本是 {year} 年沪深 300、本地不复权日线，未来 1/5/10 日，`long_short=False`。能切成 5 个等人数档的按 5 档；取值并列、切不开的按实际取值分档。",
        "",
        "| 因子 | 1日 IC | 5日 IC | 10日 IC | 1日年化 alpha | 1日 beta | 1日高低档差(基点) | 1日自相关 | 期末净值 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in ok_sorted:
        s = r["stats"]
        lines.append(
            f"| [{r['name']}]({r['name']}/README.md) | {fnum(s['ic_1d'])} | {fnum(s['ic_5d'])} | {fnum(s['ic_10d'])} | "
            f"{fnum(s['alpha_1d'])} | {fnum(s['beta_1d'])} | {fnum(s['spread_1d_bps'], 1)} | {fnum(s['ac_1d'])} | {fnum(s['nav_end'])} |"
        )
    if bad:
        lines += ["", "## 没能出图", ""]
        for r in bad:
            lines.append(f"- [{r['name']}]({r['name']}/README.md)：{r.get('error', '失败')}")
    if missing_nums:
        shown = "、".join(f"{n:03d}" for n in missing_nums)
        lines += ["", "## 因子文件不存在", "", shown, ""]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def process_one(path, meta, formulas, prices, close, year, out_dir, n_missing):
    num = int(re.search(r"(\d+)", path.stem).group(1))
    formula = formulas.get(num, {}).get("formula", "")
    dest = out_dir / f"alpha{num:03d}"
    fig_dir = dest / "figures"
    try:
        factor = read_factor(path, year)
        import contextlib

        def _clean(series, quantiles):
            buf = StringIO()
            with contextlib.redirect_stdout(buf):
                data = get_clean_factor_and_forward_returns(
                    series, close, quantiles=quantiles, max_loss=0.95
                )
            drop_line = ""
            for line in buf.getvalue().splitlines():
                if line.startswith("Dropped"):
                    drop_line = "清洗结果：" + line
            return data, drop_line

        try:
            factor_data, drop_line = _clean(factor, 5)
        except MaxLossExceededError:
            noise = factor.groupby(level="date").rank(method="first") * 1e-6
            factor_data, drop_line = _clean(factor + noise, 5)

            def _label(values):
                order = {v: i + 1 for i, v in enumerate(sorted(values.dropna().unique()))}
                return values.map(order)

            labels = factor.groupby(level="date", group_keys=False).apply(_label)
            if labels.index.nlevels > factor.index.nlevels:
                labels = labels.droplevel(list(range(labels.index.nlevels - factor.index.nlevels)))
            factor_data = factor_data.copy()
            factor_data["factor"] = factor.reindex(factor_data.index)
            factor_data["factor_quantile"] = labels.reindex(factor_data.index)
            factor_data = factor_data.dropna(subset=["factor_quantile"])
            factor_data["factor_quantile"] = factor_data["factor_quantile"].astype(int)
            drop_line += (
                " 等人数的 5 档切不开，因为同一天有大量股票取值并列。"
                "下面按当天实际取值分档，不同的数各成一档。"
            )
        ctx = analyze(factor_data)
        render_figures(factor_data, ctx, fig_dir)
        text, stats = markdown_doc(
            num, meta, formula, year, factor_data, ctx, prices, drop_line, close.shape[1], n_missing
        )
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "README.md").write_text(text, encoding="utf-8")
        return {"num": num, "name": f"alpha{num:03d}", "status": "ok", "stats": stats}
    except Exception as exc:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "README.md").write_text(
            failure_doc(num, meta, formula, year, traceback.format_exc()),
            encoding="utf-8",
        )
        return {"num": num, "name": f"alpha{num:03d}", "status": "fail", "error": f"{type(exc).__name__}: {exc}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", default="alpha191", choices=["alpha191", "alpha101"])
    parser.add_argument("--year", type=int, default=2019)
    parser.add_argument("--only", default="", help="逗号分隔的因子编号，空则全部")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    meta = {
        "alpha191": {
            "library": "alpha191",
            "class_name": "Alphas191",
            "module": "alphas191",
            "factor_dir": ROOT / "alphas" / "Alphas191" / str(args.year),
            "expected": 191,
        },
        "alpha101": {
            "library": "alpha101",
            "class_name": "Alphas101",
            "module": "alphas101",
            "factor_dir": ROOT / "alphas" / "Alphas101" / str(args.year),
            "expected": 101,
        },
    }[args.library]
    factor_dir = meta["factor_dir"]
    files = sorted(factor_dir.glob("alpha*.csv"))
    if args.only:
        want = {int(x) for x in args.only.split(",") if x.strip()}
        files = [p for p in files if int(re.search(r"(\d+)", p.stem).group(1)) in want]
    out_dir = ROOT / "docs" / args.library / str(args.year)
    out_dir.mkdir(parents=True, exist_ok=True)
    record_path = out_dir / "_results.jsonl"
    done = {}
    if record_path.exists():
        for line in record_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["name"]] = rec
    print(f"load formulas and prices, factors={len(files)}", flush=True)
    formulas = load_formulas(meta["module"])
    codes = factor_universe(factor_dir)
    prices, missing = load_prices(codes, f"{args.year}-01-01", f"{args.year + 1}-01-01")
    close = prices.pivot(index="date", columns="asset", values="close").sort_index()
    print(f"prices {close.shape}, missing {len(missing)}", flush=True)
    total = len(files)
    for i, path in enumerate(files, 1):
        num = int(re.search(r"(\d+)", path.stem).group(1))
        name = f"alpha{num:03d}"
        dest = out_dir / name / "README.md"
        if dest.exists() and not args.force and done.get(name, {}).get("status") == "ok":
            print(f"[{i}/{total}] skip {name}", flush=True)
            continue
        print(f"[{i}/{total}] {name}", flush=True)
        rec = process_one(path, meta, formulas, prices, close, args.year, out_dir, len(missing))
        done[name] = rec
        record_path.write_text(
            "\n".join(json.dumps(done[k], ensure_ascii=False) for k in sorted(done)),
            encoding="utf-8",
        )
        print(f"[{i}/{total}] {name} {rec['status']}", flush=True)
    have = {int(re.search(r"(\d+)", p.stem).group(1)) for p in factor_dir.glob("alpha*.csv")}
    missing_nums = [n for n in range(1, meta["expected"] + 1) if n not in have]
    write_index(out_dir, args.year, meta["class_name"], list(done.values()), missing_nums)
    print("index written", out_dir / "README.md", flush=True)


if __name__ == "__main__":
    main()
