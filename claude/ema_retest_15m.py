"""
Estratégia: reteste de EMA em tendência, velas de 15 min (long e short).

  Regime  : LONG  -> fecho > EMA(800) e EMA(200) > EMA(800)   (~ EMA200/EMA50 de 1h)
            SHORT -> fecho < EMA(800) e EMA(200) < EMA(800)
  Gatilho : recuo (long) / repique (short) à zona EMA20/EMA50 sem a perder no fecho,
            e vela de confirmação:
              long : fecho > EMA20 e > máximo da vela anterior
              short: fecho < EMA20 e < mínimo da vela anterior
  Stop    : entrada -/+ stop_atr * ATR(1h)   (ATR de velas de 1h JÁ FECHADAS)
  TP      : entrada +/- tp_r * risco
  Filtros : risco% >= min_risk_over_cost * custo ida-e-volta; cooldown

Tudo é causal: o sinal na vela i só usa dados até à vela i (prefixo-invariante).
Assume `Signal` (BUY/SELL/HOLD) já importado no teu código.
  BUY  = abrir long  (SL abaixo, TP acima)
  SELL = abrir short (SL acima,  TP abaixo)
"""


# ------------------------------------------------------------ auxiliares
def _ema(values, period):
    k = 2.0 / (period + 1)
    out, prev = [], None
    for v in values:
        prev = v if prev is None else prev + k * (v - prev)
        out.append(prev)
    return out


def _atr_wilder(highs, lows, closes, period):
    n = len(closes)
    out = [None] * n
    if n < period + 1:
        return out
    tr = [highs[0] - lows[0]]
    for i in range(1, n):
        tr.append(max(highs[i] - lows[i],
                      abs(highs[i] - closes[i - 1]),
                      abs(lows[i] - closes[i - 1])))
    val = sum(tr[:period]) / period
    out[period - 1] = val
    for i in range(period, n):
        val = (val * (period - 1) + tr[i]) / period
        out[i] = val
    return out


def _aggregate(highs, lows, closes, n):
    """Agrega n velas por posição (bloco k = velas n*k .. n*k+n-1).
    Só devolve blocos completos. Alinha o início dos dados a uma hora cheia
    para o bloco coincidir com as velas reais de 1h."""
    m = len(closes) // n
    h = [max(highs[k * n:(k + 1) * n]) for k in range(m)]
    l = [min(lows[k * n:(k + 1) * n]) for k in range(m)]
    c = [closes[(k + 1) * n - 1] for k in range(m)]
    return h, l, c


# -------------------------------------------------------------- método
def analyze_candles_history(
        self,
        enable_long: bool = True,
        enable_short: bool = True,
        ema_regime_fast: int = 200,
        ema_regime_slow: int = 800,
        ema_a: int = 20,
        ema_b: int = 50,
        retest_window: int = 6,       # velas onde o recuo pode ter ocorrido
        tf_factor: int = 4,           # 4 x 15m = 1h
        atr_period: int = 14,
        stop_atr: float = 2.0,
        tp_r: float = 3.0,
        cost_roundtrip_pct: float = 0.003,   # AJUSTA à tua corretora
        min_risk_over_cost: float = 3.0,
        cooldown_bars: int = 8,
):
    closes = [float(x) for x in self.closes]
    highs = [float(x) for x in self.highs]
    lows = [float(x) for x in self.lows]
    n = len(closes)

    signals = [Signal.HOLD] * n
    stops = [0.0] * n
    tps = [0.0] * n
    funnel = {f"{t}_{k}": 0 for t in ("long", "short")
              for k in ("regime", "retest", "confirmed", "cost_ok", "signals")}
    self.last_funnel = funnel

    warmup = ema_regime_slow + tf_factor * (atr_period + 2)
    if n <= warmup:
        return signals, stops, tps

    e_fast = _ema(closes, ema_regime_fast)
    e_slow = _ema(closes, ema_regime_slow)
    e_a = _ema(closes, ema_a)
    e_b = _ema(closes, ema_b)

    hh, hl, hc = _aggregate(highs, lows, closes, tf_factor)
    atr_1h = _atr_wilder(hh, hl, hc, atr_period)

    last_signal = -10 ** 9
    for i in range(warmup, n):
        c = closes[i]

        # 1) REGIME (long e short são mutuamente exclusivos)
        if c > e_slow[i] and e_fast[i] > e_slow[i]:
            side = 1
        elif c < e_slow[i] and e_fast[i] < e_slow[i]:
            side = -1
        else:
            continue
        if (side == 1 and not enable_long) or (side == -1 and not enable_short):
            continue
        tag = "long" if side == 1 else "short"
        funnel[tag + "_regime"] += 1

        # 2) RECUO / REPIQUE à zona EMA20/EMA50 sem a perder no fecho
        touched, broke = False, False
        for j in range(i - retest_window, i + 1):
            if side == 1:
                if lows[j] <= max(e_a[j], e_b[j]):
                    touched = True
                if closes[j] < min(e_a[j], e_b[j]):
                    broke = True
            else:
                if highs[j] >= min(e_a[j], e_b[j]):
                    touched = True
                if closes[j] > max(e_a[j], e_b[j]):
                    broke = True
        if not touched or broke:
            continue
        funnel[tag + "_retest"] += 1

        # 3) CONFIRMAÇÃO
        if side == 1:
            ok = c > e_a[i] and c > highs[i - 1]
        else:
            ok = c < e_a[i] and c < lows[i - 1]
        if not ok:
            continue
        funnel[tag + "_confirmed"] += 1

        # 4) ATR de 1h usando apenas blocos de 1h já fechados até à vela i
        done = (i + 1) // tf_factor
        a = atr_1h[done - 1] if done >= 1 else None
        if a is None or a <= 0:
            continue

        risk = stop_atr * a
        sl = c - side * risk
        if (risk / c) < min_risk_over_cost * cost_roundtrip_pct:
            continue
        funnel[tag + "_cost_ok"] += 1

        # 5) COOLDOWN
        if i - last_signal < cooldown_bars:
            continue
        last_signal = i

        signals[i] = Signal.BUY if side == 1 else Signal.SELL
        stops[i] = sl
        tps[i] = c + side * tp_r * risk
        funnel[tag + "_signals"] += 1

    return signals, stops, tps


# --------------------------------------------- avaliação rápida em R
def _stats(R, bars, risks):
    if not R:
        return {"n_trades": 0}
    wins = [r for r in R if r > 0]
    losses = [r for r in R if r <= 0]
    cum, peak, dd = 0.0, 0.0, 0.0
    for r in R:
        cum += r
        peak = max(peak, cum)
        dd = max(dd, peak - cum)
    return {
        "n_trades": len(R),
        "win_rate": round(len(wins) / len(R), 3),
        "expectancy_R": round(sum(R) / len(R), 3),
        "avg_win_R": round(sum(wins) / len(wins), 2) if wins else 0.0,
        "avg_loss_R": round(sum(losses) / len(losses), 2) if losses else 0.0,
        "profit_factor": round(sum(wins) / -sum(losses), 2) if losses and sum(losses) < 0 else None,
        "max_dd_R": round(dd, 2),
        "avg_risk_pct": round(100 * sum(risks) / len(risks), 2),
        "avg_bars": round(sum(bars) / len(bars), 1),
    }


def quick_eval_R(opens, highs, lows, closes, signals, stops, tps,
                 fee=0.001, slippage=0.0005, entry="next_open",
                 buy=None, sell=None, side="both", return_trades=False,
                 allow_overlap=False):
    """Simula SL/TP fixos (sem trailing e sem funding) para medir a expectância.
    Entrada na abertura da vela seguinte (realista) ou no fecho do sinal.
    SL e TP na mesma vela -> assume SL primeiro. Uma posição de cada vez.
    side: "both", "long" ou "short". Fee=0 e slippage=0 dão o resultado BRUTO.
    Devolve métricas globais + 'by_side' com long e short separados.
    allow_overlap=True avalia CADA sinal de forma independente, mesmo com outro trade
    aberto (dá mais amostras para diagnóstico, mas trades sobrepostos são correlacionados;
    não uses essas métricas como desempenho real)."""
    enum = type(signals[0])
    buy = buy if buy is not None else enum.BUY
    sell = sell if sell is not None else getattr(enum, "SELL", None)
    n = len(closes)
    trades = []          # (direção, R, barras, risco%)
    next_free = 0
    for i in range(n - 1):
        if i < next_free and not allow_overlap:
            continue
        if signals[i] == buy and side in ("both", "long"):
            d = 1
        elif sell is not None and signals[i] == sell and side in ("both", "short"):
            d = -1
        else:
            continue
        sl, tp = stops[i], tps[i]
        px = opens[i + 1] if entry == "next_open" else closes[i]
        ent = px * (1 + d * slippage)             # long paga mais, short vende mais baixo
        risk = d * (ent - sl)
        if risk <= 0 or d * (tp - ent) <= 0:
            continue
        ex, k = None, None
        for j in range(i + 1, n):
            if d == 1:
                if lows[j] <= sl:
                    ex, k = min(opens[j], sl), j
                    break
                if highs[j] >= tp:
                    ex, k = tp, j
                    break
            else:
                if highs[j] >= sl:
                    ex, k = max(opens[j], sl), j
                    break
                if lows[j] <= tp:
                    ex, k = tp, j
                    break
        if ex is None:
            ex, k = closes[-1], n - 1
        ex_eff = ex * (1 - d * slippage)
        pnl = d * (ex_eff - ent) - fee * (ent + ex_eff)
        trades.append((d, pnl / risk, k - i, risk / ent, i, pnl / ent))
        next_free = k

    def pick(dirn=None):
        t = [x for x in trades if dirn is None or x[0] == dirn]
        return _stats([x[1] for x in t], [x[2] for x in t], [x[3] for x in t])

    out = pick()
    out["by_side"] = {"long": pick(1), "short": pick(-1)}
    if return_trades:
        out["trades"] = [{"i": x[4], "dir": x[0], "R": round(x[1], 4), "ret": round(x[5], 6)} for x in trades]
    return out


# ------------------------------------- que TP é alcançável? (grelha de TP)
def tp_grid(opens, highs, lows, closes, signals, stops, ks=(1.0, 1.5, 2.0, 3.0),
            fee=0.00045, slippage=0.0002, buy=None, sell=None):
    """Para os MESMOS sinais e stops, varia só a distância do TP (em R) e mede:
      hit_rate   : fração de trades que acaba em lucro (≈ chega ao TP)
      benchmark  : 1/(1+k) = taxa de acerto esperada SEM edge, num mercado sem drift
      expectancy : expectância líquida em R (SL e TP fixos, sem trailing)
    Se hit_rate ≈ benchmark, a entrada não acrescenta nada. Se hit_rate fica
    claramente acima e a expectância líquida é > 0, há edge nessa distância de TP.
    Defaults de custos: perps da Hyperliquid, escalão base, como taker."""
    enum = type(signals[0])
    buy = buy if buy is not None else enum.BUY
    sell = sell if sell is not None else getattr(enum, "SELL", None)
    out = []
    for k in ks:
        tps = [0.0] * len(closes)
        for i, s in enumerate(signals):
            if s == buy:
                d = 1
            elif sell is not None and s == sell:
                d = -1
            else:
                continue
            tps[i] = closes[i] + d * k * abs(closes[i] - stops[i])
        r = quick_eval_R(opens, highs, lows, closes, signals, stops, tps,
                         fee=fee, slippage=slippage, buy=buy, sell=sell)
        out.append({
            "tp_R": k,
            "n_trades": r["n_trades"],
            "hit_rate": r.get("win_rate"),
            "benchmark": round(1 / (1 + k), 3),
            "expectancy_R": r.get("expectancy_R"),
            "avg_bars": r.get("avg_bars"),
        })
    return out


# ------------------------------- a tendência está a esgotar-se? (medidas)
def signal_features(opens, highs, lows, closes, signals, buy=None, sell=None,
                    ema_regime_fast=200, ema_regime_slow=800, tf_factor=4,
                    atr_period=14, look=96):
    """Para cada sinal, calcula medidas de maturidade/esgotamento da tendência
    usando SÓ dados até à vela do sinal. Todas são normalizadas pelo ATR de 1h e
    orientadas pela direção do trade (positivo = a favor da tendência).
      ext_atr             : distância do preço à EMA lenta (em ATR). Alto = esticado.
      regime_age_h        : horas desde que o regime atual começou. Alto = tendência velha.
      slope_atr           : subida da EMA rápida nas últimas `look` velas (24h por defeito).
      gap_change_atr      : variação da distância EMA rápida-EMA lenta. Negativo = a fechar.
      accel_atr           : momento atual menos o momento anterior. Negativo = a desacelerar.
      pullback_depth_atr  : profundidade do recuo face ao extremo recente. Alto = recuo fundo.
      er                  : eficiência (Kaufman) nas últimas `look` velas = |variação líquida| / caminho
                            percorrido. Perto de 0 = mercado lateral (vai e vem); perto de 1 = linha reta.
      gap_level_atr       : distância EMA rápida - EMA lenta, em ATR. Baixo = EMAs coladas (sem tendência).
      range_atr           : amplitude (máx - mín) das últimas `look` velas, em ATR. Baixo = mercado comprimido.
    Devolve {índice_do_sinal: {medidas}}."""
    enum = type(signals[0])
    buy = buy if buy is not None else enum.BUY
    sell = sell if sell is not None else getattr(enum, "SELL", None)
    n = len(closes)
    e_fast = _ema(closes, ema_regime_fast)
    e_slow = _ema(closes, ema_regime_slow)
    hh, hl, hc = _aggregate(highs, lows, closes, tf_factor)
    atr_1h = _atr_wilder(hh, hl, hc, atr_period)

    age, prev = [0] * n, 0
    for i in range(n):
        if closes[i] > e_slow[i] and e_fast[i] > e_slow[i]:
            side = 1
        elif closes[i] < e_slow[i] and e_fast[i] < e_slow[i]:
            side = -1
        else:
            side = 0
        age[i] = (age[i - 1] + 1 if (i > 0 and side == prev) else 1) if side else 0
        prev = side

    out = {}
    for i in range(2 * look, n):
        if signals[i] == buy:
            d = 1
        elif sell is not None and signals[i] == sell:
            d = -1
        else:
            continue
        done = (i + 1) // tf_factor
        a = atr_1h[done - 1] if done >= 1 else None
        if a is None or a <= 0:
            continue
        mom_now = d * (closes[i] - closes[i - look])
        mom_prev = d * (closes[i - look] - closes[i - 2 * look])
        extreme = max(highs[i - look + 1:i + 1]) if d == 1 else min(lows[i - look + 1:i + 1])
        gap_now = d * (e_fast[i] - e_slow[i]) / a
        path = sum(abs(closes[k] - closes[k - 1]) for k in range(i - look + 1, i + 1))
        gap_before = d * (e_fast[i - look] - e_slow[i - look]) / a
        out[i] = {
            "dir": d,
            "ext_atr": d * (closes[i] - e_slow[i]) / a,
            "regime_age_h": age[i] * 15 / 60,
            "slope_atr": d * (e_fast[i] - e_fast[i - look]) / a,
            "gap_change_atr": gap_now - gap_before,
            "accel_atr": (mom_now - mom_prev) / a,
            "pullback_depth_atr": d * (extreme - closes[i]) / a,
            "er": (abs(closes[i] - closes[i - look]) / path) if path > 0 else 0.0,
            "gap_level_atr": gap_now,
            "range_atr": (max(highs[i - look + 1:i + 1]) - min(lows[i - look + 1:i + 1])) / a,
        }
    return out


def feature_report(features, trades, min_per_bucket=8):
    """Junta as medidas com o resultado de cada trade (em R) e, para cada medida,
    divide os trades em tercis (baixo/médio/alto) e mostra nº, taxa de acerto e
    expectância em R. Uma medida só serve se os tercis forem claramente diferentes
    E isso se repetir noutros ativos e períodos. `trades` vem de
    quick_eval_R(..., return_trades=True)["trades"]."""
    rows = [(features[t["i"]], t["R"], t.get("pnl")) for t in trades if t["i"] in features]
    if len(rows) < 3 * min_per_bucket:
        return {"aviso": f"poucos trades ({len(rows)}): precisas de mais dados"}
    report = []
    for name in next(iter(features.values())):
        if name == "dir":
            continue
        ordered = sorted(rows, key=lambda r: r[0][name])
        k = len(ordered) // 3
        buckets = [ordered[:k], ordered[k:2 * k], ordered[2 * k:]]
        report.append({
            "medida": name,
            "tercis (baixo, médio, alto)": [
                {"n": len(b),
                 "valor_mediano": round(sorted(r[0][name] for r in b)[len(b) // 2], 2),
                 "acerto": round(sum(1 for r in b if r[1] > 0) / len(b), 3),
                 "exp_R": round(sum(r[1] for r in b) / len(b), 3),
                 **({"pnl_total": round(sum(r[2] for r in b), 1),
                     "pnl_medio": round(sum(r[2] for r in b) / len(b), 2)}
                    if all(r[2] is not None for r in b) else {})}
                for b in buckets],
        })
    return report


# ----------------------------- filtro opcional de fim de tendência (A/B)
def exhaustion_filter(opens, highs, lows, closes, signals, stops, tps,
                      buy=None, sell=None, look=96,
                      max_ext_atr=None,          # veta se o preço está esticado demais face à EMA lenta
                      max_regime_age_h=None,     # veta se o regime já dura demasiadas horas
                      min_slope_atr=None,        # veta se a EMA rápida quase não sobe/desce
                      min_gap_change_atr=None,   # veta se as EMAs não estão a abrir o suficiente
                      min_accel_atr=None,        # veta se o momento está a desacelerar demais
                      max_pullback_atr=None,     # veta se o recuo é demasiado fundo
                      min_er=None,               # veta mercado lateral (eficiência abaixo deste valor)
                      min_gap_level_atr=None):   # veta EMAs coladas (distância abaixo deste valor)
    """Pós-filtro sobre os sinais já gerados. Todos os limiares a None = sem filtro
    (devolve os sinais iguais). Usa só dados até à vela do sinal (causal).
    Devolve (signals, stops, tps, n_vetados). Define os limiares a partir do
    feature_report (tercis) num período e confirma noutro; os valores de exemplo
    não são recomendações."""
    enum = type(signals[0])
    hold = enum.HOLD
    feats = signal_features(opens, highs, lows, closes, signals,
                            buy=buy, sell=sell, look=look)
    s2, sl2, tp2 = list(signals), list(stops), list(tps)
    vetoed = 0
    for i, f in feats.items():
        veto = (
            (max_ext_atr is not None and f["ext_atr"] > max_ext_atr) or
            (max_regime_age_h is not None and f["regime_age_h"] > max_regime_age_h) or
            (min_slope_atr is not None and f["slope_atr"] < min_slope_atr) or
            (min_gap_change_atr is not None and f["gap_change_atr"] < min_gap_change_atr) or
            (min_accel_atr is not None and f["accel_atr"] < min_accel_atr) or
            (max_pullback_atr is not None and f["pullback_depth_atr"] > max_pullback_atr) or
            (min_er is not None and f["er"] < min_er) or
            (min_gap_level_atr is not None and f["gap_level_atr"] < min_gap_level_atr)
        )
        if veto:
            s2[i], sl2[i], tp2[i] = hold, 0.0, 0.0
            vetoed += 1
    return s2, sl2, tp2, vetoed


# -------------------- as piores entradas distinguem-se nalguma medida?
def worst_trades_profile(features, trades, n=10, min_trades=20):
    """Para os n piores trades (menor R), mostra em que PERCENTIL cada medida
    se situa face a todos os trades (0 = o valor mais baixo de todos, 100 = o mais alto).
    Se os piores trades se concentram num extremo de alguma medida (média do percentil
    perto de 100 ou de 0), essa medida pode servir de filtro. Se a média fica por volta
    de 50, a medida não os distingue e o filtro não os apanha.
    Referência: a média dos percentis de n trades ao acaso varia ~29/sqrt(n) pontos em
    torno de 50 (n=10: ±9). Só valores claramente fora de 35-65 merecem atenção, e
    precisam de confirmação noutros ativos e períodos.
    `trades` vem de quick_eval_R(..., return_trades=True)["trades"]."""
    rows = [(t, features[t["i"]]) for t in trades if t["i"] in features]
    if len(rows) < max(n, min_trades):
        # devolve sempre as mesmas chaves, para o código que chama não rebentar
        return {"aviso": f"poucos trades com medidas ({len(rows)}); precisas de pelo menos {max(n, min_trades)}",
                "n_trades": len(rows), "n_piores": 0,
                "percentil_medio_dos_piores": None, "detalhe": []}
    names = [k for k in rows[0][1] if k != "dir"]
    sorted_vals = {nm: sorted(f[nm] for _, f in rows) for nm in names}
    worst = sorted(rows, key=lambda r: r[0]["R"])[:n]
    detail = []
    for t, f in worst:
        pct = {nm: round(100 * sum(1 for v in sorted_vals[nm] if v <= f[nm]) / len(rows))
               for nm in names}
        detail.append({"i": t["i"], "dir": f["dir"], "R": t["R"], "pnl": t.get("pnl"), "percentis": pct})
    summary = {nm: round(sum(d["percentis"][nm] for d in detail) / len(detail)) for nm in names}
    return {"n_trades": len(rows), "n_piores": n,
            "percentil_medio_dos_piores": summary, "detalhe": detail}


# ------------------- onde muda o desempenho? (varredura de limiar por medida)
def threshold_scan(features, trades, name, thresholds, group_fn=None, tp_r=3.0, episode_fn=None):
    """Para cada limiar T compara os sinais com `name` > T (acima) e <= T (abaixo):
    n, taxa de acerto, expectância em R e z da taxa de acerto face à referência
    1/(1+tp_r) (sem edge). group_fn(chave_do_sinal) -> rótulo (ex.: o ativo ou a
    metade do período) repete o grupo 'acima' por rótulo, para veres se o efeito
    se repete. episode_fn(chave, medidas) -> id do episódio (ex.: onde o regime começou):
    acrescenta o nº de episódios independentes e a expectância média por episódio, porque
    sinais do mesmo episódio partilham o mesmo caminho de preço (amostra efetiva menor).
    `trades` vem de quick_eval_R(..., return_trades=True)['trades'].
    Cuidado: comparar muitos limiares e medidas gera falsos positivos; valida
    sempre noutros ativos/períodos antes de usares um limiar."""
    import math
    rows = [(t["i"], features[t["i"]][name], t["R"], t.get("pnl")) for t in trades if t["i"] in features]
    bench = 1 / (1 + tp_r)

    def agg(sub):
        n = len(sub)
        if n == 0:
            return {"n": 0}
        hit = sum(1 for x in sub if x[2] > 0) / n
        res = {"n": n, "acerto": round(hit, 3),
               "exp_R": round(sum(x[2] for x in sub) / n, 3),
               "z": round((hit - bench) / math.sqrt(bench * (1 - bench) / n), 1)}
        if all(x[3] is not None for x in sub):
            res["pnl_total"] = round(sum(x[3] for x in sub), 1)
            res["pnl_medio"] = round(sum(x[3] for x in sub) / n, 2)
        if episode_fn:
            eps = {}
            for x in sub:
                eps.setdefault(episode_fn(x[0], features[x[0]]), []).append(x[2])
            res["episodios"] = len(eps)
            res["exp_R_por_episodio"] = round(sum(sum(v) / len(v) for v in eps.values()) / len(eps), 3)
        return res

    out = []
    for T in thresholds:
        item = {"limiar": T,
                "acima": agg([x for x in rows if x[1] > T]),
                "abaixo": agg([x for x in rows if x[1] <= T])}
        if group_fn:
            grupos = sorted({group_fn(x[0]) for x in rows})
            item["acima_por_grupo"] = {g: agg([x for x in rows if x[1] > T and group_fn(x[0]) == g])
                                       for g in grupos}
            item["abaixo_por_grupo"] = {g: agg([x for x in rows if x[1] <= T and group_fn(x[0]) == g])
                                        for g in grupos}
        out.append(item)
    return out


# ------------- simulador das regras de saída do bot (momentum + escada + TP 5%)
LADDER_PADRAO = ((0.02, 0.015), (0.01, 0.006), (0.004, 0.001), (0.002, 0.0))


def _ladder_secure(p, tiers, wake, giveback, dynamic=True):
    """Lucro a garantir (fração do preço, ou múltiplos de R se as unidades forem R) para um PnL p.
    None = não mexer. Abaixo de `wake` não faz nada; até ao maior alvo usa a tabela; acima,
    garante (1 - giveback) do PnL (dynamic=True) ou mantém o último degrau (dynamic=False)."""
    if p < wake:
        return None
    if p <= tiers[0][0]:
        for target, secure in tiers:
            if p >= target:
                return secure
    return p * (1 - giveback) if dynamic else tiers[0][1]


def quick_eval_exits(opens, highs, lows, closes, signals, stops, tps,
                     fee=0.00045, slippage=0.0002, buy=None, sell=None, side="both",
                     momentum=True, mom_activation=0.006, mom_giveback=0.0025,
                     ladder=True, ladder_wake=0.008, ladder_tiers=LADDER_PADRAO,
                     ladder_giveback=0.30, tp_after_ladder=0.05, eval_mode="close",
                     units="pct", ladder_dynamic=True, pause_after_sl=0):
    """Simula as regras de saída do teu bot sobre os sinais, uma posição de cada vez:
      SL e TP do sinal (ordens na exchange; SL primeiro se ambos na mesma vela);
      momentum: com pico >= mom_activation, fecha a mercado se o PnL recuar mom_giveback do pico;
      escada: sobe o SL para entrada*(1+/-secure) conforme o PnL (só sobe) e, na 1.ª subida,
      substitui o TP por tp_after_ladder (None = mantém o TP do sinal). SEM timeout.
    eval_mode='close': momentum e escada avaliados ao fecho de cada vela (backtest típico);
    eval_mode='intrabar': pior caso ~tick a tick (pico pelas máximas/mínimas da vela, o recuo
    pode ocorrer na própria vela). Efeitos da escada aplicam-se a partir da vela seguinte.
    units='pct': limiares em fração do preço (como no bot). units='R': limiares em múltiplos do
    risco inicial do trade (mom_activation=1.0 => +1R; tiers/wake/tp_after_ladder também em R).
    pause_after_sl=N: depois de uma saída em prejuízo pelo SL, ignora sinais durante N velas
    (testa se reduzir as reentradas seguidas na mesma zona ajuda).
    Devolve estatísticas + 'trades' [{i, dir, R, ret, motivo, barras}] e 'motivos'."""
    enum = type(signals[0])
    buy = buy if buy is not None else enum.BUY
    sell = sell if sell is not None else getattr(enum, "SELL", None)
    n = len(closes)
    out_trades, next_free = [], 0
    for i in range(n - 1):
        if i < next_free:
            continue
        if signals[i] == buy and side in ("both", "long"):
            d = 1
        elif sell is not None and signals[i] == sell and side in ("both", "short"):
            d = -1
        else:
            continue
        sl0, tp0 = stops[i], tps[i]
        ent = opens[i + 1] * (1 + d * slippage)
        risk = d * (ent - sl0)
        if risk <= 0 or d * (tp0 - ent) <= 0:
            continue
        sc = risk / ent if units == "R" else 1.0            # converte limiares em R para fração do preço
        m_act, m_gb, l_wake = mom_activation * sc, mom_giveback * sc, ladder_wake * sc
        tiers = tuple((t_ * sc, s_ * sc) for t_, s_ in ladder_tiers)
        tp_ov = tp_after_ladder * sc if tp_after_ladder else None
        sl, tp, peak, last_adj = sl0, tp0, 0.0, None
        ex = k = motivo = None
        for j in range(i + 1, n):
            if d == 1:
                if lows[j] <= sl:
                    ex, k, motivo = min(opens[j], sl), j, "SL"
                    break
                if highs[j] >= tp:
                    ex, k, motivo = tp, j, "TP"
                    break
            else:
                if highs[j] >= sl:
                    ex, k, motivo = max(opens[j], sl), j, "SL"
                    break
                if lows[j] <= tp:
                    ex, k, motivo = tp, j, "TP"
                    break
            if eval_mode == "close":
                p_now = d * (closes[j] - ent) / ent
                peak = max(peak, p_now)
                if momentum and peak >= m_act and peak - p_now >= m_gb:
                    ex, k, motivo = closes[j], j, "MOMENTUM"
                    break
                p_lad = p_now
            else:
                fav = highs[j] if d == 1 else lows[j]
                adv = lows[j] if d == 1 else highs[j]
                peak = max(peak, d * (fav - ent) / ent)
                if momentum and peak >= m_act and peak - d * (adv - ent) / ent >= m_gb:
                    ex, k, motivo = ent * (1 + d * (peak - m_gb)), j, "MOMENTUM"
                    break
                p_lad = peak
            if ladder:
                adj = _ladder_secure(p_lad, tiers, l_wake, ladder_giveback, ladder_dynamic)
                if adj is not None and (last_adj is None or adj > last_adj):
                    last_adj = adj
                    new_sl = ent * (1 + d * adj)
                    if d * (new_sl - sl) > 0:
                        sl = new_sl
                    if tp_ov:
                        tp = ent * (1 + d * tp_ov)
        if ex is None:
            ex, k, motivo = closes[-1], n - 1, "FIM"
        ex_eff = ex * (1 - d * slippage)
        pnl = d * (ex_eff - ent) - fee * (ent + ex_eff)
        out_trades.append({"i": i, "dir": d, "R": round(pnl / risk, 4), "ret": round(pnl / ent, 6),
                           "motivo": motivo, "barras": k - i, "risk_pct": risk / ent,
                           "entrada": ent, "saida": ex, "j": k})
        next_free = k + (pause_after_sl if (motivo == "SL" and pnl < 0) else 0)
    res = _stats([t["R"] for t in out_trades], [t["barras"] for t in out_trades],
                 [t["risk_pct"] for t in out_trades])
    res["trades"] = out_trades
    res["motivos"] = {}
    for t in out_trades:
        res["motivos"][t["motivo"]] = res["motivos"].get(t["motivo"], 0) + 1
    return res


# ---------------------- estratégia 2: extremo das bandas de Bollinger + gatilho RSI
def _rolling_mean_std(values, n):
    """Média e desvio-padrão (populacional, como no TradingView) das últimas n velas."""
    import math
    out_m, out_s = [None] * len(values), [None] * len(values)
    s1 = s2 = 0.0
    for i, v in enumerate(values):
        s1 += v
        s2 += v * v
        if i >= n:
            u = values[i - n]
            s1 -= u
            s2 -= u * u
        if i >= n - 1:
            mean = s1 / n
            out_m[i] = mean
            out_s[i] = math.sqrt(max(s2 / n - mean * mean, 0.0))
    return out_m, out_s


def _rsi_wilder(closes, period=14):
    n = len(closes)
    out = [50.0] * n
    if n <= period:
        return out
    gains = losses = 0.0
    for i in range(1, period + 1):
        ch = closes[i] - closes[i - 1]
        gains += max(ch, 0.0)
        losses += max(-ch, 0.0)
    ag, al = gains / period, losses / period
    out[period] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    for i in range(period + 1, n):
        ch = closes[i] - closes[i - 1]
        ag = (ag * (period - 1) + max(ch, 0.0)) / period
        al = (al * (period - 1) + max(-ch, 0.0)) / period
        out[i] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    return out


def analyze_candles_bb_rsi(
        self,
        mode: str = "ambos",          # "ambos" | "so_gatilho" | "so_zona" (para medir cada parte)
        zone_def: str = "bandas_cruzamento",  # "bandas_cruzamento" | "bandas_estado" | "preco_fora_ambas"
        bb_fast: int = 20,
        bb_slow: int = 80,
        bb_k: float = 2.0,            # nº de desvios-padrão das duas bandas
        zone_window: int = 24,        # velas (24 = 6h) em que a zona extrema conta como "recente"
        rsi_period: int = 14,
        rsi_ema: int = 9,
        rsi_level: float = 50.0,
        rsi_hold: int = 1,            # velas seguidas com o RSI do lado certo de rsi_level
        tf_factor: int = 4,
        atr_period: int = 14,
        stop_atr: float = 2.0,
        tp_r: float = 3.0,
        cost_roundtrip_pct: float = 0.0012,
        min_risk_over_cost: float = 3.0,
        cooldown_bars: int = 8,
        warmup: int = 872,
):
    """Zona extrema (zone_def), LONG = sobrevendido, SHORT = sobrecomprado, nas últimas zone_window velas:
      'bandas_cruzamento': a banda inferior das BB(bb_fast) CRUZA para baixo da inferior das BB(bb_slow)
          (e a superior para cima da superior): o momento do cruzamento.
      'bandas_estado': a banda rápida está abaixo/acima da lenta (condição contínua). ATENÇÃO: com o mesmo
          nº de desvios nas duas, isto acontece ~metade do tempo, porque reflete sobretudo a média curta
          estar abaixo/acima da longa; não é um extremo.
      'preco_fora_ambas': o fecho está fora das duas bandas inferiores (LONG) ou superiores (SHORT). Gatilho: o RSI cruza a sua EMA para cima e fica acima de rsi_level (LONG) ou
    cruza para baixo e fica abaixo (SHORT). Entrada: o sinal está na vela i fechada; a entrada é na
    abertura seguinte. SL = fecho -/+ stop_atr x ATR(1h); TP = tp_r x risco. É contra-tendência.
    mode='so_gatilho' ignora a zona; mode='so_zona' entra quando a zona extrema começa (sem RSI)."""
    closes = [float(x) for x in self.closes]
    highs = [float(x) for x in self.highs]
    lows = [float(x) for x in self.lows]
    n = len(closes)
    signals = [Signal.HOLD] * n
    stops = [0.0] * n
    tps = [0.0] * n
    funnel = {k: 0 for k in ("zona_long", "zona_short", "gatilho_long", "gatilho_short", "setup", "cost_ok", "signals")}
    self.last_funnel = funnel
    if n <= warmup + 5:
        return signals, stops, tps

    m_f, s_f = _rolling_mean_std(closes, bb_fast)
    m_s, s_s = _rolling_mean_std(closes, bb_slow)
    rsi = _rsi_wilder(closes, rsi_period)
    rsi_e = _ema(rsi, rsi_ema)
    hh, hl, hc = _aggregate(highs, lows, closes, tf_factor)
    atr_1h = _atr_wilder(hh, hl, hc, atr_period)

    last_zl = last_zs = -10 ** 9
    last_signal = -10 ** 9
    prev_l = prev_s = False                  # estado da condição de bandas na vela anterior
    for i in range(1, n):
        started_zl = started_zs = False
        if m_s[i] is not None:
            low_f, low_s = m_f[i] - bb_k * s_f[i], m_s[i] - bb_k * s_s[i]
            up_f, up_s = m_f[i] + bb_k * s_f[i], m_s[i] + bb_k * s_s[i]
            state_l, state_s = low_f < low_s, up_f > up_s
            if zone_def == "bandas_cruzamento":
                zl, zs = state_l and not prev_l, state_s and not prev_s
            elif zone_def == "bandas_estado":
                zl, zs = state_l, state_s
            elif zone_def == "preco_fora_ambas":
                zl, zs = closes[i] < min(low_f, low_s), closes[i] > max(up_f, up_s)
            else:
                raise ValueError("zone_def inválido")
            prev_l, prev_s = state_l, state_s
            if zl:
                started_zl = (i - last_zl) > 1
                last_zl = i
            if zs:
                started_zs = (i - last_zs) > 1
                last_zs = i
        if i < warmup:
            continue
        recent_l = (i - last_zl) < zone_window
        recent_s = (i - last_zs) < zone_window
        funnel["zona_long"] += recent_l
        funnel["zona_short"] += recent_s

        up = rsi[i - 1] <= rsi_e[i - 1] and rsi[i] > rsi_e[i] and all(rsi[i - k] > rsi_level for k in range(rsi_hold))
        dn = rsi[i - 1] >= rsi_e[i - 1] and rsi[i] < rsi_e[i] and all(rsi[i - k] < rsi_level for k in range(rsi_hold))
        funnel["gatilho_long"] += up
        funnel["gatilho_short"] += dn

        if mode == "ambos":
            side = 1 if (recent_l and up) else (-1 if (recent_s and dn) else 0)
        elif mode == "so_gatilho":
            side = 1 if up else (-1 if dn else 0)
        elif mode == "so_zona":
            side = 1 if started_zl else (-1 if started_zs else 0)
        else:
            raise ValueError("mode deve ser 'ambos', 'so_gatilho' ou 'so_zona'")
        if side == 0:
            continue
        funnel["setup"] += 1

        done = (i + 1) // tf_factor
        a = atr_1h[done - 1] if done >= 1 else None
        if a is None or a <= 0:
            continue
        c = closes[i]
        risk = stop_atr * a
        if (risk / c) < min_risk_over_cost * cost_roundtrip_pct:
            continue
        funnel["cost_ok"] += 1
        if i - last_signal < cooldown_bars:
            continue
        last_signal = i
        signals[i] = Signal.BUY if side == 1 else Signal.SELL
        stops[i] = c - side * risk
        tps[i] = c + side * tp_r * risk
        funnel["signals"] += 1
    return signals, stops, tps