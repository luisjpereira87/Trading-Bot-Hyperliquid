"""
Análise OFFLINE (não faz parte do bot): as piores entradas têm algo em comum?

Corre a estratégia em cada ativo SEPARADAMENTE (nunca juntes candles de ativos
diferentes) e agrega só os resultados: medidas de esgotamento + resultado em R.

Uso:
    python analise_esgotamento.py              # DEMO=True: dados sintéticos; DEMO=False: Hyperliquid
    python analise_esgotamento.py --comparar   # compara testnet e mainnet (DEMO=False)
"""
import asyncio
import enum
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import ema_retest_15m as m

DEMO = False                                   # False para usar dados reais (Hyperliquid)
TESTNET = False                               # análise com MAINNET: dados públicos, sem chaves
DATA_DIR = "dados_candles"                    # histórico guardado, cresce a cada execução
FONTE = "binance"                         # "binance" = histórico longo (validação fora da amostra)
DESDE = "2025-01-01"                          # só para FONTE = "binance"
CAPITAL_USDC = 3000                           # saldo da conta: só para expressar PnL e drawdown em % do saldo
SIZING = "fixo"                               # "fixo": nocional constante (como o teu bot) | "risco": risco constante
NOTIONAL_USDC = 3000                          # nocional por trade = margem por ativo x alavancagem (só para SIZING = "fixo")
                                              # ex.: 30% de 3000 = 900 de margem x 10x = 9000. AJUSTA ao teu bot
RISK_USDC = 10                                # 1% de 1000 USDC por trade (só para SIZING = "risco")
ESTRATEGIA = "bb_rsi"                        # "reteste" (EMA) | "bb_rsi" (extremo Bollinger + gatilho RSI)
MODO_BB = "ambos"                             # só para "bb_rsi": "ambos" | "so_gatilho" | "so_zona" (mede cada parte)
PARAMS_BB = {"zone_def": "preco_fora_ambas"}                                 # só para "bb_rsi": ex. {"zone_window": 48, "bb_k": 2.0, "rsi_period": 14}
LISTAR_TRADES = None                           # ex.: ("ETH/USDC:USDC", "momentum + escada [vela]"): lista os trades simulados
COMPARAR_SAIDAS = True                        # compara as regras de saída do bot com SL/TP fixos
VARIANTES_SAIDA = {                           # nome: parâmetros do quick_eval_exits
    "SL/TP fixos":                dict(momentum=False, ladder=False, tp_after_ladder=None),
    "SL/TP fixos + pausa 24 após SL": dict(momentum=False, ladder=False, tp_after_ladder=None, pause_after_sl=24),
    "SL/TP fixos + pausa 48 após SL": dict(momentum=False, ladder=False, tp_after_ladder=None, pause_after_sl=48),
    "+ momentum":                 dict(momentum=True, ladder=False, tp_after_ladder=None),
    "+ escada (SL sobe, TP 5%)":  dict(momentum=False, ladder=True),
    "momentum + escada [vela]":   dict(momentum=True, ladder=True, eval_mode="close"),
    "momentum + escada [tick]":   dict(momentum=True, ladder=True, eval_mode="intrabar"),
    # variantes em múltiplos de R (os limiares escalam com o risco de cada trade), sem substituir o TP
    "breakeven a +1R":            dict(momentum=False, ladder=True, units="R", ladder_wake=1.0,
                                       ladder_tiers=((1.0, 0.0),), ladder_dynamic=False, tp_after_ladder=None),
    "BE +1R, stop +1R aos 2R":    dict(momentum=False, ladder=True, units="R", ladder_wake=1.0,
                                       ladder_tiers=((2.0, 1.0), (1.0, 0.0)), ladder_dynamic=False,
                                       tp_after_ladder=None),
    "momentum R: 1R / 0.5R":      dict(momentum=True, ladder=False, units="R", mom_activation=1.0,
                                       mom_giveback=0.5, tp_after_ladder=None),
    "momentum R: 1.5R / 1R":      dict(momentum=True, ladder=False, units="R", mom_activation=1.5,
                                       mom_giveback=1.0, tp_after_ladder=None),
}
ENTRADAS = {                                  # nome: filtros do exhaustion_filter aplicados aos sinais
    "todas": {},
    "slope>=3": {"min_slope_atr": 3.0},
    "slope>=4": {"min_slope_atr": 4.0},
    "gap_change>=1.5": {"min_gap_change_atr": 1.5},
    "er>=0.20": {"min_er": 0.20},                       # exclui mercado lateral
    "er>=0.25": {"min_er": 0.25},
    "er>=0.20 + slope>=2": {"min_er": 0.20, "min_slope_atr": 2.0},
}
MOSTRAR_TABELAS_SAIDAS = True                 # False = só o RESUMO final
MIN_TRADES_CAND = 50                          # critérios do RESUMO de candidatas
MIN_ATIVOS_POS = 2
SCANS = {                                      # medida: limiares a varrer (acima vs abaixo)
    "regime_age_h": [24, 72, 120, 168],
    "gap_change_atr": [-1.0, 0.0, 0.5, 1.0, 1.5, 2.0],
    "slope_atr": [0.0, 1.0, 2.0, 3.0, 4.0],
    "er": [0.08, 0.12, 0.16, 0.20, 0.25],            # eficiência: perto de 0 = lateral
    "gap_level_atr": [1.0, 2.0, 3.0, 4.0, 6.0],      # EMAs coladas = pouca tendência
}
MAPA_BINANCE = {"ETH/USDC:USDC": "ETH/USDT", "BTC/USDC:USDC": "BTC/USDT", "SOL/USDC:USDC": "SOL/USDT"}
ATIVOS = ["ETH/USDC:USDC", "BTC/USDC:USDC", "SOL/USDC:USDC"]
COST_ROUNDTRIP = 0.0012                       # perps Hyperliquid (taker + slippage), ajusta
FEE, SLIPPAGE = 0.00045, 0.0002
N_PIORES = 10
BAR_MS = 15 * 60 * 1000


def pnl_usdc(t):
    """PnL teórico do trade em USDC, já com comissões e slippage (SL e TP fixos)."""
    return t["ret"] * NOTIONAL_USDC if SIZING == "fixo" else t["R"] * RISK_USDC


class Signal(enum.Enum):                      # só para esta análise
    BUY = 1
    SELL = -1
    HOLD = 0


m.Signal = Signal


# ---------------------------------------------------------------- dados
async def _fetch(simbolo, n, testnet):
    """Mesmo padrão do teu get_historical_ohlcv, mas sem cache e com a rede escolhida."""
    from ccxt.async_support import hyperliquid
    exchange = hyperliquid({"enableRateLimit": True, "testnet": testnet})
    # Se precisares do teu patch de mercados spot, aplica-o aqui:
    # exchange.fetch_spot_markets = safe_fetch_spot_markets.__get__(exchange, exchange.__class__)
    await exchange.load_markets()
    try:
        return await exchange.fetch_ohlcv(simbolo, "15m", limit=n)
    finally:
        await exchange.close()


async def _fetch_binance(simbolo, desde_ms):
    """Histórico longo por páginas de 1000 velas (só dados públicos, sem chaves)."""
    from ccxt.async_support import binance
    ex = binance({"enableRateLimit": True})
    await ex.load_markets()
    try:
        out, since = [], desde_ms
        while True:
            lote = await ex.fetch_ohlcv(simbolo, "15m", since=since, limit=1000)
            if not lote:
                break
            out += [k for k in lote if not out or k[0] > out[-1][0]]
            since = lote[-1][0] + BAR_MS
            if len(lote) < 1000:
                break
        return out
    finally:
        await ex.close()


def _ficheiro(simbolo):
    rede = "binance" if FONTE == "binance" else ("testnet" if TESTNET else "mainnet")
    return os.path.join(DATA_DIR, simbolo.replace("/", "_").replace(":", "_") + f"_{rede}.json")


def carregar_klines(simbolo, n=5000):
    """Devolve [[timestamp_ms, open, high, low, close, volume], ...] só com velas FECHADAS.
    Em modo real, junta o que descarregou agora com o que já guardaste (por timestamp),
    para o histórico crescer mesmo que a API só devolva as velas mais recentes."""
    if DEMO:
        rnd = random.Random(abs(hash(simbolo)) % 10_000)
        c, t0, out = 100.0, 1_700_000_000_000 // 3_600_000 * 3_600_000, []
        for i in range(n):
            d = 0.0004 + 0.0035 * math.cos(2 * math.pi * i / 160)
            o = c
            c = o * math.exp(d + rnd.gauss(0, 0.004))
            hi = max(o, c) * (1 + abs(rnd.gauss(0, 0.0015)))
            lo = min(o, c) * (1 - abs(rnd.gauss(0, 0.0015)))
            out.append([t0 + i * BAR_MS, o, hi, lo, c, 1.0])
        return out
    agora = time.time() * 1000
    if FONTE == "binance":
        desde_ms = int(datetime.fromisoformat(DESDE).replace(tzinfo=timezone.utc).timestamp() * 1000)
        brutos = asyncio.run(_fetch_binance(MAPA_BINANCE[simbolo], desde_ms))
    else:
        brutos = asyncio.run(_fetch(simbolo, n, TESTNET))
    novos = [k for k in brutos if k[0] + BAR_MS <= agora]
    os.makedirs(DATA_DIR, exist_ok=True)
    caminho = _ficheiro(simbolo)
    antigos = json.load(open(caminho)) if os.path.exists(caminho) else []
    por_ts = {k[0]: k for k in antigos}
    por_ts.update({k[0]: k for k in novos})
    todos = [por_ts[t] for t in sorted(por_ts)]
    json.dump(todos, open(caminho, "w"))
    return todos


def comparar_redes(simbolo, n=300):
    """Compara os fechos da testnet e da mainnet nas mesmas velas (diferença em %)."""
    a = {k[0]: k[4] for k in asyncio.run(_fetch(simbolo, n, True))}
    b = {k[0]: k[4] for k in asyncio.run(_fetch(simbolo, n, False))}
    comuns = sorted(set(a) & set(b))
    if not comuns:
        print(f"{simbolo}: sem velas em comum")
        return
    difs = [abs(a[t] - b[t]) / b[t] * 100 for t in comuns]
    print(f"{simbolo}: {len(comuns)} velas em comum | diferença média {sum(difs) / len(difs):.3f}% "
          f"| máxima {max(difs):.3f}%")


def preparar(klines):
    """Alinha a uma hora cheia e verifica buracos. A última vela deve estar FECHADA."""
    while klines and klines[0][0] % (60 * 60 * 1000) != 0:
        klines = klines[1:]
    ts = [k[0] for k in klines]
    buracos = [i for i in range(len(ts) - 1) if ts[i + 1] - ts[i] != BAR_MS]
    d = SimpleNamespace(
        opens=[float(k[1]) for k in klines], highs=[float(k[2]) for k in klines],
        lows=[float(k[3]) for k in klines], closes=[float(k[4]) for k in klines], ts=ts)
    return d, buracos


# ------------------------------------------------------------- saídas
def comparar_saidas(ativos):
    """Corre as regras de saída do bot sobre os mesmos sinais (uma posição de cada vez por
    ativo) e compara o PnL em USDC com SL/TP fixos, para vários filtros de entrada."""
    print("\n" + "=" * 112)
    print("REGRAS DE SAÍDA vs SL/TP FIXOS: uma posição de cada vez por ativo, PnL em USDC com comissões e slippage")
    print("[vela] = momentum e escada avaliados ao fecho de cada vela | [tick] = pior caso dentro da vela "
          "(o resultado real fica algures entre os dois)")
    nomes = [x.split("/")[0] for x in ativos]
    resultados = []
    for nome_ent, filtro in ENTRADAS.items():
        filtrados = {}
        for simbolo, (d, s, sl, tp) in ativos.items():
            if filtro:
                s2, sl2, tp2, _ = m.exhaustion_filter(d.opens, d.highs, d.lows, d.closes, s, sl, tp,
                                                      buy=Signal.BUY, sell=Signal.SELL, **filtro)
            else:
                s2, sl2, tp2 = s, sl, tp
            filtrados[simbolo] = (s2, sl2, tp2)
        if MOSTRAR_TABELAS_SAIDAS:
            print(f"\n--- ENTRADAS: {nome_ent} ---")
            print(f"{'variante':32}" + "".join(f"{n:>8}" for n in nomes) + f"{'TOTAL':>9} {'%saldo':>6} | "
              f"{'DD máx (% saldo)':>17} | trades acerto  exp_R | 1ª metade 2ª metade | saídas")
        for nome_var, kw in VARIANTES_SAIDA.items():
            pnl_a, todos = [], []
            for simbolo, (d, *_resto) in ativos.items():
                s2, sl2, tp2 = filtrados[simbolo]
                r = m.quick_eval_exits(d.opens, d.highs, d.lows, d.closes, s2, sl2, tp2,
                                       fee=FEE, slippage=SLIPPAGE, buy=Signal.BUY, sell=Signal.SELL, **kw)
                for t in r["trades"]:
                    t["pnl"] = pnl_usdc(t)
                    t["metade"] = 1 if t["i"] < len(d.closes) / 2 else 2
                pnl_a.append(sum(t["pnl"] for t in r["trades"]))
                todos += r["trades"]
            if not todos:
                if MOSTRAR_TABELAS_SAIDAS:
                    print(f"{nome_var:32} sem trades")
                continue
            n_t = len(todos)
            acerto = sum(1 for t in todos if t["R"] > 0) / n_t
            exp_r = sum(t["R"] for t in todos) / n_t
            h1 = sum(t["pnl"] for t in todos if t["metade"] == 1)
            h2 = sum(t["pnl"] for t in todos if t["metade"] == 2)
            mot = {}
            for t in todos:
                mot[t["motivo"]] = mot.get(t["motivo"], 0) + 1
            mot_txt = " ".join(f"{k}:{v}" for k, v in sorted(mot.items(), key=lambda kv: -kv[1]))
            cum = peak = dd = minimo = 0.0                       # curva de PnL da carteira, por data de saída
            for t in sorted(todos, key=lambda t: t["i"] + t["barras"]):
                cum += t["pnl"]
                peak = max(peak, cum)
                dd = max(dd, peak - cum)
                minimo = min(minimo, cum)
            aviso = " SALDO ESGOTADO" if CAPITAL_USDC + minimo <= 0 else ""
            dd_txt = f"{dd:,.0f} ({dd / CAPITAL_USDC:.0%})"
            resultados.append({"entrada": nome_ent, "variante": nome_var, "total": sum(pnl_a), "h1": h1, "h2": h2,
                               "trades": n_t, "dd": dd, "ativos": pnl_a, "exp_r": exp_r})
            if MOSTRAR_TABELAS_SAIDAS:
                print(f"{nome_var:32}" + "".join(f"{x:>+8,.0f}" for x in pnl_a) + f"{sum(pnl_a):>+9,.0f} "
                      f"{sum(pnl_a) / CAPITAL_USDC:>+6.0%} | {dd_txt:>17} | "
                      f"{n_t:6d} {acerto:5.0%} {exp_r:+6.2f}R | {h1:>+9,.0f} {h2:>+9,.0f} | {mot_txt}{aviso}")
    resumo_candidatas(resultados)


def resumo_candidatas(res):
    """Resume tudo numa lista curta: só as configurações (entrada + saída) com PnL > 0 nas DUAS
    metades, positivas em pelo menos MIN_ATIVOS_POS ativos e com trades suficientes."""
    cand = [r for r in res if r["total"] > 0 and r["h1"] > 0 and r["h2"] > 0
            and sum(1 for x in r["ativos"] if x > 0) >= MIN_ATIVOS_POS and r["trades"] >= MIN_TRADES_CAND]
    cand.sort(key=lambda r: r["total"] / max(r["dd"], 1.0), reverse=True)
    print("\n" + "=" * 112)
    print(f"RESUMO: configurações com PnL > 0 nas DUAS metades, positivas em >= {MIN_ATIVOS_POS} ativos e "
          f">= {MIN_TRADES_CAND} trades, por ordem de PnL / drawdown")
    print("ATENÇÃO: é DENTRO DA AMOSTRA. Com muitas configurações testadas, algumas passam por sorte; "
          "o teste a sério é em dados novos.")
    if not cand:
        print("  nenhuma configuração cumpre os critérios")
        return
    for r in cand[:15]:
        print(f"  {r['entrada']:20} {r['variante']:32} PnL {r['total']:>+6,.0f} ({r['total'] / CAPITAL_USDC:>+4.0%}) "
              f"DD {r['dd'] / CAPITAL_USDC:>4.0%} PnL/DD {r['total'] / max(r['dd'], 1.0):4.1f} "
              f"trades {r['trades']:4d} metades {r['h1']:>+6,.0f}/{r['h2']:>+6,.0f} "
              f"ativos {' '.join(f'{x:+,.0f}' for x in r['ativos'])}")
    print(f"  {len(cand)} de {len(res)} configurações cumprem os critérios")


def listar_trades(ativos, simbolo, nome_var):
    """Lista os trades simulados de um ativo (entradas 'todas'), para comparar linha a linha
    com o relatório do teu executor: hora UTC, lado, preço de entrada e de saída, motivo, PnL."""
    d, s, sl, tp = ativos[simbolo]
    r = m.quick_eval_exits(d.opens, d.highs, d.lows, d.closes, s, sl, tp, fee=FEE, slippage=SLIPPAGE,
                           buy=Signal.BUY, sell=Signal.SELL, **VARIANTES_SAIDA[nome_var])
    iso = lambda ms: datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%m-%d %H:%M")
    print(f"\nTRADES SIMULADOS: {simbolo} | {nome_var} | {len(r['trades'])} trades")
    for n, t in enumerate(r["trades"], 1):
        print(f"  {n:3d} {iso(d.ts[t['i'] + 1])} -> {iso(d.ts[t['j']])} {'BUY ' if t['dir'] == 1 else 'SELL'} "
              f"entrada {t['entrada']:.2f} saída {t['saida']:.2f} {t['motivo']:9} PnL {pnl_usdc(t):+7.1f}")


# ------------------------------------------------------------- análise
def main():
    print(f"ESTRATÉGIA: {ESTRATEGIA}" + (f" (modo {MODO_BB})" if ESTRATEGIA == "bb_rsi" else ""))
    if DEMO:
        print("⚠️  MODO DEMO: dados SINTÉTICOS. Os resultados não significam nada. "
              "Põe DEMO = False para usar a Hyperliquid.\n")
    feats_all, trades_all, lens, ativos = {}, [], {}, {}
    for simbolo in ATIVOS:
        d, buracos = preparar(carregar_klines(simbolo))
        if buracos:
            print(f"{simbolo}: ATENÇÃO, {len(buracos)} buracos nos dados (1.º no índice {buracos[0]})")
        lens[simbolo] = len(d.closes)
        if ESTRATEGIA == "bb_rsi":
            s, sl, tp = m.analyze_candles_bb_rsi(d, mode=MODO_BB, cost_roundtrip_pct=COST_ROUNDTRIP, **PARAMS_BB)
        else:
            s, sl, tp = m.analyze_candles_history(d, cost_roundtrip_pct=COST_ROUNDTRIP)
        ativos[simbolo] = (d, s, sl, tp)
        r = m.quick_eval_R(d.opens, d.highs, d.lows, d.closes, s, sl, tp,
                           fee=FEE, slippage=SLIPPAGE, buy=Signal.BUY, sell=Signal.SELL,
                           return_trades=True, allow_overlap=True)
        r1 = m.quick_eval_R(d.opens, d.highs, d.lows, d.closes, s, sl, tp,
                            fee=FEE, slippage=SLIPPAGE, buy=Signal.BUY, sell=Signal.SELL,
                            return_trades=True)                      # uma posição de cada vez
        pnl_1pos = sum(pnl_usdc(t) for t in r1["trades"])
        pnl_todos = sum(pnl_usdc(t) for t in r["trades"])
        f = m.signal_features(d.opens, d.highs, d.lows, d.closes, s,
                              buy=Signal.BUY, sell=Signal.SELL)
        iso = lambda ms: datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d %H:%M")
        print(f"{simbolo}: {len(d.closes)} velas de {iso(d.ts[0])} a {iso(d.ts[-1])} UTC"
              f" | último fecho {d.closes[-1]:.2f} | sinais {sum(1 for x in s if x != Signal.HOLD)}"
              f" | com medidas {len(f)}\n   PnL (1 posição de cada vez, {len(r1['trades'])} trades): {pnl_1pos:+,.0f} USDC"
              f" | PnL (todos os {len(r['trades'])} sinais): {pnl_todos:+,.0f} USDC\n   funil {d.last_funnel}")
        for i, v in f.items():
            feats_all[(simbolo, i)] = v
        for t in r["trades"]:
            trades_all.append({**t, "i": (simbolo, t["i"]), "pnl": pnl_usdc(t)})

    tot = sum(t["pnl"] for t in trades_all)
    print(f"\nPnL em USDC: SL e TP fixos, SEM trailing/momentum, já com comissões e slippage; "
          f"sizing {SIZING} ({NOTIONAL_USDC} USDC de nocional)" if SIZING == "fixo" else
          f"\nPnL em USDC: SL e TP fixos, SEM trailing/momentum, com comissões e slippage; "
          f"risco constante de {RISK_USDC} USDC por trade")
    print(f"TOTAL agregado: {len(trades_all)} sinais com resultado | PnL total {tot:+,.0f} USDC "
          f"({tot / len(trades_all):+.2f} por sinal)")
    print("NOTA: estes PnL são somas TEÓRICAS: sem limite de saldo, sem liquidação e (nas linhas 'todos os sinais', "
          "tercis e varreduras) com posições sobrepostas que nunca existiriam ao mesmo tempo. Não são uma curva de conta.")

    p = m.worst_trades_profile(feats_all, trades_all, n=N_PIORES)
    if p.get("aviso"):
        print(p["aviso"])
    else:
        print(f"\nPIORES {p['n_piores']} DE {p['n_trades']} TRADES: percentil médio por medida "
              f"(50 = não os distingue; fora de ~35-65 merece atenção)")
        for nome, pct in sorted(p["percentil_medio_dos_piores"].items(),
                                key=lambda kv: abs(kv[1] - 50), reverse=True):
            print(f"  {nome:20} {pct:3d}")
        print("\nOs piores, um a um:")
        for x in p["detalhe"]:
            ativo, idx = x["i"]
            print(f"  {ativo} idx {idx} {'BUY ' if x['dir'] == 1 else 'SELL'} R={x['R']:+.2f} "
                  f"PnL={x['pnl']:+.1f}  {x['percentis']}")

    rep = m.feature_report(feats_all, trades_all)
    if isinstance(rep, dict):
        print(rep["aviso"])
    else:
        print("\nTERCIS por medida (baixo, médio, alto): n | valor mediano | acerto | exp_R | PnL total (médio/sinal)")
        for row in rep:
            partes = [f"{t['n']:4d} | {t['valor_mediano']:7.2f} | {t['acerto']:.0%} | {t['exp_R']:+.2f}R | "
                      f"{t['pnl_total']:+,.0f} ({t['pnl_medio']:+.1f})"
                      for t in row["tercis (baixo, médio, alto)"]]
            print(f"  {row['medida']:20} " + "   ##   ".join(partes))

    # onde muda o desempenho? varredura de limiares por medida
    ep = lambda k, f: (k[0], k[1] - int(round(f["regime_age_h"] * 4)))      # onde o regime começou
    meio = lambda k: ("1ª" if k[1] < lens[k[0]] / 2 else "2ª") + " metade"
    fmt = lambda v: (f"n={v['n']:4d} {v.get('acerto', 0):.0%} {v.get('exp_R', 0):+.2f}R "
                     f"{v.get('pnl_total', 0):+,.0f}$" if v["n"] else "n=   0")
    for medida, limiares in SCANS.items():
        print(f"\n=== VARREDURA {medida}: ACIMA vs ABAIXO do limiar "
              f"(z = acerto face a 25%; eps = episódios de regime; exp/ep = R médio por episódio) ===")
        for row in m.threshold_scan(feats_all, trades_all, medida, limiares,
                                    group_fn=lambda k: k[0], episode_fn=ep):
            a_, b_ = row["acima"], row["abaixo"]
            por = " | ".join(f"{g.split('/')[0]} {fmt(v)}" for g, v in row["acima_por_grupo"].items())
            print(f"  T={row['limiar']:>6}: acima {fmt(a_)} z={a_.get('z', 0):+.1f} eps={a_.get('episodios', 0):3d} "
                  f"exp/ep={a_.get('exp_R_por_episodio', 0):+.2f} || abaixo {fmt(b_)} || acima por ativo: {por}")
        print("  por metade do período (1ª / 2ª):")
        for row in m.threshold_scan(feats_all, trades_all, medida, limiares, group_fn=meio):
            ac = " | ".join(f"{g[:2]} {fmt(v)}" for g, v in row["acima_por_grupo"].items())
            ab = " | ".join(f"{g[:2]} {fmt(v)}" for g, v in row["abaixo_por_grupo"].items())
            print(f"    T={row['limiar']:>6}: acima {ac}  ##  abaixo {ab}")

    if COMPARAR_SAIDAS:
        comparar_saidas(ativos)
    if LISTAR_TRADES:
        listar_trades(ativos, *LISTAR_TRADES)

if __name__ == "__main__":
    if "--comparar" in sys.argv:
        for _s in ATIVOS:
            comparar_redes(_s)
    else:
        main()