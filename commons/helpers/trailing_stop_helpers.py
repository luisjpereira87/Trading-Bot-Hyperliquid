class TrailingStopHelpers:

    # Configuração Única de Trailing Stop
    # Nota: Manter ordenado do maior para o menor target
    TRAILING_STRATEGY = [
        {"target": 0.02,  "secure": 0.015, "icon": "🔥", "name": "META_2", "log": "🔥 Meta de 2% atingida! Stop subiu para garantir 1.5%"},
        {"target": 0.01,  "secure": 0.006, "icon": "💰", "name": "META_1", "log": "💰 Meta de 1% atingida! Stop subiu para garantir 0.6%"},
        {"target": 0.004, "secure": 0.001, "icon": "🛡️", "name": "BREAK_EVEN", "log": "🛡️ Break-even ativo! Taxas cobertas e lucro mínimo garantido."},
        {"target": 0.002, "secure": 0.0, "icon": "🐌", "name": "LATERAL_EXIT", "log": "🐌 Break-even ativo! Taxas cobertas."}
    ]

    # Parâmetros de comportamento para Alavancagem
    STEP_SIZE = 0.002  # Sobe o stop a cada 0.2% de lucro (ajustável)
    MIN_TARGET = 0.003  # Só ativa o trailing após 0.3% (proteção de taxas)

    @staticmethod
    def get_trailing_adjustment_old(pnl_pct):
        """
        Percorre a estratégia e devolve o ajuste correspondente ao lucro atual.
        """
        # Garantimos que estamos a avaliar patamares positivos
        # Se o PNL for negativo, nem vale a pena percorrer o loop
        if pnl_pct <= 0:
            return 0, "", "📊 PNL atual: {pnl_pct:.2%}. Nenhum patamar de trailing atingido."

        for level in TrailingStopHelpers.TRAILING_STRATEGY:
            if pnl_pct >= level["target"]:
                return level["secure"], level["icon"], level["log"]
        
        return 0, "", "📊 PNL atual: {pnl_pct:.2%}. Nenhum patamar de trailing atingido."

    @staticmethod
    def get_trailing_adjustment(pnl_pct):
        """
        Calcula o ajuste de forma dinâmica (loop de intervalos).
        Quanto maior o PNL, maior a % de lucro que protegemos.
        """
        if pnl_pct < TrailingStopHelpers.MIN_TARGET:
            return 0, "", f"📊 PNL: {pnl_pct:.2%}. Abaixo do alvo mínimo."

        # 1. Calcular em que "degrau" estamos (ex: 1.2% / 0.2% = degrau 6)
        num_steps = int(pnl_pct / TrailingStopHelpers.STEP_SIZE)
        current_step_target = num_steps * TrailingStopHelpers.STEP_SIZE

        # 2. Definir a % de proteção (Ratio) baseada na zona de lucro
        # Isto evita saíres cedo em movimentos parabólicos
        if pnl_pct < 0.01:  # Zona 1: Até 1% -> Protege 40% (Dá muita folga)
            ratio = 0.40
            icon, name = "🛡️", "SAFE_ZONE"
        elif pnl_pct < 0.03:  # Zona 2: 1% a 3% -> Protege 65% (Equilíbrio)
            ratio = 0.65
            icon, name = "💰", "TREND_ZONE"
        else:  # Zona 3: > 3% -> Protege 85% (Tranca o lucro no topo)
            ratio = 0.85
            icon, name = "🚀", "MOON_ZONE"

        # 3. Calcular o valor de segurança (secure)
        # O 'secure' é uma fatia do alvo do degrau atual
        secure = current_step_target * ratio

        log = f"{icon} [{name}] Degrau {num_steps} atingido ({current_step_target:.2%}). Stop subiu para {secure:.2%}"

        return secure, icon, log