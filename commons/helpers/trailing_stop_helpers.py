class TrailingStopHelpers:
    # Configuração Única de Trailing Stop
    # Nota: Manter ordenado do maior para o menor target
    TRAILING_STRATEGY = [
        {"target": 0.02, "secure": 0.015, "icon": "🔥", "name": "META_2",
         "log": "🔥 Meta de 2% atingida! Stop subiu para garantir 1.5%"},
        {"target": 0.01, "secure": 0.006, "icon": "💰", "name": "META_1",
         "log": "💰 Meta de 1% atingida! Stop subiu para garantir 0.6%"},
        {"target": 0.004, "secure": 0.001, "icon": "🛡️", "name": "BREAK_EVEN",
         "log": "🛡️ Break-even ativo! Taxas cobertas e lucro mínimo garantido."},
        {"target": 0.002, "secure": 0.0, "icon": "🐌", "name": "LATERAL_EXIT",
         "log": "🐌 Break-even ativo! Taxas cobertas."}
    ]

    # Parâmetros de comportamento para Alavancagem
    STEP_SIZE = 0.002  # Sobe o stop a cada 0.2% de lucro (ajustável)
    MIN_TARGET = 0.003  # Só ativa o trailing após 0.3% (proteção de taxas)
    MIN_TARGET_NEW = 0.008  # Só ativa o trailing após 0.8% de lucro

    @staticmethod
    def get_trailing_adjustment_(max_pnl_pct):
        """
        max_pnl_pct: O maior lucro (PnL %) atingido desde a abertura da trade.
        Retorna o valor de lucro que deve ser garantido (secure_pnl).
        """
        if max_pnl_pct < TrailingStopHelpers.MIN_TARGET_NEW:
            return 0, "⏳", "Aguardando fôlego inicial..."

        # DEFINIÇÃO DINÂMICA DO RÁCIO DE PROTEÇÃO
        # Quanto mais lucro temos, mais "colamos" o stop ao preço
        if max_pnl_pct < 0.02:  # Até 2% de lucro
            ratio = 0.50  # Protege 50% (Dá 50% de folga para balanço)
            zone = "SAFE"
        elif max_pnl_pct < 0.05:  # De 2% a 5%
            ratio = 0.70  # Protege 70% (Dá 30% de folga)
            zone = "TREND"
        elif max_pnl_pct < 0.15:  # De 5% a 15%
            ratio = 0.85  # Protege 85% (Dá 15% de folga)
            zone = "STRONG"
        else:  # Acima de 15% (Moonshot)
            ratio = 0.95  # Protege 95% (Colado ao preço para não devolver nada)
            zone = "MOON"

        # O CÁLCULO MÁGICO: Sem degraus.
        # Se o lucro subir 0.01%, o 'secure' sobe proporcionalmente na hora.
        secure = max_pnl_pct * ratio

        log = f"🚀 [{zone}] Max PnL: {max_pnl_pct:.2%}. Stop garantido em: {secure:.2%}"

        return secure, "🎯", log

    @staticmethod
    def get_trailing_adjustment(max_pnl_pct):
        # 1. ACORDAR MAIS CEDO: Começa a proteger logo aos 0.3% ou 0.4%
        if max_pnl_pct < 0.008:
            return 0, "⏳", "Aguardando fôlego (0.4%)..."

        # 2. VERIFICAR A LISTA DE METAS (O que tu já tinhas definido)
        # Isto garante que se bateres 1% ou 0.4%, ele fixa logo um lucro
        for strategy in TrailingStopHelpers.TRAILING_STRATEGY:
            if max_pnl_pct >= strategy["target"]:
                secure = strategy["secure"]
                return secure, strategy["icon"], strategy["log"]

        # 3. LÓGICA DINÂMICA (Caso não esteja na lista)
        # Se estiver entre metas, usamos um rácio para o stop subir ponto a ponto
        ratio = 0.50
        if max_pnl_pct >= 0.01: ratio = 0.65

        secure = max_pnl_pct * ratio
        return secure, "🚀", f"Proteção dinâmica: {secure:.2%}"
