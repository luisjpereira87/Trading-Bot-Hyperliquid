class TrailingStopHelpers:

    # Configuração Única de Trailing Stop
    # Nota: Manter ordenado do maior para o menor target
    TRAILING_STRATEGY = [
        {"target": 0.02,  "secure": 0.015, "icon": "🔥", "name": "META_2", "log": "🔥 Meta de 2% atingida! Stop subiu para garantir 1.5%"},
        {"target": 0.01,  "secure": 0.006, "icon": "💰", "name": "META_1", "log": "💰 Meta de 1% atingida! Stop subiu para garantir 0.6%"},
        {"target": 0.004, "secure": 0.001, "icon": "🛡️", "name": "BREAK_EVEN", "log": "🛡️ Break-even ativo! Taxas cobertas e lucro mínimo garantido."},
        {"target": 0.002, "secure": 0.0, "icon": "🐌", "name": "LATERAL_EXIT", "log": "🐌 Break-even ativo! Taxas cobertas."}
    ]

    @staticmethod
    def get_trailing_adjustment(pnl_pct):
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