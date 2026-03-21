from commons.helpers.trailing_stop_helpers import TrailingStopHelpers


class ProfitManager:
    def __init__(self):
        self.max_pnl_seen = {}
        self.active_trailing_levels = {}

    def update_and_check(self, symbol, pnl_pct):
        """
        Calcula se devemos fechar por recuo (Momentum)
        ou se devemos subir o patamar do SL (Trailing).
        """
        # 1. Atualizar o Pico de Lucro (ATH)
        if symbol not in self.max_pnl_seen:
            self.max_pnl_seen[symbol] = 0

        if pnl_pct > self.max_pnl_seen[symbol]:
            self.max_pnl_seen[symbol] = pnl_pct

        max_p = self.max_pnl_seen[symbol]

        # --- LÓGICA A: FECHO POR RECUO (MOMENTUM) ---
        # Se lucro > 0.6% e caiu 0.25% desde o pico -> SAÍDA MERCADO
        should_market_close = False
        if max_p >= 0.006:
            recoil = max_p - pnl_pct
            if recoil >= 0.0025:
                should_market_close = True

        # --- LÓGICA B: SUBIDA DE PATAMAR (TRAILING) ---
        # Usa o teu Helper que já criámos
        adjustment, icon, log_msg = TrailingStopHelpers.get_trailing_adjustment(pnl_pct)
        last_applied = self.active_trailing_levels.get(symbol, 0)

        should_update_trailing = False
        if adjustment > last_applied:
            should_update_trailing = True
            self.active_trailing_levels[symbol] = adjustment

        return {
            "should_market_close": should_market_close,
            "should_update_trailing": should_update_trailing,
            "adjustment": adjustment,
            "max_pnl": max_p,
            "log_msg": log_msg if should_update_trailing else None
        }

    def clear(self, symbol):
        """Limpa a memória quando a posição fecha"""
        self.max_pnl_seen.pop(symbol, None)
        self.active_trailing_levels.pop(symbol, None)