from commons.enums.signal_enum import Signal


class RiskUtils:

    @staticmethod
    def adjust_sl_tp(
        signal, sl, tp, price_ref, atr, support_levels=None, resistance_levels=None,
        rr_min: float = 1.5, atr_multiplier: float = 0.5, tp_multiplier: float = 2.0
    ) -> tuple[float | None, float | None]:
        """
        Ajusta e valida SL/TP com base no ATR e suportes/resistências.
        Garante que SL/TP estão na direção correta e não muito colados ao preço.
        """

        # --- Ajuste mínimo baseado no ATR ---
        min_distance = atr_multiplier * atr
        

        if signal == Signal.BUY:
            # SL mínimo
            if sl is not None and price_ref - sl < min_distance:
                sl = price_ref - min_distance
            # TP mínimo
            if tp is not None and tp - price_ref < min_distance:
                tp = price_ref + tp_multiplier * min_distance

            # --- Validação contra resistências ---
            if resistance_levels:
                nearest_res = min([r for r in resistance_levels if r > price_ref], default=None)
                if nearest_res and tp > nearest_res:
                    tp = nearest_res * 0.995  # margem ligeira abaixo da resistência

            # --- Garantir TP acima do preço ---
            if tp is not None and tp <= price_ref:
                tp = price_ref + min_distance

        elif signal == Signal.SELL:
            # SL mínimo
            if sl is not None and sl - price_ref < min_distance:
                sl = price_ref + min_distance
            # TP mínimo
            if tp is not None and price_ref - tp < min_distance:
                tp = price_ref - tp_multiplier * min_distance

            # --- Validação contra suportes ---
            if support_levels:
                nearest_sup = max([s for s in support_levels if s < price_ref], default=None)
                if nearest_sup and tp < nearest_sup:
                    tp = nearest_sup * 1.005  # margem ligeira acima do suporte

            # --- Garantir TP abaixo do preço ---
            if tp is not None and tp >= price_ref:
                tp = price_ref - min_distance

        # --- Filtro final: TP muito colado ao preço é ignorado ---
        if tp is None or (abs(tp - price_ref) / price_ref < 0.002):
            return None, None
        
        
        if sl is not None and tp is not None:
            if signal == Signal.BUY:
                max_sl_distance = (tp - price_ref) * rr_min
                if price_ref - sl > max_sl_distance:
                    sl = price_ref - max_sl_distance
            elif signal == Signal.SELL:
                max_sl_distance = (price_ref - tp) * rr_min
                if sl - price_ref > max_sl_distance:
                    sl = price_ref + max_sl_distance


        return sl, tp
    
    @staticmethod
    def adjust_sl_tp_3(
        signal, sl, tp, price_ref, atr, support_levels=None, resistance_levels=None,
        rr_min: float = 1.5, atr_multiplier: float = 0.5
    ) -> tuple[float | None, float | None]:
        """
        Ajusta e valida SL/TP com base no ATR, suportes/resistências e Risk/Reward.
        - SL é garantido a pelo menos min_distance do preço.
        - TP é ajustado dinamicamente com base no risco (RR mínimo).
        """

        # --- Ajuste mínimo baseado no ATR ---
        min_distance = atr_multiplier * atr

        if signal == Signal.BUY:
            # SL mínimo
            if sl is not None and price_ref - sl < min_distance:
                sl = price_ref - min_distance

            # TP dinâmico baseado no RR
            if sl is not None and (tp is None or tp - price_ref < min_distance):
                risk = price_ref - sl
                tp = price_ref + min(rr_min * risk, min_distance)
            elif tp is None:  # fallback caso não haja SL
                tp = price_ref + 2 * min_distance

            # --- Validação contra resistências ---
            if resistance_levels:
                nearest_res = min([r for r in resistance_levels if r > price_ref], default=None)
                if nearest_res and tp > nearest_res:
                    tp = nearest_res * 0.995  # margem ligeira abaixo da resistência

            # --- Garantir TP acima do preço ---
            if tp is not None and tp <= price_ref:
                tp = price_ref + min_distance

        elif signal == Signal.SELL:
            # SL mínimo
            if sl is not None and sl - price_ref < min_distance:
                sl = price_ref + min_distance

            # TP dinâmico baseado no RR
            if sl is not None and (tp is None or price_ref - tp < min_distance):
                risk = sl - price_ref
                tp = price_ref - min(rr_min * risk, min_distance)
            elif tp is None:  # fallback caso não haja SL
                tp = price_ref - 2 * min_distance

            # --- Validação contra suportes ---
            if support_levels:
                nearest_sup = max([s for s in support_levels if s < price_ref], default=None)
                if nearest_sup and tp < nearest_sup:
                    tp = nearest_sup * 1.005  # margem ligeira acima do suporte

            # --- Garantir TP abaixo do preço ---
            if tp is not None and tp >= price_ref:
                tp = price_ref - min_distance

        # --- Filtro final: TP muito colado ao preço é ignorado ---
        if tp is None or (abs(tp - price_ref) / price_ref < 0.002):
            return None, None

        return sl, tp
    

    @staticmethod
    def adjust_sl_tp_2(
        signal, sl, tp, price_ref, atr, support_levels=None, resistance_levels=None,
        rr_min: float = 1.5, atr_multiplier: float = 1.0
    ) -> tuple[float | None, float | None, float | None]:
        """
        Ajusta e valida SL/TP com base no ATR e suportes/resistências.
        Garante que SL/TP estão na direção correta e não muito colados ao preço.
        Retorna também o Risk/Reward ratio (RR).
        """

        # --- Ajuste mínimo baseado no ATR ---
        min_distance = atr_multiplier * atr
        rr = None

        if signal == Signal.BUY:
            # SL mínimo
            if sl is not None and price_ref - sl < min_distance:
                sl = price_ref - min_distance
            # TP mínimo
            if tp is not None and tp - price_ref < min_distance:
                tp = price_ref + 2 * min_distance

            # --- Validação contra resistências ---
            if resistance_levels:
                nearest_res = min([r for r in resistance_levels if r > price_ref], default=None)
                if nearest_res and tp > nearest_res:
                    tp = nearest_res * 0.995

            # --- Garantir TP acima do preço ---
            if tp is not None and tp <= price_ref:
                tp = price_ref + min_distance

        elif signal == Signal.SELL:
            # SL mínimo
            if sl is not None and sl - price_ref < min_distance:
                sl = price_ref + min_distance
            # TP mínimo
            if tp is not None and price_ref - tp < min_distance:
                tp = price_ref - 2 * min_distance

            # --- Validação contra suportes ---
            if support_levels:
                nearest_sup = max([s for s in support_levels if s < price_ref], default=None)
                if nearest_sup and tp < nearest_sup:
                    tp = nearest_sup * 1.005

            # --- Garantir TP abaixo do preço ---
            if tp is not None and tp >= price_ref:
                tp = price_ref - min_distance

        # --- Calcular Risk/Reward ratio ---
        if sl and tp:
            risk = abs(price_ref - sl)
            reward = abs(tp - price_ref)
            if risk > 0:
                rr = reward / risk

        # --- Filtro final: rejeita se TP muito colado ou RR insuficiente ---
        if tp is None or (abs(tp - price_ref) / price_ref < 0.002) or (rr is not None and rr < rr_min):
            return None, None, None

        return sl, tp, rr