from collections import defaultdict
from typing import Dict, List

from commons.models.trade_snapashot_dclass import TradeSnapshot


class TradeFeaturesMemory:
    NUMERIC_FIELDS = [
        'rsi', 'stochastic', 'adx', 'macd', 'cci',
        'weights_trend', 'weights_momentum', 'weights_divergence', 'weights_oscillators', 
        'weights_price_action', 'weights_price_levels', 'volume_ratio', 'atr_ratio',
        'penalty_exhaustion', 'penalty_factor', 'penalty_manipulation', 'penalty_confirmation_candle',
        'weights_channel_position'
    ]


    def __init__(self):
        # Agora uma lista de snapshots temporários por trade_id
        self._temp_snapshots: Dict[str, List[TradeSnapshot]] = defaultdict(list)
        self._profitable_snapshots: List[TradeSnapshot] = []

    def add_trade_snapshot(self, trade_id: str, snapshot: TradeSnapshot):
        self._temp_snapshots[trade_id].append(snapshot)

    def finalize_trade(self, trade_id: str, profitable: bool):
        snapshots = self._temp_snapshots.pop(trade_id, [])
        if profitable and snapshots:
            # Exemplo: guardar só o último snapshot
            self._profitable_snapshots.append(snapshots[-1])
            # Ou podes guardar a média de snapshots (se fizer sentido para ti)
            # avg_snap = self._average_snapshots(snapshots)
            # self._profitable_snapshots.append(avg_snap)

    def _average_snapshots(self, snapshots: List[TradeSnapshot]) -> TradeSnapshot:
        # Método para calcular a média dos campos numéricos de vários snapshots
        count = len(snapshots)
        if count == 0:
            raise ValueError("No snapshots to average")

        avg_values = {}
        for field in self.NUMERIC_FIELDS:
            avg_values[field] = sum(getattr(snap, field) for snap in snapshots) / count

        # Para campos não numéricos, como symbol, candle_type e timestamp, podes escolher o primeiro, último ou algum padrão
        return TradeSnapshot(
            symbol=snapshots[-1].symbol,
            candle_type=snapshots[-1].candle_type,
            rsi=avg_values['rsi'],
            stochastic=avg_values['stochastic'],
            adx=avg_values['adx'],
            macd=avg_values['macd'],
            cci=avg_values['cci'],
            weights_trend=avg_values['weights_trend'],
            weights_momentum=avg_values['weights_momentum'],
            weights_divergence=avg_values['weights_divergence'],
            weights_oscillators=avg_values['weights_oscillators'],
            weights_price_action=avg_values['weights_price_action'],
            weights_price_levels=avg_values['weights_price_levels'],
            weights_channel_position=avg_values['weights_channel_position'], 
            penalty_exhaustion=avg_values['penalty_exhaustion'],
            penalty_factor=avg_values['penalty_factor'],
            penalty_manipulation=avg_values['penalty_manipulation'],
            penalty_confirmation_candle=avg_values['penalty_confirmation_candle'],
            volume_ratio=avg_values['volume_ratio'],
            atr_ratio=avg_values['atr_ratio'],
            timestamp=snapshots[-1].timestamp,
            signal=snapshots[-1].signal,
            entry_price=snapshots[-1].entry_price,
            sl=snapshots[-1].sl,
            tp=snapshots[-1].tp,
            size=snapshots[-1].size,
        )
    
    def get_temp_snapshots(self) -> Dict[str, List[TradeSnapshot]]:
        return self._temp_snapshots

    def get_profitable_snapshots(self) -> List[TradeSnapshot]:
        return self._profitable_snapshots

    def average_features(self) -> Dict[str, float]:
        if not self._profitable_snapshots:
            return {}

        sums = {field: 0.0 for field in self.NUMERIC_FIELDS}
        count = len(self._profitable_snapshots)

        for snap in self._profitable_snapshots:
            for field in self.NUMERIC_FIELDS:
                sums[field] += getattr(snap, field)

        averages = {field: sums[field] / count for field in self.NUMERIC_FIELDS}
        return averages
    
    def get_last_temp_snapshot(self, trade_id: str) -> TradeSnapshot | None:
        snapshots = self._temp_snapshots.get(trade_id)
        if snapshots:
            return snapshots[-1]
        return None

    
    def remove_temp_snapshot(self, trade_id: str):
        self._temp_snapshots.pop(trade_id, None)