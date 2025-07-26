from collections import defaultdict
from typing import Dict, List

from commons.models.trade_snapashot import TradeSnapshot


class TradeFeaturesMemory:
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

        numeric_fields = [
            'rsi', 'stochastic', 'adx', 'macd', 'cci',
            'proximity_to_bands', 'price_action', 'exhaustion_score',
            'divergence_score', 'volume_ratio', 'atr_ratio'
        ]

        avg_values = {}
        for field in numeric_fields:
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
            trend=avg_values['trend'],
            momentum=avg_values['momentum'],
            divergence=avg_values['divergence'],
            oscillators=avg_values['oscillators'],
            price_action=avg_values['price_action'],
            price_levels=avg_values['price_levels'],
            volume_ratio=avg_values['volume_ratio'],
            atr_ratio=avg_values['atr_ratio'],
            timestamp=snapshots[-1].timestamp,
        )

    def get_profitable_snapshots(self) -> List[TradeSnapshot]:
        return self._profitable_snapshots

    def average_features(self) -> Dict[str, float]:
        if not self._profitable_snapshots:
            return {}

        numeric_fields = [
            'rsi', 'stochastic', 'adx', 'macd', 'cci',
            'proximity_to_bands', 'price_action', 'exhaustion_score',
            'divergence_score', 'volume_ratio', 'atr_ratio'
        ]

        sums = {field: 0.0 for field in numeric_fields}
        count = len(self._profitable_snapshots)

        for snap in self._profitable_snapshots:
            for field in numeric_fields:
                sums[field] += getattr(snap, field)

        averages = {field: sums[field] / count for field in numeric_fields}
        return averages