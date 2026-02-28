from commons.utils.indicators.custom_indicators_utils import \
    CustomIndicatorsUtils
from commons.utils.indicators.tv_indicators_utils import TvIndicatorsUtils
from commons.utils.ohlcv_wrapper import OhlcvWrapper


class IndicatorsUtils(CustomIndicatorsUtils, TvIndicatorsUtils):
    def __init__(self, ohlcv: OhlcvWrapper, mode='ta'):
        super().__init__(ohlcv, mode)
        # Se as outras classes precisarem de inicialização, chamas o super()
        # ou inicializas cada uma.
