import random

from commons.enums.mode_enum import ModeEnum


class GridSearchParams:


    # Lista refinada com os valores que forneceste
    param_grid_refinado = {
        'mode': [ModeEnum.CONSERVATIVE, ModeEnum.AGGRESSIVE],
        'multiplier': [0.5, 1.0, 1.5, 2.0],  # adicionado 0.5 e 1.5
        'adx_threshold': [15, 20, 25, 30, 35, 40],  # adicionados 15 e 40
        'rsi_buy_threshold': [20, 25, 30, 35, 40],  # adicionados 20 e 40
        'rsi_sell_threshold': [60, 65, 70, 75, 80],  # adicionados 60 e 80
        'weights_trend': [0.3, 0.4, 0.5, 0.6],  # mais valores entre 0.3 e 0.5
        'weights_rsi': [0.0, 0.1, 0.2, 0.3, 0.4],  # adicionados 0.1 e 0.3
        'weights_stochastic': [0.4, 0.6, 0.8, 1.0],  # adicionados 0.4 e 1.0
        'weights_price_action': [0.4, 0.6, 0.8],  # adicionados 0.4 e 0.8
        'weights_proximity_to_bands': [0.1, 0.2, 0.3, 0.4, 0.6],  # adicionado 0.3
        'sl_multiplier_aggressive': [1.0, 1.5, 2.0, 2.5],  # adicionado 1.0 e 2.5
        'tp_multiplier_aggressive': [2.5, 3.0, 4.0, 5.0, 6.0],  # adicionado 2.5 e 6.0
        'sl_multiplier_conservative': [1.5, 2.0, 2.5],  # adicionado 1.5 e 2.5
        'tp_multiplier_conservative': [2.5, 3.0, 4.0, 5.0],  # adicionado 2.5 e 4.0
        'volume_threshold_ratio': [0.3, 0.4, 0.5, 0.6, 0.7],  # mais granular
        'atr_threshold_ratio': [0.4, 0.5, 0.6, 0.7],  # mais granular
    }

    param_grid_refinado1 = {
        'mode': [ModeEnum.AGGRESSIVE],  # foco no modo agressivo, que se mostrou melhor
        'multiplier': [1.0, 1.25, 1.5],  # testar variações pequenas do multiplier
        'adx_threshold': [30, 35, 40],  # valores médios-altos
        'rsi_buy_threshold': [30, 32, 35],  # faixa estreita para buy
        'rsi_sell_threshold': [60, 65, 70],  # faixa estreita para sell
        'weights_trend': [0.05, 0.1, 0.15],  # pesos pequenos para tendência
        'weights_rsi': [0.1, 0.15, 0.2],  # explorar peso médio para RSI
        'weights_stochastic': [0.2, 0.25, 0.3],  # peso forte, como indicado
        'weights_price_action': [0.15, 0.2, 0.25],  # peso relevante, mas variado
        'weights_proximity_to_bands': [0.05, 0.1],  # peso baixo
        'sl_multiplier_aggressive': [1.8, 2.0, 2.2],  # SL próximo do melhor, ligeira variação
        'tp_multiplier_aggressive': [4.5, 5.0, 5.5],  # TP perto do melhor achado
        'sl_multiplier_conservative': [1.5],  # mantemos fixo para agora
        'tp_multiplier_conservative': [2.5],  # idem
        'volume_threshold_ratio': [0.55, 0.6, 0.65],  # filtragem boa para liquidez
        'atr_threshold_ratio': [0.65, 0.7, 0.75],  # filtragem para volatilidade
    }

    param_grid_refinado2 = {
        'mode': [ModeEnum.AGGRESSIVE],  # modo agressivo é claramente melhor
        'multiplier': [1.0, 1.1, 1.25, 1.4],  # em torno do 1.25 ótimo
        'adx_threshold': [35, 38, 40, 42],  # perto do 40 vencedor
        'rsi_buy_threshold': [32, 34, 35, 37],  # valores perto do 35
        'rsi_sell_threshold': [60, 62, 65, 68],  # em torno do 65
        'weights_trend': [0.12, 0.14, 0.15, 0.17],  # próximos valores dos pesos bons
        'weights_rsi': [0.18, 0.19, 0.20, 0.21], 
        'weights_stochastic': [0.23, 0.24, 0.25, 0.26], 
        'weights_price_action': [0.14, 0.15, 0.16, 0.17], 
        'weights_proximity_to_bands': [0.08, 0.09, 0.10], 
        'sl_multiplier_aggressive': [1.6, 1.7, 1.8, 1.9, 2.0],  # perto do 1.8 ótimo
        'tp_multiplier_aggressive': [5.0, 5.2, 5.5, 5.7, 6.0],  # próximo de 5.5 vencedor
        'sl_multiplier_conservative': [1.4, 1.5, 1.6],  # manter pequeno teste aqui
        'tp_multiplier_conservative': [2.0, 2.5, 3.0], 
        'volume_threshold_ratio': [0.60, 0.62, 0.65, 0.67], 
        'atr_threshold_ratio': [0.65, 0.68, 0.70, 0.73], 
    }

    @staticmethod
    def get_random_param_samples(n_samples=20, seed=None):
        if seed is not None:
            random.seed(seed)

        keys = list(GridSearchParams.param_grid_refinado.keys())

        peso_keys = [
            'weights_trend',
            'weights_rsi',
            'weights_stochastic',
            'weights_price_action',
            'weights_proximity_to_bands'
            'weights_exhaustion'
        ]

        for _ in range(n_samples):
            sample = {}
            # 1. Selecionar valores dos parâmetros normais
            for k in keys:
                if k not in peso_keys:
                    sample[k] = random.choice(GridSearchParams.param_grid_refinado[k])

            # 2. Gerar pesos aleatórios que somam 1
            random_weights = [random.random() for _ in peso_keys]
            total = sum(random_weights)
            normalized_weights = [w / total for w in random_weights]

            for k, w in zip(peso_keys, normalized_weights):
                sample[k] = w

            yield sample