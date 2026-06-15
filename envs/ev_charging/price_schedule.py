import numpy as np

MAX_PRICE = 190.4


def get_price_schedule():
    """
    24시간(144 타임스텝) 전기 요금 스케줄 (원/kWh)
    한국 전기차 충전전력요금 고압 + 여름철 기준

    경부하 (22:00~08:00): 79.2원/kWh
    중간부하 (08:00~11:00, 12:00~13:00, 18:00~22:00): 137.4원/kWh
    최대부하 (11:00~12:00, 13:00~18:00): 190.4원/kWh
    """
    prices = np.zeros(144)
    for t in range(144):
        hour = (t * 10) // 60
        if 22 <= hour or hour < 8:
            prices[t] = 79.2
        elif (11 <= hour < 12) or (13 <= hour < 18):
            prices[t] = 190.4
        else:
            prices[t] = 137.4
    return prices
