from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class ClosedOrder:
    id: str
    clientOrderId: Optional[Union[str, None]]
    timestamp: Optional[Union[int, None]]
    datetime: Optional[Union[str, None]]
    symbol: str
    type: Optional[Union[str, None]]
    side: str
    price: float
    amount: Optional[Union[float, None]]
    reduceOnly: bool
    orderType: Optional[Union[str, None]]






"""


{'info': {'order': {'coin': 'BTC', 'side': 'B', 'limitPx': '117500.0', 'sz': '0.0', 'oid': '36454403129', 'timestamp': '1753463714771', 
'triggerCondition': 'N/A', 'isTrigger': False, 'triggerPx': '0.0', 'children': [], 'isPositionTpsl': False, 
'reduceOnly': False, 'orderType': 'Limit', 'origSz': '0.00969', 'tif': 'Ioc', 'cloid': None}, 
'status': 'filled', 'statusTimestamp': '1753463714771'}, 

"""
 