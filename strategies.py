from abc import ABC, abstractmethod
from typing import Any, Dict
import pandas as pd
from datetime import time
from indicators import calculate_ema, calculate_atr
from datetime import datetime


class Strategy(ABC):
    """Abstract base class for trading strategies."""
    NAME: str = "Base Strategy"

    @abstractmethod
    def decide(self, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'action': 'hold',
            'comment': f'{self.NAME} not implemented',
            'sl_offset': None,
            'tp_offset': None
        }

    @abstractmethod
    def get_required_bars(self) -> Dict[str, int]:
        """Returns a dict of {'timeframe_str': count} required by the strategy."""
        # Default implementation, subclasses should override if they have requirements.
        return {}

class SafeStrategy(Strategy):
    """
    Enhanced Safe (Low-Risk) Trend-Following Scalper with:
      - Volatility regime filters
      - Trend buffer zone around EMA
      - Session time filter
      - Volume spike filter
      - Trailing stop activation
    """
    NAME = "Safe (Low-Risk) Trend-Following Scalper"

    def __init__(self,
                 settings,
                 ema_period: int = 50,
                 atr_period: int = 14,
                 stop_mult: float = 1.0,
                 target_mult: float = 0.5,
                 buffer_mult: float = 0.05, #changed from 0.2 to allow more trades, revert back after testing
                 volume_mult: float = 1.5,
                 session_start: time = time(8, 0),
                 session_end: time = time(16, 0)):
        # Trend & volatility settings
        self.settings = settings
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.stop_mult = stop_mult
        self.target_mult = target_mult
        self.buffer_mult = buffer_mult
        self.volume_mult = volume_mult
        # Trading session window
        self.session_start = session_start
        self.session_end = session_end
        # Trailing stop state
        self.trailing_activated = False

    def get_required_bars(self) -> Dict[str, int]:
        return {'1m': self.settings.general.min_bars_for_trading}

    def in_session(self, timestamp: pd.Timestamp) -> bool:
        if isinstance(timestamp, (int, float)):
            timestamp = datetime.fromtimestamp(timestamp)
        t = timestamp.time()
        return self.session_start <= t <= self.session_end

    def _hold(self, reason: str) -> Dict[str, Any]:
        return {
            'action': 'hold',
            'comment': f"{self.NAME}: {reason}",
            'sl_offset': None,
            'tp_offset': None
        }

    def decide(self, data: Dict[str, Any]) -> Dict[str, Any]:
        df: pd.DataFrame = data.get('ohlc_1m')
        print("DECIDE() called - OHLC shape:", df.shape if df is not None else "None")
        if df is None or len(df) < self.settings.general.min_bars_for_trading:
            print("Returning: insufficient data")

            return self._hold("insufficient data")

        #   Session filter
        print("DEBUG - df.index:", df.index)
        print("DEBUG - df.index[-1]:", df.index[-1], type(df.index[-1]))
        
        now = df.index[-1]
        if not self.in_session(now):
            return self._hold("outside trading session")

        close = df['close']
        vol = df.get('volume', pd.Series(dtype=float))

        ema = calculate_ema(df, self.ema_period).iloc[-1]
        atr = calculate_atr(df, self.atr_period).iloc[-1]
        price = close.iloc[-1]
        avg_vol = None if vol.empty else vol.rolling(self.atr_period).mean().iloc[-1]


        # Buffer zone filter
        buffer = atr * self.buffer_mult
        if abs(price - ema) < buffer:
            return self._hold("within buffer zone")

        # Determine trade direction
        if price > ema:
            action = 'buy'
            comment = f"price {price:.5f} above EMA{self.ema_period} + buffer"
        else:
            action = 'sell'
            comment = f"price {price:.5f} below EMA{self.ema_period} - buffer"

        # Base stops
        sl = atr * self.stop_mult
        tp = atr * self.target_mult
        
        # Convert offsets from price distance to pips (1 pip = 0.0001 for most pairs)
        pip_factor = 10000  # Use 100 for JPY pairs like USDJPY
        sl_pips = sl * pip_factor
        tp_pips = tp * pip_factor

        # Trailing stop logic
        if not self.trailing_activated and (
            (action == 'buy' and price > ema + 2 * buffer) or
            (action == 'sell' and price < ema - 2 * buffer)
        ):
            self.trailing_activated = True
            comment += "; trailing stop activated"

        if self.trailing_activated:
            breakeven_offset = atr * 0.1
            prev_close = close.iloc[-2]
            if action == 'buy':
                sl = min(sl, price - (prev_close + breakeven_offset))
            else:
                sl = min(sl, (prev_close - breakeven_offset) - price)

        return {
            'action': action,
            'comment': f"{self.NAME}: {comment}",
            'sl_offset': sl_pips,
            'tp_offset': tp_pips,
            'risk_percentage': self.settings.general.risk_percentage
        }

class ModerateStrategy(Strategy):
    NAME = "Moderate Trend-Following Scalper"
    def __init__(self, settings):
        self.settings = settings
        self.ema_period = 20
        self.atr_period = 14
        self.stop_multiplier = 1.5
        self.target_multiplier = 1.0

    def get_required_bars(self) -> Dict[str, int]:
        return {'1m': self.settings.general.min_bars_for_trading}

    def decide(self, data: Dict[str, Any]) -> Dict[str, Any]:
        df: pd.DataFrame = data.get('ohlc_1m')
        if df is None or len(df) < self.settings.general.min_bars_for_trading:
            return {'action': 'hold', 'comment': f'{self.NAME}: insufficient data', 'sl_offset': None, 'tp_offset': None}

        ema = calculate_ema(df['close'], self.ema_period).iloc[-1]
        atr = calculate_atr(df, self.atr_period).iloc[-1]
        price = df['close'].iloc[-1]

        if price > ema:
            action = 'buy'
            comment = f'{self.NAME}: bullish trend detected'
        elif price < ema:
            action = 'sell'
            comment = f'{self.NAME}: bearish trend detected'
        else:
            return {'action': 'hold', 'comment': f'{self.NAME}: no clear trend', 'sl_offset': None, 'tp_offset': None}

        sl_offset = atr * self.stop_multiplier
        tp_offset = atr * self.target_multiplier
        return {'action': action, 'comment': comment, 'sl_offset': sl_offset, 'tp_offset': tp_offset}

class AggressiveStrategy(Strategy):
    NAME = "Aggressive Trend-Following Scalper"
    def __init__(self, settings):
        self.settings = settings
        self.ema_period = 10
        self.atr_period = 7
        self.stop_multiplier = 2.0
        self.target_multiplier = 1.5

    def get_required_bars(self) -> Dict[str, int]:
        return {'1m': self.settings.general.min_bars_for_trading}

    def decide(self, data: Dict[str, Any]) -> Dict[str, Any]:
        df: pd.DataFrame = data.get('ohlc_1m')
        if df is None or len(df) < self.settings.general.min_bars_for_trading:
            return {'action': 'hold', 'comment': f'{self.NAME}: insufficient data', 'sl_offset': None, 'tp_offset': None}

        ema = calculate_ema(df['close'], self.ema_period).iloc[-1]
        atr = calculate_atr(df, self.atr_period).iloc[-1]
        price = df['close'].iloc[-1]

        if price > ema:
            action = 'buy'
            comment = f'{self.NAME}: going long aggressively'
        elif price < ema:
            action = 'sell'
            comment = f'{self.NAME}: going short aggressively'
        else:
            return {'action': 'hold', 'comment': f'{self.NAME}: awaiting breakout', 'sl_offset': None, 'tp_offset': None}

        sl_offset = atr * self.stop_multiplier
        tp_offset = atr * self.target_multiplier
        return {'action': action, 'comment': comment, 'sl_offset': sl_offset, 'tp_offset': tp_offset}

class MomentumStrategy(Strategy):
    NAME = "Momentum Fade Scalper"
    def __init__(self, settings):
        self.settings = settings
        self.ema_period = 20
        self.atr_period = 14
        self.fade_threshold = 1.5  # ATR multiples
        self.stop_multiplier = 1.0
        self.target_multiplier = 1.5

    def get_required_bars(self) -> Dict[str, int]:
        return {'1m': self.settings.general.min_bars_for_trading}

    def decide(self, data: Dict[str, Any]) -> Dict[str, Any]:
        df: pd.DataFrame = data.get('ohlc_1m')
        if df is None or len(df) < self.settings.general.min_bars_for_trading:
            return {'action': 'hold', 'comment': f'{self.NAME}: insufficient data', 'sl_offset': None, 'tp_offset': None}

        ema = calculate_ema(df['close'], self.ema_period).iloc[-1]
        atr = calculate_atr(df, self.atr_period).iloc[-1]
        price = df['close'].iloc[-1]
        diff = price - ema

        if diff > atr * self.fade_threshold:
            action = 'sell'
            comment = f'{self.NAME}: fading overextension'
        elif diff < -atr * self.fade_threshold:
            action = 'buy'
            comment = f'{self.NAME}: fading downside spike'
        else:
            return {'action': 'hold', 'comment': f'{self.NAME}: no fade opportunity', 'sl_offset': None, 'tp_offset': None}

        sl_offset = atr * self.stop_multiplier
        tp_offset = atr * self.target_multiplier
        return {'action': action, 'comment': comment, 'sl_offset': sl_offset, 'tp_offset': tp_offset}

class MeanReversionStrategy(Strategy):
    NAME = "Mean-Reversion Scalper"
    def __init__(self, settings):
        self.settings = settings
        self.ema_period = 20
        self.atr_period = 14
        self.band_multiplier = 2.0  # ATR multiples
        self.stop_multiplier = 1.0
        self.target_multiplier = 2.0

    def get_required_bars(self) -> Dict[str, int]:
        return {'1m': self.settings.general.min_bars_for_trading}

    def decide(self, data: Dict[str, Any]) -> Dict[str, Any]:
        df: pd.DataFrame = data.get('ohlc_1m')
        if df is None or len(df) < self.settings.general.min_bars_for_trading:
            return {'action': 'hold', 'comment': f'{self.NAME}: insufficient data', 'sl_offset': None, 'tp_offset': None}

        ema = calculate_ema(df['close'], self.ema_period).iloc[-1]
        atr = calculate_atr(df, self.atr_period).iloc[-1]
        price = df['close'].iloc[-1]
        upper = ema + atr * self.band_multiplier
        lower = ema - atr * self.band_multiplier

        if price > upper:
            action = 'sell'
            comment = f'{self.NAME}: price above upper band'
        elif price < lower:
            action = 'buy'
            comment = f'{self.NAME}: price below lower band'
        else:
            return {'action': 'hold', 'comment': f'{self.NAME}: within bands', 'sl_offset': None, 'tp_offset': None}

        sl_offset = atr * self.stop_multiplier
        tp_offset = atr * self.target_multiplier
        return {'action': action, 'comment': comment, 'sl_offset': sl_offset, 'tp_offset': tp_offset}
