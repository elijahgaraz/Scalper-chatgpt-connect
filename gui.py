import time
import threading
import tkinter as tk
import queue
from tkinter import ttk, messagebox, simpledialog
from typing import List # Added for type hinting
import pandas as pd # Added for OHLC data handling
from trading import Trader, AiAdvice # adjust import path if needed
from strategies import (
    SafeStrategy, ModerateStrategy, AggressiveStrategy,
    MomentumStrategy, MeanReversionStrategy
)
from indicators import (
    calculate_ema, calculate_atr, calculate_rsi, calculate_adx
)

class MainApplication(tk.Tk):
    def __init__(self, settings):
        super().__init__()
        self.title("Forex Scalper")

        # make window resizable
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.settings = settings
        self.trader = Trader(self.settings)

        container = ttk.Frame(self)
        container.grid(row=0, column=0, sticky="nsew")
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        self.pages = {}
        for Page in (SettingsPage, TradingPage):
            page = Page(container, self)
            page.grid(row=0, column=0, sticky="nsew")
            self.pages[Page] = page

        self.show_page(SettingsPage)

    def show_page(self, page_cls):
        self.pages[page_cls].tkraise()


class SettingsPage(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, padding=10)
        self.controller = controller
        self.columnconfigure(0, weight=1)

        acct = ttk.Labelframe(self, text="Account Summary", padding=10)
        acct.grid(row=0, column=0, sticky="ew", pady=(0,10)) # Changed row from 1 to 0
        acct.columnconfigure(1, weight=1)

        self.account_id_var = tk.StringVar(value="–")
        ttk.Label(acct, text="Account ID:").grid(row=0, column=0, sticky="w", padx=(0,5))
        ttk.Label(acct, textvariable=self.account_id_var).grid(row=0, column=1, sticky="w")

        self.balance_var = tk.StringVar(value="–")
        ttk.Label(acct, text="Balance:").grid(row=1, column=0, sticky="w", padx=(0,5))
        ttk.Label(acct, textvariable=self.balance_var).grid(row=1, column=1, sticky="w")

        self.equity_var = tk.StringVar(value="–")
        ttk.Label(acct, text="Equity:").grid(row=2, column=0, sticky="w", padx=(0,5))
        ttk.Label(acct, textvariable=self.equity_var).grid(row=2, column=1, sticky="w")

        self.margin_var = tk.StringVar(value="–")
        ttk.Label(acct, text="Margin:").grid(row=3, column=0, sticky="w", padx=(0,5))
        ttk.Label(acct, textvariable=self.margin_var).grid(row=3, column=1, sticky="w")

        # --- Actions & Status ---
        actions = ttk.Frame(self)
        actions.grid(row=1, column=0, sticky="ew", pady=(10,0)) # Changed row from 2 to 1
        # Removed "Save Settings" button as settings are primarily config file / env var based for OpenAPI
        ttk.Button(actions, text="Connect", command=self.attempt_connection).pack(side="left", padx=5)

        self.status = ttk.Label(self, text="Disconnected", anchor="center")
        self.status.grid(row=2, column=0, sticky="ew", pady=(5,0)) # Changed row from 3 to 2

    def save_settings(self):
        
        pass

    def attempt_connection(self):
        # self.save_settings() # No longer needed as FIX settings are removed
        t = self.controller.trader

        t.settings = self.controller.settings

  
        self.status.config(text="Processing connection...", foreground="orange")

        def _connect_thread_target():
 
            if t.connect(): # This blocks, then attempts to start client service
 
                self.after(0, lambda: self.status.config(text="Connection successful. Authenticating account...", foreground="orange"))
                self.after(100, self._check_connection) # Start polling for actual connection status
            else:
                # connect() returned False. An error occurred.
                # trader._last_error should have the details of what failed.
                _, msg = t.get_connection_status()
                final_msg = f"Failed: {msg}" if msg else "Connection failed."

                self.after(0, lambda: messagebox.showerror("Connection Failed", final_msg))
                self.after(0, lambda: self.status.config(text=final_msg, foreground="red"))

        connect_thread = threading.Thread(target=_connect_thread_target, daemon=True)
        connect_thread.start()
        
    # Poll connection status until connected or error
    def _check_connection(self):
        t = self.controller.trader
        connected, msg = t.get_connection_status()
        if connected:
            # proceed to post-connection
            self._on_successful_connection(t) # Renamed
        else:
            if msg: # If there's an error message, connection attempt failed
                messagebox.showerror("Connection Failed", msg)
                self.status.config(text=f"Failed: {msg}", foreground="red")
            else: # No error message yet, still trying
                self.after(200, self._check_connection)


    def _on_successful_connection(self, t): # Renamed from _extracted_from_attempt_connection_14
        # t.start_heartbeat() # Heartbeat is typically managed by the Trader/API library after connection
        summary = t.get_account_summary()
        
        account_id_from_summary = summary.get("account_id")
        balance_from_summary = summary.get("balance")

        if account_id_from_summary == "connecting..." or \
           account_id_from_summary == "–" or \
           account_id_from_summary is None or \
           balance_from_summary is None:
            # This can happen if get_account_summary is called before trader details (like account_id)
            # are fully populated after connection and ProtoOATraderRes.
            self.status.config(text="Fetching account details...", foreground="orange") # More informative status
            self.after(300, lambda: self._on_successful_connection(t)) # Retry shortly
            return

        # Account ID
        account_id_val = summary.get("account_id", "–")
        self.account_id_var.set(str(account_id_val) if account_id_val is not None else "–")

        # Balance
        balance_val = summary.get("balance")
        self.balance_var.set(f"{balance_val:.2f}" if balance_val is not None else "–")

        # Equity
        equity_val = summary.get("equity")
        self.equity_var.set(f"{equity_val:.2f}" if equity_val is not None else "Can't retrieve equity")
        
        # Margin
        margin_val = summary.get("margin")
        self.margin_var.set(f"{margin_val:.2f}" if margin_val is not None else "–")

        # Prepare display strings for messagebox, handling None gracefully
        display_account_id = str(account_id_val) if account_id_val is not None else "N/A"
        display_balance = f"{balance_val:.2f}" if balance_val is not None else "N/A"
        display_equity = f"{equity_val:.2f}" if equity_val is not None else "N/A"
        display_margin = f"{margin_val:.2f}" if margin_val is not None else "N/A"

        messagebox.showinfo(
            "Connected",
            f"Successfully connected!\n\n"
            f"Account ID: {display_account_id}\n"
            f"Balance: {display_balance}\n" # Already handles None correctly for display_balance
            f"Equity: {display_equity}\n"   # Already handles None correctly for display_equity
            f"Margin: {display_margin}"     # Already handles None correctly for display_margin
        )
        self.status.config(text="Connected ✅", foreground="green")

        # Update TradingPage with account info
        trading_page = self.controller.pages[TradingPage]
        trading_page.update_account_info(
            account_id=summary.get("account_id", "–"),
            balance=summary.get("balance"),
            equity=summary.get("equity")
        )

        available_symbols = t.get_available_symbol_names()
        if available_symbols: # Ensure there are symbols before trying to populate
            trading_page.populate_symbols_dropdown(available_symbols)
        else:
            # If no symbols returned by trader (e.g. map empty), populate with empty/error message
            trading_page.populate_symbols_dropdown([])
            self._log_to_trading_page("Warning: No symbols received from the trader to populate dropdown.")


        self.controller.show_page(TradingPage)

    def _log_to_trading_page(self, message: str):
        """Helper to log messages to the TradingPage's output log if available."""
        if TradingPage in self.controller.pages:
            trading_page = self.controller.pages[TradingPage]
            if hasattr(trading_page, '_log'):
                trading_page._log(f"[SettingsPage] {message}") # Prefix to identify source


class TradingPage(ttk.Frame):
    # COMMON_PAIRS removed, will be populated dynamically

    def __init__(self, parent, controller):
        super().__init__(parent, padding=10)
        self.controller = controller
        self.trader = controller.trader

        # Thread-safe event queue for UI updates
        self._ui_queue = queue.Queue()
        self.after(100, self._process_ui_queue)

        self.is_scalping = False
        self.scalping_thread = None

        # Account Info StringVars
        self.account_id_var_tp = tk.StringVar(value="–")
        self.balance_var_tp = tk.StringVar(value="–")
        self.equity_var_tp = tk.StringVar(value="–")

        # configure grid
        # Adjusted row count for new account info section AND data readiness label
        for r in range(13): # Increased range for new row + data readiness
            self.rowconfigure(r, weight=0)
        self.rowconfigure(13, weight=1) # Adjusted log row index
        self.columnconfigure(1, weight=1)


        # ← Settings button
        ttk.Button(self, text="← Settings", command=lambda: controller.show_page(SettingsPage)).grid(
            row=0, column=0, columnspan=2, pady=(0,10), sticky="w" # columnspan to align with other full-width elements
        )

        # Account Info Display
        acc_info_frame = ttk.Labelframe(self, text="Account Information", padding=5)
        acc_info_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0,10))
        acc_info_frame.columnconfigure(1, weight=1)

        ttk.Label(acc_info_frame, text="Account ID:").grid(row=0, column=0, sticky="w", padx=(0,5))
        ttk.Label(acc_info_frame, textvariable=self.account_id_var_tp).grid(row=0, column=1, sticky="w")

        ttk.Label(acc_info_frame, text="Balance:").grid(row=1, column=0, sticky="w", padx=(0,5))
        ttk.Label(acc_info_frame, textvariable=self.balance_var_tp).grid(row=1, column=1, sticky="w")

        ttk.Label(acc_info_frame, text="Equity:").grid(row=2, column=0, sticky="w", padx=(0,5))
        ttk.Label(acc_info_frame, textvariable=self.equity_var_tp).grid(row=2, column=1, sticky="w")


        # Symbol dropdown
        # Row indices are +1 from original due to Account Info section added at row=1
        ttk.Label(self, text="Symbol:").grid(row=2, column=0, sticky="w", padx=(0,5))
        self.symbol_var = tk.StringVar(value="Loading symbols...") # Initial placeholder
        self.cb_symbol = ttk.Combobox(self, textvariable=self.symbol_var,
                                 values=[], state="readonly") # Initially empty
        self.cb_symbol.grid(row=2, column=1, sticky="ew") # Corrected from row=1
        self.cb_symbol.bind("<<ComboboxSelected>>", lambda e: self.refresh_price())

        # Price display + refresh
        ttk.Label(self, text="Price:").grid(row=3, column=0, sticky="w", padx=(0,5)) # Was row=2
        self.price_var = tk.StringVar(value="–")
        pf = ttk.Frame(self)
        pf.grid(row=3, column=1, sticky="ew") # Was row=2
        ttk.Label(pf, textvariable=self.price_var,
                  font=("TkDefaultFont", 12, "bold")).pack(side="left")
        ttk.Button(pf, text="↻", width=2, command=self.refresh_price).pack(side="right")

        # Profit target
        ttk.Label(self, text="Profit Target (pips):").grid(row=4, column=0, sticky="w", padx=(0,5)) # Was row=3
        self.tp_var = tk.DoubleVar(value=10.0)
        ttk.Entry(self, textvariable=self.tp_var).grid(row=4, column=1, sticky="ew") # Was row=3

        # Order size
        ttk.Label(self, text="Order Size (lots):").grid(row=5, column=0, sticky="w", padx=(0,5)) # Was row=4
        self.size_var = tk.DoubleVar(value=1.0)
        ttk.Entry(self, textvariable=self.size_var).grid(row=5, column=1, sticky="ew") # Was row=4

        # Stop-loss
        ttk.Label(self, text="Stop Loss (pips):").grid(row=6, column=0, sticky="w", padx=(0,5)) # Was row=5
        self.sl_var = tk.DoubleVar(value=5.0)
        ttk.Entry(self, textvariable=self.sl_var).grid(row=6, column=1, sticky="ew") # Was row=5

        # Batch profit target
        ttk.Label(self, text="Batch Profit Target:").grid(row=7, column=0, sticky="w", padx=(0,5))
        self.batch_profit_var = tk.DoubleVar(value=self.controller.settings.general.batch_profit_target)
        ttk.Entry(self, textvariable=self.batch_profit_var).grid(row=7, column=1, sticky="ew")

        # Strategy selector
        ttk.Label(self, text="Strategy:").grid(row=8, column=0, sticky="w", padx=(0,5))
        self.strategy_var = tk.StringVar(value="Safe")
        strategy_names = ["Safe", "Moderate", "Aggressive", "Momentum", "Mean Reversion"]
        cb_strat = ttk.Combobox(self, textvariable=self.strategy_var, values=strategy_names, state="readonly")
        cb_strat.grid(row=8, column=1, sticky="ew")
        cb_strat.bind("<<ComboboxSelected>>", lambda e: self._update_data_readiness_display(execute_now=True))


        # Data Readiness Display
        ttk.Label(self, text="Data Readiness:").grid(row=9, column=0, sticky="w", padx=(0,5), pady=(10,0))
        self.data_readiness_var = tk.StringVar(value="Initializing...")
        self.data_readiness_label = ttk.Label(self, textvariable=self.data_readiness_var)
        self.data_readiness_label.grid(row=9, column=1, sticky="ew", pady=(10,0))

        # ChatGPT Analysis Button
        self.ai_button = ttk.Button(self, text="ChatGPT Analysis", command=self.run_chatgpt_analysis)
        self.ai_button.grid(row=10, column=0, columnspan=2, pady=(10, 0))

        # Start/Stop Scalping buttons
        self.start_button = ttk.Button(self, text="Begin Scalping", command=self.start_scalping, state="normal") # Initially disabled
        self.start_button.grid(row=11, column=0, columnspan=2, pady=(10,0))
        self.stop_button  = ttk.Button(self, text="Stop Scalping", command=self.stop_scalping, state="disabled")
        self.stop_button.grid(row=12, column=0, columnspan=2, pady=(5,0))

        # Session Stats frame
        stats = ttk.Labelframe(self, text="Session Stats", padding=10)
        stats.grid(row=13, column=0, columnspan=2, sticky="ew", pady=(10,0))
        stats.columnconfigure(1, weight=1)

        self.pnl_var       = tk.StringVar(value="0.00")
        self.trades_var    = tk.StringVar(value="0")
        self.win_rate_var = tk.StringVar(value="0%")

        ttk.Label(stats, text="P&L:").grid(row=0, column=0, sticky="w", padx=(0,5))
        ttk.Label(stats, textvariable=self.pnl_var).grid(row=0, column=1, sticky="w")
        ttk.Label(stats, text="# Trades:").grid(row=1, column=0, sticky="w", padx=(0,5))
        ttk.Label(stats, textvariable=self.trades_var).grid(row=1, column=1, sticky="w")
        ttk.Label(stats, text="Win Rate:").grid(row=2, column=0, sticky="w", padx=(0,5))
        ttk.Label(stats, textvariable=self.win_rate_var).grid(row=2, column=1, sticky="w")

        # Output log
        self.output = tk.Text(self, height=8, wrap="word", state="disabled")
        self.output.grid(row=14, column=0, columnspan=2, sticky="nsew", pady=(10,0))
        sb = ttk.Scrollbar(self, command=self.output.yview)
        sb.grid(row=14, column=2, sticky="ns")
        self.output.config(yscrollcommand=sb.set)

        # Internal counters
        self.total_pnl    = 0.0
        self.total_trades = 0
        self.wins         = 0
        self.batch_size = 5
        self.current_batch_trades = 0
        self.batch_start_equity = 0.0

        # self.refresh_price() # Removed: Price will be refreshed when symbols are populated

        self.after(1000, self._update_data_readiness_display) # Start the data readiness update loop


    def _update_data_readiness_display(self, execute_now=False): # Added execute_now for immediate updates
        # Check if trader is available and connected
        if not self.trader or not hasattr(self.trader, 'is_connected') or not self.trader.is_connected:
            self.data_readiness_var.set("Trader disconnected")
            if hasattr(self, 'data_readiness_label'): # Check if label exists
                self.data_readiness_label.config(foreground="gray")
            if hasattr(self, 'start_button'):
                self.start_button.config(state="disabled")
            if not execute_now: # Schedule next call only if not an immediate execution
                self.after(2000, self._update_data_readiness_display)
            return

        selected_strategy_name = self.strategy_var.get()
        strategy_instance = None

        # Instantiate the selected strategy to get its requirements
        # This could be optimized by caching strategy instances or their requirements
        if selected_strategy_name == "Safe": strategy_instance = SafeStrategy(self.controller.settings)
        elif selected_strategy_name == "Moderate": strategy_instance = ModerateStrategy(self.controller.settings)
        elif selected_strategy_name == "Aggressive": strategy_instance = AggressiveStrategy(self.controller.settings)
        elif selected_strategy_name == "Momentum": strategy_instance = MomentumStrategy(self.controller.settings)
        elif selected_strategy_name == "Mean Reversion": strategy_instance = MeanReversionStrategy(self.controller.settings)

        if not strategy_instance:
            self.data_readiness_var.set("Select a strategy")
            if hasattr(self, 'data_readiness_label'):
                self.data_readiness_label.config(foreground="black")
            if hasattr(self, 'start_button'):
                self.start_button.config(state="disabled")
            if not execute_now:
                self.after(1000, self._update_data_readiness_display)
            return

        required_bars_map = strategy_instance.get_required_bars()
        available_bars_map = self.trader.get_ohlc_bar_counts()

        status_messages = []
        all_ready = True

        if not required_bars_map: # Strategy might have no specific bar requirements
            status_messages.append("No specific bar data required by strategy.")
            all_ready = True
        else:
            for tf_to_check, required_count in required_bars_map.items():
                available_count = available_bars_map.get(tf_to_check, 0)
                status_messages.append(f"{tf_to_check}: {available_count}/{required_count}")
                if available_count < required_count:
                    all_ready = False

        final_status_text = ", ".join(status_messages)
        current_fg_color = "black" # Default

        if all_ready:
            final_status_text += " (Ready)"
            current_fg_color = "green"
            if hasattr(self, 'start_button'):
                 self.start_button.config(state="normal" if not self.is_scalping else "disabled")
        else:
            final_status_text += " (Waiting...)"
            current_fg_color = "orange"
            if hasattr(self, 'start_button'):
                 self.start_button.config(state="disabled")

        self.data_readiness_var.set(final_status_text)
        if hasattr(self, 'data_readiness_label'):
            self.data_readiness_label.config(foreground=current_fg_color)

        if not execute_now:
            self.after(2000, self._update_data_readiness_display) # Poll every 2 seconds


    def populate_symbols_dropdown(self, symbol_names: List[str]):
        """Updates the symbol dropdown with the given list of names."""
        if not symbol_names:
            self.cb_symbol.config(values=[]) # Clear previous values if any
            self.symbol_var.set("No symbols available")
            self.price_var.set("–") # Reset price display
            return

        self.cb_symbol.config(values=symbol_names)

        configured_default = self.controller.settings.general.default_symbol # e.g., "GBPUSD"

        if configured_default in symbol_names:
            self.symbol_var.set(configured_default)
        elif symbol_names: # If default not found, but list is not empty, select first one
            self.symbol_var.set(symbol_names[0])
        else: # Should be caught by the initial 'if not symbol_names:'
            self.symbol_var.set("No symbols available")

        # Refresh price for the newly set/defaulted symbol, if it's a valid symbol string
        current_selection = self.symbol_var.get()
        if current_selection not in ["No symbols available", "Loading symbols...", ""]:
            self.refresh_price()
        else:
            self.price_var.set("–") # Ensure price is reset if no valid symbol selected


    def update_account_info(self, account_id: str, balance: float | None, equity: float | None):
        """Public method to update account info StringVars from outside (e.g., SettingsPage)."""
        self.account_id_var_tp.set(str(account_id) if account_id is not None else "–")
        self.balance_var_tp.set(f"{balance:.2f}" if balance is not None else "–")
        self.equity_var_tp.set(f"{equity:.2f}" if equity is not None else "–")
        
        # Note: TradingPage does not currently display margin, so no update for it here.

    def _process_ui_queue(self):
        """Called on the mainloop to drain the UI event queue."""
        try:
            while True:
                func, args = self._ui_queue.get_nowait()
                func(*args)
        except queue.Empty:
            pass
        finally:
            self.after(100, self._process_ui_queue)

    def refresh_price(self):
        symbol = self.symbol_var.get().replace("/", "")
        try:
            price = self.trader.get_market_price(symbol)
            if price is not None:
                self.price_var.set(f"{price:.5f}")
                self._log(f"Refreshed price for {symbol}: {price:.5f}")
            else:
                self.price_var.set("–")
                self._log(f"Price for {symbol} is currently unavailable (None).")
        except Exception as e:
            self.price_var.set("ERR")
            self._log(f"Error fetching price: {e}")

    def run_chatgpt_analysis(self):
        """Initiates the AI analysis in a separate thread to avoid freezing the GUI."""
        self._log("Requesting ChatGPT Analysis...")
        self.ai_button.config(state="disabled")

        analysis_thread = threading.Thread(target=self._chatgpt_analysis_thread, daemon=True)
        analysis_thread.start()

    def _chatgpt_analysis_thread(self):
        """The actual logic that runs in a thread for AI analysis."""
        try:
            # 1. Gather data
            symbol = self.symbol_var.get().replace("/", "")
            price = self.trader.get_market_price(symbol)
            ohlc_1m_df = self.trader.ohlc_history.get('1m', pd.DataFrame())

            if price is None or ohlc_1m_df.empty:
                self._ui_queue.put((self._show_ai_error, ("Could not perform analysis: Market data is missing.",)))
                return

            # 2. Calculate all required indicators
            # Using default periods from the C# script for this manual analysis
            fast_ema = calculate_ema(ohlc_1m_df, 9).iloc[-1]
            slow_ema = calculate_ema(ohlc_1m_df, 21).iloc[-1]
            rsi = calculate_rsi(ohlc_1m_df, 14).iloc[-1]
            atr = calculate_atr(ohlc_1m_df, 14).iloc[-1]
            adx_df = calculate_adx(ohlc_1m_df, 14)
            adx = adx_df[f'ADX_14'].iloc[-1] if not adx_df.empty else 0

            # 3. Construct payload dictionaries
            features = {
                "price_bid": price,
                "ema_fast": fast_ema,
                "ema_slow": slow_ema,
                "rsi": rsi,
                "adx": adx,
                "atr": atr,
                "spread_pips": 0 # Not easily available here, sending 0
            }
            bot_proposal = {
                "side": "n/a", # No bot proposal for manual analysis
                "sl_pips": self.sl_var.get(),
                "tp_pips": self.tp_var.get()
            }

            # 4. Call the trader's AI advice method
            # We assume 'long' intent for manual analysis; the AI should evaluate based on features regardless
            advice = self.trader.get_ai_advice(intent="long", features=features, bot_proposal=bot_proposal)

            # 5. Queue the result for display on the main thread
            if advice:
                self._ui_queue.put((self._show_ai_advice, (advice,)))
            else:
                self._ui_queue.put((self._show_ai_error, ("Failed to get advice from the AI Overseer.",)))

        except Exception as e:
            self._ui_queue.put((self._show_ai_error, (f"An error occurred during analysis: {e}",)))
        finally:
            # Re-enable the button via the UI queue
            self._ui_queue.put((lambda: self.ai_button.config(state="normal"), ()))

    def _show_ai_advice(self, advice: AiAdvice):
        """Displays the AI advice in a messagebox. Runs on the main UI thread."""
        self._log(f"ChatGPT Analysis Result: {advice.action.upper()} (Conf: {advice.confidence:.2%}) - {advice.reason}")
        messagebox.showinfo(
            "ChatGPT Analysis",
            f"Direction: {advice.action.upper()}\n"
            f"Confidence: {advice.confidence:.2%}\n\n"
            f"Reason: {advice.reason}\n\n"
            f"(Proposed SL/TP: {advice.sl_pips} / {advice.tp_pips} pips)"
        )

    def _show_ai_error(self, message: str):
        """Displays an AI-related error in a messagebox."""
        self._log(f"ChatGPT Analysis Error: {message}")
        messagebox.showerror("ChatGPT Analysis Failed", message)


    def start_scalping(self):
        self._log("start_scalping() called")

        sel = self.strategy_var.get()
        if sel == "Safe":
            strategy = SafeStrategy(self.controller.settings)
        elif sel == "Moderate":
            strategy = ModerateStrategy(self.controller.settings)
        elif sel == "Aggressive":
            strategy = AggressiveStrategy(self.controller.settings)
        elif sel == "Mean Reversion":
            strategy = MeanReversionStrategy(self.controller.settings)
        else:
            strategy = MomentumStrategy(self.controller.settings)

        self._log(f"Strategy created: {strategy.NAME}")

        symbol = self.symbol_var.get().replace("/", "")
        tp     = self.tp_var.get()
        sl     = self.sl_var.get()
        size   = self.size_var.get()

        summary = self.trader.get_account_summary()
        self.batch_start_equity = summary.get("equity", 0.0) or 0.0
        self.current_batch_trades = 0

        batch_target = self.batch_profit_var.get()

        self._toggle_scalping_ui(True)

        # Start real trading loop
        self.scalping_thread = threading.Thread(
            target=self._scalp_loop,
            args=(symbol, tp, sl, size, strategy, batch_target),
            daemon=True
        )
        self.scalping_thread.start()

        messagebox.showinfo("Scalping Started", f"Live scalping thread started for {symbol}")

    def stop_scalping(self):
        if self.is_scalping:
            self._toggle_scalping_ui(False)
            try:
                self.trader.close_all_positions()
            except Exception as e:
                self._log(f"Error closing positions: {e}")

    def _toggle_scalping_ui(self, on: bool):
        self.is_scalping = on
        state_start = "disabled" if on else "normal"
        state_stop  = "normal"   if on else "disabled"
        self.start_button.config(state=state_start)
        self.stop_button.config(state=state_stop)

    # gui.py

    def start_scalping(self):
        self._log("start_scalping() called")

        # GET THE STRATEGY NAME, NOT THE OBJECT
        strategy_name = self.strategy_var.get() 
        self._log(f"Selected Strategy: {strategy_name}")

        symbol = self.symbol_var.get().replace("/", "")
        tp     = self.tp_var.get()
        sl     = self.sl_var.get()
        size   = self.size_var.get()

        summary = self.trader.get_account_summary()
        self.batch_start_equity = summary.get("equity", 0.0) or 0.0
        self.current_batch_trades = 0

        batch_target = self.batch_profit_var.get()

        self._toggle_scalping_ui(True)

        # Start real trading loop
        self.scalping_thread = threading.Thread(
            target=self._scalp_loop,
            # PASS THE NAME OF THE STRATEGY, NOT THE OBJECT ITSELF
            args=(symbol, tp, sl, size, strategy_name, batch_target),
            daemon=True
        )
        self.scalping_thread.start()

        messagebox.showinfo("Scalping Started", f"Live scalping thread started for {symbol}")

    def _scalp_loop(self, symbol: str, tp: float, sl: float, size: float, strategy_name: str, batch_target: float):
        print("SCALP LOOP STARTED")
        while self.is_scalping:
            # THIS IS THE KEY CHANGE: CREATE A NEW STRATEGY OBJECT IN EVERY LOOP
            if strategy_name == "Safe":
                strategy = SafeStrategy(self.controller.settings)
            elif strategy_name == "Moderate":
                strategy = ModerateStrategy(self.controller.settings)
            elif strategy_name == "Aggressive":
                strategy = AggressiveStrategy(self.controller.settings)
            elif strategy_name == "Mean Reversion":
                strategy = MeanReversionStrategy(self.controller.settings)
            else: # Momentum
                strategy = MomentumStrategy(self.controller.settings)

            if self.current_batch_trades >= self.batch_size:
                summary = self.trader.get_account_summary()
                equity = summary.get("equity", 0.0) or 0.0
                if equity - self.batch_start_equity >= batch_target:
                    self._ui_queue.put((self._log, ("Batch profit target reached. Closing positions.",)))
                    try:
                        self.trader.close_all_positions()
                    except Exception as e:
                        self._ui_queue.put((self._log, (f"Error closing positions: {e}",)))
                    self.batch_start_equity = equity
                    self.current_batch_trades = 0
                else:
                    time.sleep(1)
                    continue

            print("Fetching tick price...")
            current_tick_price = self.trader.get_market_price(symbol)
            print(f"Tick price: {current_tick_price}")

            # Fetch OHLC
            ohlc_1m_df = self.trader.ohlc_history.get('1m', pd.DataFrame())
            
            if (
                isinstance(ohlc_1m_df, pd.DataFrame)
                and not ohlc_1m_df.empty
                and 'timestamp' in ohlc_1m_df.columns
                and isinstance(ohlc_1m_df.index, pd.RangeIndex)
            ):
                try:
                    ohlc_1m_df['timestamp'] = pd.to_datetime(ohlc_1m_df['timestamp'], utc=True)
                    ohlc_1m_df.set_index('timestamp', inplace=True)
                except Exception as e:
                    print(f"Error during timestamp normalization: {e}")

            # Strategy decision
            action_details = strategy.decide({
                'ohlc_1m': ohlc_1m_df,
                'ohlc_15s': self.trader.ohlc_history.get('15s', pd.DataFrame()),
                'current_equity': self.trader.equity,
                'pip_position': None,
                'current_price_tick': current_tick_price
            }, self.trader)

            print(f"Strategy decision: {action_details}")

            if action_details and isinstance(action_details, dict):
                trade_action = action_details.get('action')
                if trade_action in ("buy", "sell"):
                    sl_offset = action_details.get('sl_offset')
                    tp_offset = action_details.get('tp_offset')
                    comment = action_details.get('comment', '')

                    self._ui_queue.put((self._log, (f"Strategy signal: {trade_action.upper()} for {symbol}. {comment}",)))
                    self._ui_queue.put((
                        self._execute_trade,
                        (trade_action, symbol, current_tick_price, size, tp, sl, sl_offset, tp_offset, comment)
                    ))
                else:
                    comment = action_details.get('comment', "Strategy returned HOLD or no action.")
                    self._ui_queue.put((self._log, (comment,)))
            else:
                self._ui_queue.put((self._log, ("Strategy did not return a valid action dictionary.",)))

            time.sleep(1)
   
    def _execute_trade(self,
                       side: str,
                       symbol: str,
                       price: float, # This is current_tick_price
                       size: float,
                       tp_pips_gui: float, # Original TP in pips from GUI
                       sl_pips_gui: float, # Original SL in pips from GUI
                       # Parameters from strategy's decision dictionary:
                       sl_offset_strategy: float | None,
                       tp_offset_strategy: float | None,
                       strategy_comment: str):
        """Runs on the Tk mainloop—safe to update UI."""
        price_str = f"{price:.5f}" if price is not None else "N/A (unknown)"
        sl = sl_pips_gui
        tp = tp_pips_gui
        self._log(f"{side.upper()} scalp: {symbol} at {price_str} | "
                  f"size={size} lots | SL={sl} pips | TP={tp} pips")

        if price is None:
            self._log("Trade execution skipped: Market price is unavailable.")
            return

        final_tp_pips = tp_offset_strategy if tp_offset_strategy is not None else tp_pips_gui
        final_sl_pips = sl_offset_strategy if sl_offset_strategy is not None else sl_pips_gui

        self._log(f"Attempting to place market order: {side.upper()} {size} lots of {symbol} at market price.")
        if final_tp_pips is not None:
            self._log(f"  with TP: {final_tp_pips} pips")
        if final_sl_pips is not None:
            self._log(f"  with SL: {final_sl_pips} pips")
        if strategy_comment:
            self._log(f"  Strategy comment: {strategy_comment}")

        success, message = self.trader.place_market_order(
            symbol_name=symbol,
            volume_lots=size,
            side=side,
            take_profit_pips=final_tp_pips,
            stop_loss_pips=final_sl_pips
            # client_msg_id could be generated here if needed for GUI-specific tracking
        )

        if success:
            self._log(f"Order request successful: {message}")
            self.total_trades += 1
            self.trades_var.set(str(self.total_trades))
            self.current_batch_trades += 1

        else:
            self._log(f"Order request failed: {message}")
 

    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.output.configure(state="normal")
        self.output.insert("end", f"[{ts}] {msg}\n")
        self.output.see("end")
        self.output.configure(state="disabled")


if __name__ == "__main__":
    import settings
    app = MainApplication(settings.Settings.load())
    app.mainloop()
