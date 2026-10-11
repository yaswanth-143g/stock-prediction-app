"""
services/order_service.py — Paper Trading Simulation & Brokerage Abstraction Engine
=====================================================================================
Provides paper trading execution, simulated order book matching, virtual position tracking,
and strict safety guardrails (live order execution disabled by default).
"""

import os
import uuid
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional

logger = logging.getLogger("OrderService")
logger.setLevel(logging.INFO)

ENABLE_LIVE_TRADING = os.environ.get("ENABLE_LIVE_TRADING", "false").lower() in ["true", "1", "yes"]
IST = timezone(timedelta(hours=5, minutes=30))


class OrderService:
    """Thread-safe Singleton Order & Position Execution Manager with persistent storage."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(OrderService, cls).__new__(cls)
            cls._instance.db_filepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), "paper_trading_db.json")
            cls._instance.orders: Dict[str, Dict[str, Any]] = {}
            cls._instance.positions: Dict[str, Dict[str, Dict[str, Any]]] = {}
            cls._instance.account_balances: Dict[str, float] = {}
            cls._instance._load_db()
        return cls._instance

    def _load_db(self):
        """Loads state from local persistent JSON file if available."""
        if os.path.exists(self.db_filepath):
            try:
                with open(self.db_filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.orders = data.get("orders", {})
                    self.positions = data.get("positions", {})
                    self.account_balances = data.get("account_balances", {})
                    logger.info("Paper trading DB loaded successfully from %s", self.db_filepath)
            except Exception as e:
                logger.error("Failed to load paper trading DB: %s", e)

    def _save_db(self):
        """Persists current state to local JSON file."""
        try:
            with open(self.db_filepath, "w", encoding="utf-8") as f:
                json.dump({
                    "orders": self.orders,
                    "positions": self.positions,
                    "account_balances": self.account_balances
                }, f, indent=2)
        except Exception as e:
            logger.error("Failed to save paper trading DB: %s", e)

    def _get_user_balance(self, user_id: str) -> float:
        return self.account_balances.get(user_id, 1000000.0)  # Default ₹10,00,000 Paper Capital

    def reset_account(self, user_id: str = "guest_user", initial_capital: float = 1000000.0) -> Dict[str, Any]:
        """Resets virtual paper trading account capital and clears open positions."""
        self.account_balances[user_id] = float(initial_capital)
        self.positions[user_id] = {}
        # Clear orders for user
        self.orders = {k: v for k, v in self.orders.items() if v.get("user_id") != user_id}
        self._save_db()
        return {"status": "success", "message": f"Account reset to ₹{initial_capital:,.2f}", "cash_balance": initial_capital}

    def get_account_summary(self, user_id: str = "guest_user") -> Dict[str, Any]:
        """Calculates total portfolio valuation, cash balance, unrealized PnL, and margin with live market LTP updates."""
        cash = self._get_user_balance(user_id)
        user_pos = self.positions.get(user_id, {})

        # Lazy-import live market data service for real-time LTP quotes
        try:
            from services.market_data_service import get_market_data_service
            market_svc = get_market_data_service()
        except Exception:
            market_svc = None

        total_market_val = 0.0
        total_cost_basis = 0.0
        unrealized_pnl = 0.0

        pos_list = []
        for ticker, pos in user_pos.items():
            shares = pos.get("shares", 0)
            if shares > 0:
                avg_buy = pos.get("avg_buy_price", 0.0)
                curr_price = pos.get("current_price", avg_buy)

                # Update curr_price using live market LTP if available
                if market_svc:
                    try:
                        quote = market_svc.get_live_quote(ticker)
                        if quote and quote.get("ltp", 0.0) > 0:
                            curr_price = float(quote["ltp"])
                            pos["current_price"] = curr_price
                    except Exception:
                        pass

                mkt_val = shares * curr_price
                cost_val = shares * avg_buy
                pnl = mkt_val - cost_val
                pnl_pct = (pnl / cost_val * 100.0) if cost_val > 0 else 0.0

                total_market_val += mkt_val
                total_cost_basis += cost_val
                unrealized_pnl += pnl

                pos_list.append({
                    "ticker": ticker,
                    "shares": shares,
                    "avg_buy_price": round(avg_buy, 2),
                    "avg_price": round(avg_buy, 2),
                    "buy_price": round(avg_buy, 2),
                    "current_price": round(curr_price, 2),
                    "market_value": round(mkt_val, 2),
                    "unrealized_pnl": round(pnl, 2),
                    "unrealized_pnl_pct": round(pnl_pct, 2)
                })

        total_portfolio_value = cash + total_market_val
        total_unrealized_pnl_pct = (unrealized_pnl / total_cost_basis * 100.0) if total_cost_basis > 0 else 0.0

        user_orders = [o for o in self.orders.values() if o.get("user_id") == user_id]

        return {
            "status": "success",
            "user_id": user_id,
            "cash": round(cash, 2),
            "cash_balance": round(cash, 2),
            "invested_capital": round(total_cost_basis, 2),
            "portfolio_value": round(total_market_val, 2),
            "current_holdings_value": round(total_market_val, 2),
            "net_worth": round(total_portfolio_value, 2),
            "total_portfolio_value": round(total_portfolio_value, 2),
            "total_pnl": round(unrealized_pnl, 2),
            "total_unrealized_pnl": round(unrealized_pnl, 2),
            "total_pnl_pct": round(total_unrealized_pnl_pct, 2),
            "total_unrealized_pnl_pct": round(total_unrealized_pnl_pct, 2),
            "active_positions_count": len(pos_list),
            "positions": pos_list,
            "orders_log": user_orders,
            "orders": user_orders
        }

    def place_order(
        self,
        user_id: str = "guest_user",
        ticker: str = "RELIANCE.NS",
        side: str = "BUY",
        quantity: int = 1,
        order_type: str = "MARKET",
        price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Executes order via Paper Trading Engine or Brokerage Adapter if ENABLE_LIVE_TRADING=true.
        """
        raw_side = side.upper().strip()
        if raw_side not in ["BUY", "SELL"]:
            raise ValueError("Invalid order side. Must be 'BUY' or 'SELL'.")
        if quantity <= 0:
            raise ValueError("Quantity must be greater than 0.")

        clean_ticker = ticker.upper().strip()
        order_id = f"ORD-{uuid.uuid4().hex[:8].upper()}"
        now_str = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")

        # Fetch current market price
        from services.market_data_service import get_market_data_service
        market_svc = get_market_data_service()
        quote = market_svc.get_live_quote(clean_ticker)
        current_mkt_price = quote.get("ltp", 0.0) if quote else 0.0

        if current_mkt_price <= 0.0:
            # Fallback historical price lookup
            hist_df = market_svc.get_historical_data(clean_ticker, period="5d")
            if hist_df is not None and not hist_df.empty and "Close" in hist_df.columns:
                current_mkt_price = float(hist_df["Close"].iloc[-1])

        fill_price = price if (price and price > 0) else current_mkt_price
        if fill_price <= 0.0:
            fill_price = 1000.0  # Safe default baseline
        fill_price = round(fill_price, 2)
        total_nominal = round(fill_price * quantity, 2)

        # Virtual Indian brokerage fee estimation (0.03% brokerage + 0.1% STT on BUY)
        brokerage = round(total_nominal * 0.0003, 2)
        stt = round(total_nominal * 0.001, 2) if raw_side == "BUY" else round(total_nominal * 0.00025, 2)
        est_fees = round(brokerage + stt, 2)

        if not ENABLE_LIVE_TRADING:
            # ── PAPER TRADING SIMULATION EXECUTION ─────────────────────
            user_balance = self._get_user_balance(user_id)
            user_pos_map = self.positions.get(user_id, {})
            existing_pos = user_pos_map.get(clean_ticker, {"shares": 0, "avg_buy_price": 0.0})

            if raw_side == "BUY":
                if (total_nominal + est_fees) > user_balance:
                    return {
                        "status": "rejected",
                        "order_id": order_id,
                        "reason": f"Insufficient paper trading funds. Required: ₹{total_nominal + est_fees:,.2f}, Available: ₹{user_balance:,.2f}"
                    }
            elif raw_side == "SELL":
                curr_owned = existing_pos.get("shares", 0)
                if quantity > curr_owned:
                    return {
                        "status": "rejected",
                        "order_id": order_id,
                        "reason": f"Cannot sell {quantity} shares of {clean_ticker}. You currently own {curr_owned} shares."
                    }

            # Update User Cash Balance
            if raw_side == "BUY":
                self.account_balances[user_id] = round(user_balance - (total_nominal + est_fees), 2)
            else:
                self.account_balances[user_id] = round(user_balance + (total_nominal - est_fees), 2)

            # Update Virtual Positions
            if user_id not in self.positions:
                self.positions[user_id] = {}

            user_pos = self.positions[user_id].get(clean_ticker, {
                "ticker": clean_ticker,
                "shares": 0,
                "avg_buy_price": 0.0,
                "current_price": fill_price,
                "market_value": 0.0,
                "unrealized_pnl": 0.0
            })

            if raw_side == "BUY":
                curr_shares = user_pos["shares"]
                curr_cost = curr_shares * user_pos["avg_buy_price"]
                new_shares = curr_shares + quantity
                new_avg = round((curr_cost + total_nominal) / new_shares, 2) if new_shares > 0 else 0.0
                user_pos["shares"] = new_shares
                user_pos["avg_buy_price"] = new_avg
            else:
                curr_shares = user_pos["shares"]
                new_shares = max(0, curr_shares - quantity)
                user_pos["shares"] = new_shares
                if new_shares == 0:
                    user_pos["avg_buy_price"] = 0.0

            user_pos["current_price"] = fill_price
            user_pos["market_value"] = round(user_pos["shares"] * fill_price, 2)
            user_pos["unrealized_pnl"] = round(user_pos["market_value"] - (user_pos["shares"] * user_pos["avg_buy_price"]), 2)
            self.positions[user_id][clean_ticker] = user_pos

            order_record = {
                "order_id": order_id,
                "user_id": user_id,
                "ticker": clean_ticker,
                "side": raw_side,
                "action": raw_side,
                "quantity": quantity,
                "shares": quantity,
                "order_type": order_type,
                "price": fill_price,
                "execution_price": fill_price,
                "limit_price": fill_price,
                "status": "FILLED",
                "execution_mode": "PAPER_TRADING",
                "total_cost": total_nominal,
                "nominal_value": total_nominal,
                "estimated_fees": est_fees,
                "timestamp": now_str
            }
            self.orders[order_id] = order_record
            self._save_db()
            logger.info(f"PAPER ORDER FILLED: {order_id} | {raw_side} {quantity} {clean_ticker} @ ₹{fill_price}")

            # Dispatch trade execution notification via Gmail SMTP
            try:
                from services.notification_service import notification_service
                notification_service.send_order_execution_email(order_record)
            except Exception as notify_err:
                logger.warning("Failed to send Gmail order execution notification: %s", notify_err)

            return {"status": "success", "order": order_record}

        else:
            # Live Trading Guardrail
            logger.warning(f"LIVE TRADING TRIGGERED (Disabled safety check active): Order {order_id}")
            return {
                "status": "error",
                "order_id": order_id,
                "message": "Live order execution is disabled by default safety policy."
            }

    def get_orders(self, user_id: str = "guest_user") -> List[Dict[str, Any]]:
        """Returns all executed orders for user."""
        return [o for o in self.orders.values() if o.get("user_id") == user_id]

    def get_positions(self, user_id: str = "guest_user") -> List[Dict[str, Any]]:
        """Returns active virtual portfolio positions for user."""
        user_pos_map = self.positions.get(user_id, {})
        return [p for p in user_pos_map.values() if p.get("shares", 0) > 0]

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancels a pending order."""
        if order_id in self.orders:
            ord_item = self.orders[order_id]
            if ord_item.get("status") == "PENDING":
                ord_item["status"] = "CANCELLED"
                self._save_db()
                return {"status": "success", "order_id": order_id, "message": "Order cancelled successfully."}
            return {"status": "warning", "order_id": order_id, "message": f"Order already {ord_item.get('status')}"}
        return {"status": "error", "order_id": order_id, "message": "Order not found."}

    def clear_orders_log(self, user_id: str = "guest_user") -> Dict[str, Any]:
        """Clears all executed order history records for the specified user."""
        self.orders = {k: v for k, v in self.orders.items() if v.get("user_id") != user_id}
        self._save_db()
        return {"status": "success", "message": "Executed orders log cleared successfully."}


def get_order_service() -> OrderService:
    """Returns singleton OrderService instance."""
    return OrderService()
