"""
quant/order_book_engine.py
==========================
Level-2 Order Depth & Order Flow Imbalance (OFI) Engine
-------------------------------------------------------
Processes Level-2 market depth (top 5 bids/asks) to calculate
Order Flow Imbalance (OFI), Bid-Ask Spread, and Liquidity Ratio.
"""

import numpy as np
from typing import Dict, List, Any


class OrderBookEngine:
    """
    Level-2 Order Book Depth & Order Flow Imbalance processor.
    """

    @staticmethod
    def calculate_order_flow_imbalance(bids: List[Dict[str, float]], asks: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        Calculates Order Flow Imbalance (OFI), Bid-Ask Spread, and Depth Ratio from L2 depth.
        """
        if not bids or not asks:
            return {
                "order_flow_imbalance": 0.0,
                "bid_ask_spread": 0.05,
                "depth_ratio": 1.0,
                "signal": "NEUTRAL ⚪"
            }

        total_bid_qty = sum(b.get("quantity", 0) for b in bids)
        total_ask_qty = sum(a.get("quantity", 0) for a in asks)

        top_bid = bids[0].get("price", 0.0) if bids else 0.0
        top_ask = asks[0].get("price", 0.0) if asks else 0.0

        spread = max(0.0, top_ask - top_bid)

        # Order Flow Imbalance (OFI) = (Bid_Qty - Ask_Qty) / (Bid_Qty + Ask_Qty)
        total_depth = total_bid_qty + total_ask_qty + 1e-9
        ofi = (total_bid_qty - total_ask_qty) / total_depth
        depth_ratio = total_bid_qty / (total_ask_qty + 1e-9)

        if ofi >= 0.35:
            signal = "STRONG BUYING PRESSURE 🟢"
        elif ofi >= 0.15:
            signal = "BUYING PRESSURE 🟢"
        elif ofi <= -0.35:
            signal = "STRONG SELLING PRESSURE 🔴"
        elif ofi <= -0.15:
            signal = "SELLING PRESSURE 🔴"
        else:
            signal = "BALANCED DEPTH ⚪"

        return {
            "order_flow_imbalance": float(ofi),
            "bid_ask_spread": float(spread),
            "depth_ratio": float(depth_ratio),
            "total_bid_qty": int(total_bid_qty),
            "total_ask_qty": int(total_ask_qty),
            "signal": signal
        }

    @staticmethod
    def generate_simulated_l2_depth(current_price: float) -> Dict[str, Any]:
        """
        Generates realistic Level-2 market depth for active stocks.
        """
        cp = current_price if current_price > 0 else 1000.0
        bids = [
            {"price": round(cp * 0.9995, 2), "quantity": 1250, "orders": 12},
            {"price": round(cp * 0.9990, 2), "quantity": 2400, "orders": 24},
            {"price": round(cp * 0.9985, 2), "quantity": 3800, "orders": 35},
            {"price": round(cp * 0.9980, 2), "quantity": 5100, "orders": 48},
            {"price": round(cp * 0.9975, 2), "quantity": 6200, "orders": 59}
        ]
        asks = [
            {"price": round(cp * 1.0005, 2), "quantity": 980, "orders": 10},
            {"price": round(cp * 1.0010, 2), "quantity": 1850, "orders": 18},
            {"price": round(cp * 1.0015, 2), "quantity": 2900, "orders": 29},
            {"price": round(cp * 1.0020, 2), "quantity": 4100, "orders": 41},
            {"price": round(cp * 1.0025, 2), "quantity": 5300, "orders": 52}
        ]
        metrics = OrderBookEngine.calculate_order_flow_imbalance(bids, asks)
        return {
            "bids": bids,
            "asks": asks,
            "metrics": metrics
        }
