import numpy as np
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Dict, List

class RiskManagementInput(BaseModel):
    current_price: float = Field(..., description="Current stock price")
    portfolio_value: float = Field(100000, description="Total portfolio value")
    risk_per_trade: float = Field(0.02, description="Risk percentage per trade (default 2%)")
    volatility: float = Field(0.02, description="Stock volatility (daily)")
    position_type: str = Field("long", description="Position type: long or short")

class RiskManagementTool(BaseTool):
    name: str = "risk_management_tool"
    description: str = "Calculate position sizing, stop losses, and risk metrics"
    args_schema: type[BaseModel] = RiskManagementInput

    def _run(self, current_price: float, portfolio_value: float, risk_per_trade: float, 
             volatility: float, position_type: str) -> Dict:
        
        try:
            results = {
                "position_sizing": {},
                "stop_loss_levels": {},
                "take_profit_levels": {},
                "risk_metrics": {},
                "portfolio_allocation": {},
                "recommendations": []
            }

            # Calculate position sizing
            results["position_sizing"] = self._calculate_position_size(
                portfolio_value, risk_per_trade, current_price, volatility
            )

            # Calculate stop loss levels
            results["stop_loss_levels"] = self._calculate_stop_loss(
                current_price, volatility, position_type
            )

            # Calculate take profit levels
            results["take_profit_levels"] = self._calculate_take_profit(
                current_price, volatility, position_type
            )

            # Calculate risk metrics
            results["risk_metrics"] = self._calculate_risk_metrics(
                current_price, portfolio_value, risk_per_trade, volatility
            )

            # Portfolio allocation recommendations
            results["portfolio_allocation"] = self._calculate_portfolio_allocation(
                portfolio_value, volatility, results["risk_metrics"]
            )

            # Generate recommendations
            results["recommendations"] = self._generate_risk_recommendations(results)

        except Exception as e:
            return {"error": f"Risk management calculation failed: {str(e)}"}

        return results

    def _calculate_position_size(self, portfolio_value: float, risk_pct: float, 
                               current_price: float, volatility: float) -> Dict:
        """Calculate optimal position size based on risk management rules"""
        
        # Amount willing to risk
        risk_amount = portfolio_value * risk_pct
        
        # ATR-based stop loss (2 * daily volatility)
        atr_stop_distance = current_price * volatility * 2
        
        # Position size based on risk amount and stop distance
        shares_by_risk = int(risk_amount / atr_stop_distance) if atr_stop_distance > 0 else 0
        
        # Kelly Criterion position size (simplified)
        # Assuming 55% win rate and 1.5:1 reward-to-risk ratio
        win_rate = 0.55
        avg_win = 1.5 * atr_stop_distance
        avg_loss = atr_stop_distance
        if avg_win > 0:
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        else:
            kelly_fraction = 0
        
        shares_by_kelly = int((portfolio_value * kelly_fraction) / current_price) if current_price > 0 else 0
        
        # Conservative position size (1% of portfolio)
        conservative_shares = int((portfolio_value * 0.01) / current_price) if current_price > 0 else 0
        
        # Choose the most conservative approach
        recommended_shares = min(shares_by_risk, shares_by_kelly, conservative_shares)
        position_value = recommended_shares * current_price
        portfolio_allocation = (position_value / portfolio_value) * 100 if portfolio_value > 0 else 0
        
        return {
            "recommended_shares": recommended_shares,
            "position_value": round(position_value, 2),
            "portfolio_allocation_pct": round(portfolio_allocation, 2),
            "risk_amount": round(risk_amount, 2),
            "stop_distance": round(atr_stop_distance, 2),
            "method": "Conservative (minimum of risk-based, Kelly, and 1% rule)"
        }

    def _calculate_stop_loss(self, current_price: float, volatility: float, 
                           position_type: str) -> Dict:
        """Calculate multiple stop loss levels"""
        
        # ATR-based stop (2x daily volatility)
        atr_stop_distance = current_price * volatility * 2
        
        # Percentage-based stops
        tight_stop_pct = 0.015  # 1.5%
        normal_stop_pct = 0.025  # 2.5%
        wide_stop_pct = 0.04    # 4%
        
        if position_type.lower() == "long":
            atr_stop = current_price - atr_stop_distance
            tight_stop = current_price * (1 - tight_stop_pct)
            normal_stop = current_price * (1 - normal_stop_pct) 
            wide_stop = current_price * (1 - wide_stop_pct)
        else:  # short position
            atr_stop = current_price + atr_stop_distance
            tight_stop = current_price * (1 + tight_stop_pct)
            normal_stop = current_price * (1 + normal_stop_pct)
            wide_stop = current_price * (1 + wide_stop_pct)

        return {
            "atr_based": round(atr_stop, 2),
            "tight_1.5pct": round(tight_stop, 2),
            "normal_2.5pct": round(normal_stop, 2),
            "wide_4pct": round(wide_stop, 2),
            "recommended": round(normal_stop, 2),
            "position_type": position_type
        }

    def _calculate_take_profit(self, current_price: float, volatility: float,
                             position_type: str) -> Dict:
        """Calculate take profit levels based on risk-reward ratios"""
        
        # Base stop distance for R:R calculation
        stop_distance = current_price * volatility * 2
        
        # Multiple R:R ratios
        rr_ratios = [1.0, 1.5, 2.0, 3.0]
        
        take_profits = {}
        
        for rr in rr_ratios:
            if position_type.lower() == "long":
                tp_price = current_price + (stop_distance * rr)
            else:  # short
                tp_price = current_price - (stop_distance * rr)
            
            take_profits[f"tp_{rr:.1f}R"] = round(tp_price, 2)

        # Volatility-based targets
        if position_type.lower() == "long":
            vol_target_1 = current_price * (1 + volatility * 3)
            vol_target_2 = current_price * (1 + volatility * 5)
        else:
            vol_target_1 = current_price * (1 - volatility * 3)
            vol_target_2 = current_price * (1 - volatility * 5)

        return {
            "risk_reward_targets": take_profits,
            "volatility_target_1": round(vol_target_1, 2),
            "volatility_target_2": round(vol_target_2, 2),
            "recommended": take_profits.get("tp_2.0R", current_price),
            "trailing_stop_suggestion": "Use 1.5% trailing stop after reaching 1.5R"
        }

    def _calculate_risk_metrics(self, current_price: float, portfolio_value: float,
                              risk_pct: float, volatility: float) -> Dict:
        """Calculate various risk metrics"""
        
        # Value at Risk (VaR) - 95% confidence
        daily_var = current_price * volatility * 1.65  # 95% confidence
        weekly_var = daily_var * np.sqrt(5)
        monthly_var = daily_var * np.sqrt(22)
        
        # Maximum Drawdown estimate
        max_dd_estimate = volatility * np.sqrt(252) * 2  # Rough annual estimate
        
        # Sharpe Ratio components (assuming risk-free rate of 3%)
        risk_free_rate = 0.03
        excess_return_estimate = 0.08  # Assumed annual return
        annual_volatility = volatility * np.sqrt(252)
        sharpe_estimate = (excess_return_estimate - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0
        
        # Position heat (risk as % of portfolio)
        position_heat = risk_pct * 100
        
        return {
            "daily_var_95pct": round(daily_var, 2),
            "weekly_var_95pct": round(weekly_var, 2),
            "monthly_var_95pct": round(monthly_var, 2),
            "estimated_max_drawdown": round(max_dd_estimate * 100, 1),
            "estimated_sharpe_ratio": round(sharpe_estimate, 2),
            "position_heat_pct": round(position_heat, 2),
            "volatility_regime": self._classify_volatility(volatility)
        }

    def _classify_volatility(self, volatility: float) -> str:
        """Classify volatility regime"""
        if volatility < 0.015:
            return "Low volatility"
        elif volatility < 0.025:
            return "Normal volatility" 
        elif volatility < 0.035:
            return "High volatility"
        else:
            return "Extreme volatility"

    def _calculate_portfolio_allocation(self, portfolio_value: float, volatility: float,
                                     risk_metrics: Dict) -> Dict:
        """Calculate recommended portfolio allocation"""
        
        vol_regime = risk_metrics.get("volatility_regime", "Normal volatility")
        
        # Base allocation based on volatility
        if vol_regime == "Low volatility":
            max_single_position = 0.08  # 8%
            max_sector_exposure = 0.25   # 25%
        elif vol_regime == "Normal volatility":
            max_single_position = 0.05  # 5%
            max_sector_exposure = 0.20   # 20%
        elif vol_regime == "High volatility":
            max_single_position = 0.03  # 3%
            max_sector_exposure = 0.15   # 15%
        else:  # Extreme volatility
            max_single_position = 0.02  # 2%
            max_sector_exposure = 0.10   # 10%
        
        return {
            "max_single_position_pct": round(max_single_position * 100, 1),
            "max_sector_exposure_pct": round(max_sector_exposure * 100, 1),
            "recommended_cash_pct": round((1 - max_sector_exposure * 3) * 100, 1),
            "volatility_regime": vol_regime,
            "diversification_rule": f"No more than {max_single_position*100:.1f}% in any single stock"
        }

    def _generate_risk_recommendations(self, results: Dict) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        position_sizing = results.get("position_sizing", {})
        risk_metrics = results.get("risk_metrics", {})
        allocation = results.get("portfolio_allocation", {})
        
        # Position size recommendations
        if position_sizing.get("portfolio_allocation_pct", 0) > 5:
            recommendations.append("WARNING: Position size exceeds 5% of portfolio - consider reducing")
        
        # Volatility recommendations
        vol_regime = risk_metrics.get("volatility_regime", "")
        if "High" in vol_regime or "Extreme" in vol_regime:
            recommendations.append(f"CAUTION: {vol_regime} detected - reduce position sizes and use tighter stops")
        
        # Risk-reward recommendations
        recommendations.append("Use minimum 2:1 risk-reward ratio for all trades")
        recommendations.append("Never risk more than 2% of portfolio on single trade")
        
        # Portfolio recommendations
        max_single = allocation.get("max_single_position_pct", 5)
        recommendations.append(f"Maintain portfolio diversification - max {max_single}% per position")
        
        # Stop loss recommendations
        recommendations.append("Always use stop losses - no exceptions")
        recommendations.append("Consider using trailing stops after reaching 1.5R profit")
        
        return recommendations
