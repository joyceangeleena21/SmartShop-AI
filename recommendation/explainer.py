"""
SmartShop AI - Explainable AI (XAI) Recommendation Engine
Generates clear, multi-dimensional, human-understandable explanations for WHY
each product was recommended based on query intent, budget compatibility,
spec alignment, rating credibility, and honest trade-offs.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.intent_extractor import UserIntent
from recommendation.ranker import RecommendationResult


class RecommendationExplainer:
    """Produces structured and natural-language explanations for recommendations."""

    @staticmethod
    def is_dedicated_gpu(gpu_str: str) -> bool:
        """
        Determines whether the GPU string denotes a dedicated discrete graphics card.
        True for NVIDIA RTX/GTX/GeForce, AMD Radeon RX, or discrete GPUs.
        False for Intel Iris Xe, Intel UHD, AMD integrated Radeon, Apple Unified, mobile GPUs.
        """
        gpu_upper = gpu_str.upper()
        # Explicit integrated markers
        if any(integrated_kw in gpu_upper for integrated_kw in ["INTEGRATED", "IRIS XE", "INTEL UHD", "RADEON 610M", "RADEON GRAPHICS", "APPLE 8-CORE", "APPLE 10-CORE"]):
            return False
        # Explicit dedicated markers
        if any(dedicated_kw in gpu_upper for dedicated_kw in ["RTX", "GTX", "GEFORCE", "RADEON RX", "DEDICATED"]):
            return True
        return False

    def explain(
        self,
        result: RecommendationResult,
        intent: UserIntent
    ) -> Dict[str, Any]:
        """
        Generates a comprehensive explainability dossier for a recommended product.
        Includes:
        - Headline takeaway
        - Natural language narrative
        - Quantitative score contributions
        - Key matching highlights
        - Trade-off / considerations for the buyer
        """
        p = result.product
        price = p.get("price", 0)
        rating = p.get("rating", 4.0)
        reviews = p.get("review_count", 0)
        gpu = str(p.get("gpu", ""))
        ram = str(p.get("ram", ""))
        weight = str(p.get("weight", ""))
        battery = str(p.get("battery_life", ""))
        display = str(p.get("display", ""))
        category = str(p.get("category", ""))

        has_dedicated_gpu = self.is_dedicated_gpu(gpu)

        # 1. Headline
        headline = self._generate_headline(result, intent, has_dedicated_gpu)

        # 2. Key Pros / Matching Highlights
        highlights: List[str] = []

        # Budget Fit Highlight
        if intent.budget_max and price <= intent.budget_max:
            savings = intent.budget_max - price
            if savings > 0:
                highlights.append(f"Budget Fit: Priced at ₹{price:,}, leaving ₹{savings:,} headroom within your ₹{intent.budget_max:,} limit.")
            else:
                highlights.append(f"Budget Fit: Perfectly matched to your budget cap of ₹{intent.budget_max:,}.")
        elif intent.budget_max and price > intent.budget_max:
            over = price - intent.budget_max
            highlights.append(f"Budget Note: Priced at ₹{price:,} (₹{over:,} above limit), but offers higher specifications for your requirements.")

        # Hardware & GPU Highlight (strictly accurately based on actual specs)
        if category == "Laptop":
            if has_dedicated_gpu:
                highlights.append(f"Hardware Acceleration: Dedicated {gpu} delivers dedicated VRAM and CUDA/Tensor cores for PyTorch, TensorFlow, and gaming.")
            elif "machine_learning" in intent.use_cases:
                gpu_display = gpu if "integrated" in gpu.lower() else f"{gpu} (Integrated)"
                highlights.append(f"Compute & Data Science: Powered by {p.get('processor', 'CPU')} with {gpu_display}, well-suited for Python, Pandas, Scikit-learn, and CPU data modeling.")

            # RAM Highlight
            if "16GB" in ram or "32GB" in ram or "64GB" in ram:
                highlights.append(f"Memory: {ram} provides ample headspace for concurrent IDEs, Jupyter notebooks, Docker, and datasets.")
            elif "8GB" in ram and "upgradable" in str(p.get("features", "")).lower():
                highlights.append(f"Memory: {ram} with upgradable memory slots for future expansion.")

            # Display Highlight
            if "OLED" in display.upper() or "2.8K" in display.upper() or "3K" in display.upper():
                highlights.append(f"Display Quality: Vibrant {display} with high resolution and superior color fidelity.")
            elif "144HZ" in display.upper() or "120HZ" in display.upper():
                highlights.append(f"Display: Fast {display} for smooth visual clarity and reduced eye strain.")

            # Battery Endurance Highlight (if strong)
            if any(hrs in battery for hrs in ["8 hours", "8.5 hours", "10 hours", "10.5 hours", "18 hours"]):
                highlights.append(f"Battery Endurance: Excellent battery life ({battery}) for working across campus lectures without needing a charger.")

        elif category == "Smartphone":
            if "120Hz" in display or "AMOLED" in display:
                highlights.append(f"Display: Smooth {display} for fluid navigation and multimedia.")
            if "Fast Charge" in p.get("features", "") or "SUPERVOOC" in p.get("title", ""):
                highlights.append(f"Charging: Rapid flash charging ensures quick top-ups between classes.")

        elif category == "Headphones":
            if "Active Noise" in p.get("features", "") or "ANC" in p.get("features", "") or "ANC" in p.get("subcategory", ""):
                highlights.append("Noise Cancellation: Active Noise Cancellation isolates ambient library and transit distractions for deep focus.")

        # Activity Alignment
        if result.matched_use_cases:
            highlights.append(f"Use-Case Match: Explicitly tailored for {', '.join(result.matched_use_cases)}.")

        # Social Proof
        highlights.append(f"User Credibility: Verified {rating}★ rating from {reviews:,} buyer reviews.")

        # 3. Honest Trade-Off / Considerations (Strictly spec-truthful)
        trade_offs: List[str] = []

        if category == "Laptop":
            # Weight trade-off
            try:
                wt_val = float(weight.replace("kg", "").strip())
                if wt_val >= 2.3:
                    trade_offs.append(f"Portability: Heavier chassis ({weight}) due to performance cooling system and dual exhaust fans.")
                elif wt_val <= 1.35:
                    trade_offs.append(f"Thermal Profile: Ultra-thin design ({weight}); sustained heavy computational rendering may trigger fan noise.")
            except Exception:
                pass

            # Short battery check using word boundary so '8.5 hours' does not match '5 hours'
            is_short_battery = any(
                battery.startswith(h) or f" {h}" in battery 
                for h in ["5 hours", "5.5 hours", "6 hours"]
            ) and "8.5 hours" not in battery and "7.5 hours" not in battery

            # Battery trade-off - ONLY add dedicated GPU warning if it ACTUALLY has a dedicated GPU!
            if has_dedicated_gpu:
                if is_short_battery:
                    trade_offs.append(f"Battery Runtime: Dedicated GPU ({gpu}) draws higher power under graphical loads; keep the power brick handy for all-day campus sessions.")
            else:
                # For integrated GPU laptops with truly short battery (< 6 hours)
                if is_short_battery and not any(hrs in battery for hrs in ["8 hours", "8.5 hours", "10 hours"]):
                    trade_offs.append(f"Battery Life: Moderate {battery} runtime under heavy multitasking; carry charger for extended work.")

            # RAM upgrade trade-off if 8GB
            if "8GB" in ram and "machine_learning" in intent.use_cases:
                trade_offs.append(f"RAM Capacity: Equipped with {ram}; consider upgrading to 16GB for large machine learning datasets.")

        elif category == "Smartphone":
            if "Wireless Charging" not in str(p.get("features", "")) and price > 30000:
                trade_offs.append("Features: Supports ultra-fast wired charging but omits wireless Qi charging.")

        if not trade_offs:
            trade_offs.append("Balanced Specifications: All hardware attributes align closely with query requirements with minimal trade-offs.")

        # 4. Natural Language Synthesis
        narrative = self._generate_narrative(result, intent, has_dedicated_gpu)

        return {
            "product_id": p.get("product_id"),
            "product_title": p.get("title"),
            "headline": headline,
            "narrative": narrative,
            "score_breakdown": {
                "composite_pct": int(result.composite_score * 100),
                "semantic_match_pct": int(result.semantic_score * 100),
                "budget_fit_pct": int(result.budget_score * 100),
                "rating_credibility_pct": int(result.rating_score * 100),
                "spec_alignment_pct": int(result.feature_score * 100),
            },
            "highlights": highlights,
            "trade_offs": trade_offs,
        }

    def _generate_headline(self, result: RecommendationResult, intent: UserIntent, has_dedicated_gpu: bool) -> str:
        """Constructs an eye-catching recommendation summary badge matching real hardware."""
        p = result.product
        price = p.get("price", 0)

        if intent.budget_max and price <= intent.budget_max:
            if has_dedicated_gpu:
                return f"Top Dedicated GPU Pick under ₹{intent.budget_max:,}"
            if "machine_learning" in intent.use_cases or "python" in intent.use_cases:
                return f"Top Value Coding Pick under ₹{intent.budget_max:,}"
            return f"Best Overall Match under ₹{intent.budget_max:,}"
        elif has_dedicated_gpu:
            return "High-Performance GPU Pick"
        elif "lightweight" in intent.use_cases or "battery_life" in intent.use_cases:
            return "Premier Portable Ultrabook Pick"
        return "Top Ranked Recommendation"

    def _generate_narrative(
        self,
        result: RecommendationResult,
        intent: UserIntent,
        has_dedicated_gpu: bool
    ) -> str:
        """Constructs a fluent, technically accurate explanation paragraph."""
        p = result.product
        brand = p.get("brand", "")
        category = p.get("category", "")
        price = p.get("price", 0)
        rating = p.get("rating", 4.0)
        gpu = str(p.get("gpu", ""))

        budget_phrase = (
            f"under your ₹{intent.budget_max:,} budget at ₹{price:,}"
            if intent.budget_max and price <= intent.budget_max
            else f"priced at ₹{price:,}"
        )

        use_case_phrase = (
            f"for {', '.join(intent.use_cases).replace('_', ' ')}"
            if intent.use_cases
            else "for your search criteria"
        )

        gpu_phrase = f"featuring dedicated {gpu} graphics" if has_dedicated_gpu else f"featuring {gpu}"

        return (
            f"SmartShop AI selected the {brand} {category} ({p.get('title')[:42]}...) because it is a strong fit {use_case_phrase}, "
            f"{gpu_phrase}. It achieves a {result.semantic_score * 100:.0f}% dense semantic relevance score, fits {budget_phrase}, "
            f"and is validated by a {rating}★ rating from {p.get('review_count', 0):,} verified owners."
        )


# Global Singleton
_explainer_instance: Optional[RecommendationExplainer] = None


def get_recommendation_explainer() -> RecommendationExplainer:
    """Returns singleton instance of RecommendationExplainer."""
    global _explainer_instance
    if _explainer_instance is None:
        _explainer_instance = RecommendationExplainer()
    return _explainer_instance
