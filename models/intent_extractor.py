"""
SmartShop AI - Natural Language Query & Intent Extraction Pipeline
Extracts structured user requirements from natural language queries:
- Budget constraints (max budget, min budget, 'k' / 'lakh' multipliers)
- Target product category
- Hardware specification requirements (RAM, Storage, GPU, Display, OS)
- Brand preferences
- User personas and intended use cases (Machine Learning, Gaming, College, etc.)
"""

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config import VALID_CATEGORIES


class UserIntent:
    """Encapsulates structured requirements extracted from a natural-language query."""

    def __init__(
        self,
        raw_query: str,
        category: Optional[str] = None,
        budget_max: Optional[int] = None,
        budget_min: Optional[int] = None,
        brands: Optional[List[str]] = None,
        specs: Optional[Dict[str, Any]] = None,
        use_cases: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
    ):
        self.raw_query = raw_query
        self.category = category
        self.budget_max = budget_max
        self.budget_min = budget_min
        self.brands = brands or []
        self.specs = specs or {}
        self.use_cases = use_cases or []
        self.keywords = keywords or []

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the intent object to a clean dictionary."""
        return {
            "raw_query": self.raw_query,
            "category": self.category,
            "budget_max": self.budget_max,
            "budget_min": self.budget_min,
            "brands": self.brands,
            "specs": self.specs,
            "use_cases": self.use_cases,
            "keywords": self.keywords,
        }

    def summary(self) -> str:
        """Returns a human-readable summary of the extracted intent."""
        parts = []
        if self.category:
            parts.append(f"Category: {self.category}")
        if self.budget_max:
            min_str = f"₹{self.budget_min:,} - " if self.budget_min else "Up to "
            parts.append(f"Budget: {min_str}₹{self.budget_max:,}")
        if self.brands:
            parts.append(f"Preferred Brands: {', '.join(self.brands)}")
        if self.use_cases:
            parts.append(f"Target Use Cases: {', '.join(self.use_cases)}")
        if self.specs:
            spec_str = ", ".join([f"{k.upper()}: {v}" for k, v in self.specs.items()])
            parts.append(f"Required Specs: {spec_str}")
        return " | ".join(parts) if parts else "General Exploration"


class IntentExtractor:
    """Robust NLP and regex parser for e-commerce requirement parsing."""

    # Category Mapping
    CATEGORY_PATTERNS = {
        "Laptop": [
            r"\blaptops?\b", r"\bmacbooks?\b", r"\bultrabooks?\b",
            r"\bnotebooks?\b", r"\bpc\b", r"\bcomputer\b"
        ],
        "Smartphone": [
            r"\bsmartphones?\b", r"\bphones?\b", r"\bmobiles?\b",
            r"\biphones?\b", r"\bandroid\b"
        ],
        "Headphones": [
            r"\bheadphones?\b", r"\bearphones?\b", r"\bearbuds?\b",
            r"\bheadsets?\b", r"\btws\b", r"\bairpods\b", r"\baudio\b"
        ],
        "Tablet": [
            r"\btablets?\b", r"\bipads?\b", r"\btab\b"
        ],
        "Smartwatch": [
            r"\bsmartwatch(?:es)?\b", r"\bwatches\b", r"\biwatch\b"
        ]
    }

    # Brand Catalog
    KNOWN_BRANDS = [
        "Apple", "Acer", "ASUS", "Lenovo", "HP", "Dell", "MSI",
        "Samsung", "OnePlus", "Google", "Xiaomi", "Nothing",
        "Sony", "Bose", "boAt", "Sennheiser"
    ]

    # Persona & Use Case Mapping
    USE_CASE_MAP = {
        "machine_learning": [
            r"\bmachine\s+learning\b", r"\bml\b", r"\bdeep\s+learning\b",
            r"\bpytorch\b", r"\btensorflow\b", r"\bneural\s+networks?\b",
            r"\bdata\s+science\b", r"\bai\b", r"\bartificial\s+intelligence\b"
        ],
        "python": [
            r"\bpython\b", r"\bjupyter\b", r"\bpandas\b", r"\bnumpy\b"
        ],
        "coding": [
            r"\bcoding\b", r"\bprogramming\b", r"\bsoftware\s+development\b",
            r"\bdeveloper\b", r"\bweb\s+development\b", r"\bjavascript\b",
            r"\bjava\b", r"\bc\+\+\b", r"\bbackend\b", r"\bfrontend\b"
        ],
        "gaming": [
            r"\bgaming\b", r"\bgames?\b", r"\bgamer\b", r"\bhigh\s+fps\b",
            r"\besports\b", r"\baaa\s+titles\b"
        ],
        "college_work": [
            r"\bcollege\b", r"\bstudents?\b", r"\buniversity\b", r"\bassignments?\b",
            r"\bacademics?\b", r"\bengineering\b", r"\bstudy\b", r"\bschool\b"
        ],
        "video_editing": [
            r"\bvideo\s+editing\b", r"\bcontent\s+creation\b", r"\bpremiere\b",
            r"\byoutube\b", r"\bphotoshop\b", r"\b3d\s+rendering\b"
        ],
        "battery_life": [
            r"\bbattery\s+life\b", r"\ball\s+day\s+battery\b", r"\blong\s+battery\b",
            r"\bbattery\b"
        ],
        "lightweight": [
            r"\blightweight\b", r"\bportable\b", r"\bthin\s+and\s+light\b",
            r"\beasy\s+to\s+carry\b", r"\btravel\b"
        ],
        "noise_cancellation": [
            r"\bnoise\s+cancellation\b", r"\banc\b", r"\bactive\s+noise\b",
            r"\bsilent\b", r"\bquiet\b"
        ]
    }

    def __init__(self):
        pass

    def extract_intent(self, query: str) -> UserIntent:
        """Parses a natural language query and returns a structured UserIntent."""
        lower_query = query.lower()

        # 1. Extract Budget
        budget_min, budget_max = self._extract_budget(lower_query)

        # 2. Extract Category
        category = self._extract_category(lower_query)

        # 3. Extract Brands
        brands = self._extract_brands(query)

        # 4. Extract Technical Specs
        specs = self._extract_specs(lower_query)

        # 5. Extract Personas & Use Cases
        use_cases = self._extract_use_cases(lower_query)

        # 6. Extract high-signal keywords
        keywords = self._extract_keywords(lower_query)

        return UserIntent(
            raw_query=query,
            category=category,
            budget_max=budget_max,
            budget_min=budget_min,
            brands=brands,
            specs=specs,
            use_cases=use_cases,
            keywords=keywords,
        )

    # Brand Aliases Mapping
    BRAND_ALIASES = {
        "Apple": [r"\bapple\b", r"\bmacbooks?\b", r"\biphones?\b", r"\bipads?\b", r"\bairpods\b", r"\bmac\b"],
        "Lenovo": [r"\blenovo\b", r"\bthinkpad\b", r"\bideapad\b", r"\blegion\b", r"\bloq\b"],
        "ASUS": [r"\basus\b", r"\brog\b", r"\btuf\b", r"\bzenbook\b", r"\bvivobook\b"],
        "HP": [r"\bhp\b", r"\bvictus\b", r"\bpavilion\b", r"\bomen\b"],
        "Dell": [r"\bdell\b", r"\bxps\b", r"\binspiron\b", r"\balienware\b"],
        "Acer": [r"\bacer\b", r"\bnitro\b", r"\baspire\b", r"\bpredator\b", r"\bswift\b"],
        "Samsung": [r"\bsamsung\b", r"\bgalaxy\b"],
        "Google": [r"\bgoogle\b", r"\bpixel\b"],
        "OnePlus": [r"\boneplus\b", r"\bnord\b"],
        "Xiaomi": [r"\bxiaomi\b", r"\bredmi\b", r"\bpoco\b"],
        "Nothing": [r"\bnothing\b", r"\bcmp\b"],
        "Sony": [r"\bsony\b"],
        "Bose": [r"\bbose\b"],
        "boAt": [r"\bboat\b"],
        "MSI": [r"\bmsi\b"],
    }

    def _extract_budget(self, text: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Extracts budget constraints accurately from natural language.
        Prevents false matches against hardware numbers (RTX 3050, 16GB, 120Hz).
        """
        budget_min = None
        budget_max = None

        # Mask out specs with numbers so they are never confused with prices
        # e.g., 'rtx 3050', '16gb', '512gb', '120hz', '1080p', 'i5-13420h'
        masked_text = re.sub(r"\b(?:rtx|gtx)\s*\d{3,4}\w*\b", " [GPU_SPEC] ", text)
        masked_text = re.sub(r"\b\d+\s*(?:gb|tb|mb|hz|mp|ghz|mah|w|tgp|whr|nits)\b", " [SPEC_UNIT] ", masked_text)
        masked_text = re.sub(r"\bi[3579]-\d+\w*\b", " [CPU_SPEC] ", masked_text)
        masked_text = re.sub(r"\bryzen\s*[3579]\b", " [CPU_SPEC] ", masked_text)
        masked_text = re.sub(r"[₹,]|inr|rs\.?", " ", masked_text)

        # 1. Explicit Range: 'between X and Y' or 'from X to Y' or 'X - Y budget'
        range_explicit = r"(?:between|from)\s+(\d+(?:\.\d+)?\s*(?:k|lakhs?|l)?)\s+(?:and|to)\s+(\d+(?:\.\d+)?\s*(?:k|lakhs?|l)?)"
        match_range = re.search(range_explicit, masked_text)
        if match_range:
            val1 = self._parse_amount(match_range.group(1))
            val2 = self._parse_amount(match_range.group(2))
            if val1 and val2 and val1 >= 1000 and val2 >= 1000:
                budget_min = min(val1, val2)
                budget_max = max(val1, val2)
                return budget_min, budget_max

        # Range with hyphen e.g., '50k - 70k', '50000 to 75000'
        range_hyphen = r"(\d+(?:\.\d+)?\s*(?:k|lakhs?|l))\s*(?:-|to)\s*(\d+(?:\.\d+)?\s*(?:k|lakhs?|l))"
        match_hyphen = re.search(range_hyphen, masked_text)
        if match_hyphen:
            val1 = self._parse_amount(match_hyphen.group(1))
            val2 = self._parse_amount(match_hyphen.group(2))
            if val1 and val2 and val1 >= 1000 and val2 >= 1000:
                budget_min = min(val1, val2)
                budget_max = max(val1, val2)
                return budget_min, budget_max

        # 2. Upper bound patterns: 'under 70000', 'under 70k', 'less than 1.2 lakh', '<= 60k'
        max_patterns = [
            r"(?:under|below|less\s+than|within|upto|up\s+to|max(?:imum)?|<=?)\s*(\d+(?:\.\d+)?\s*(?:k|lakhs?|l)?)\b",
            r"(?:budget|price)\s*(?:is|of|around|approx)?\s*(?:around|approx)?\s*(\d+(?:\.\d+)?\s*(?:k|lakhs?|l)?)\b",
            r"(\d+(?:\.\d+)?\s*(?:k|lakhs?|l))\s*(?:budget|laptop|phone|headphone|tablet|price)",
            r"around\s+(\d+(?:\.\d+)?\s*(?:k|lakhs?|l)?)\b"
        ]

        for pat in max_patterns:
            match = re.search(pat, masked_text)
            if match:
                amount = self._parse_amount(match.group(1))
                if amount and amount >= 1000:
                    budget_max = amount
                    break

        return budget_min, budget_max

    def _parse_amount(self, token: str) -> Optional[int]:
        """Converts strings like '70k', '1.2 lakh', '70000' to integer amounts."""
        token = token.strip().lower()
        if not token:
            return None

        # Check for 'k' / thousands
        k_match = re.match(r"^(\d+(?:\.\d+)?)\s*k$", token)
        if k_match:
            return int(float(k_match.group(1)) * 1000)

        # Check for 'lakh' / 'l'
        lakh_match = re.match(r"^(\d+(?:\.\d+)?)\s*(?:lakhs?|l)$", token)
        if lakh_match:
            return int(float(lakh_match.group(1)) * 100000)

        # Raw numeric
        num_match = re.match(r"^(\d+)$", token)
        if num_match:
            return int(num_match.group(1))

        return None

    def _extract_category(self, text: str) -> Optional[str]:
        """Identifies target category from query keywords."""
        for category, patterns in self.CATEGORY_PATTERNS.items():
            for pat in patterns:
                if re.search(pat, text):
                    return category
        return None

    def _extract_brands(self, text: str) -> List[str]:
        """Identifies any explicitly mentioned brand preferences or brand aliases."""
        detected = []
        for brand, patterns in self.BRAND_ALIASES.items():
            for pat in patterns:
                if re.search(pat, text, re.IGNORECASE):
                    if brand not in detected:
                        detected.append(brand)
                    break
        return detected

    def _extract_specs(self, text: str) -> Dict[str, Any]:
        """Extracts desired hardware specifications (RAM, GPU, Storage, Display, OS)."""
        specs: Dict[str, Any] = {}

        # RAM requirement: e.g. 16GB, 8GB, 32GB RAM
        ram_match = re.search(r"\b(4|8|16|32|64)\s*(?:gb)?\s*(?:ram|memory)\b", text)
        if ram_match:
            specs["ram"] = f"{ram_match.group(1)}GB"
        elif re.search(r"\b16gb\b", text):
            specs["ram"] = "16GB"
        elif re.search(r"\b8gb\b", text):
            specs["ram"] = "8GB"
        elif re.search(r"\b32gb\b", text):
            specs["ram"] = "32GB"

        # GPU requirement
        if re.search(r"\b(rtx\s*\d{4}|dedicated\s+gpu|graphics\s+card|nvidia|geforce)\b", text):
            rtx_match = re.search(r"\b(rtx\s*\d{4})\b", text)
            specs["gpu"] = rtx_match.group(1).upper().replace(" ", "") if rtx_match else "Dedicated GPU"

        # Storage requirement: e.g. 512GB, 1TB SSD
        storage_match = re.search(r"\b(256|512|1tb|2tb)\s*(?:gb)?\s*(?:ssd|storage|rom)?\b", text)
        if storage_match:
            val = storage_match.group(1).upper()
            if not val.endswith("TB") and not val.endswith("GB"):
                val = f"{val}GB"
            specs["storage"] = val

        # Display preference (e.g. OLED, 120Hz, 144Hz)
        if re.search(r"\boled\b", text):
            specs["display"] = "OLED"
        elif re.search(r"\b(120hz|144hz|high\s+refresh)\b", text):
            specs["display"] = "High Refresh Rate (120Hz+)"

        # Operating System preference
        if re.search(r"\b(mac|macos|apple\s+silicon)\b", text):
            specs["os"] = "macOS"
        elif re.search(r"\bwindows\b", text):
            specs["os"] = "Windows"
        elif re.search(r"\bandroid\b", text):
            specs["os"] = "Android"

        return specs

    def _extract_use_cases(self, text: str) -> List[str]:
        """Identifies target activities, domains, or personas."""
        use_cases = []
        for uc, patterns in self.USE_CASE_MAP.items():
            for pat in patterns:
                if re.search(pat, text):
                    use_cases.append(uc)
                    break
        return use_cases

    def _extract_keywords(self, text: str) -> List[str]:
        """Extracts high-signal search tokens, filtering common stopwords."""
        stopwords = {
            "i", "need", "a", "an", "the", "for", "and", "in", "with",
            "want", "looking", "please", "recommend", "best", "good",
            "show", "me", "to", "buy", "under", "budget", "around", "of"
        }
        words = re.findall(r"\b[a-zA-Z0-9_\-\+\.]{2,}\b", text)
        return [w for w in words if w not in stopwords]


# Global Singleton
_intent_extractor_instance: Optional[IntentExtractor] = None


def get_intent_extractor() -> IntentExtractor:
    """Returns singleton instance of IntentExtractor."""
    global _intent_extractor_instance
    if _intent_extractor_instance is None:
        _intent_extractor_instance = IntentExtractor()
    return _intent_extractor_instance


if __name__ == "__main__":
    extractor = get_intent_extractor()
    test_queries = [
        "I need a laptop under 70000 for Python, machine learning and college work.",
        "Gaming laptop with RTX 3050 and 16GB RAM under 75k",
        "Lightweight ultrabook under 60000 for long battery life and coding",
        "Best smartphone under 25000 with 120Hz AMOLED display and good camera",
        "Sony or Bose headphones under 25k with noise cancellation for study",
        "MacBook for iOS development under 1.2 lakh",
    ]

    print("=" * 70)
    print("Testing IntentExtractor on canonical student queries:")
    print("=" * 70)
    for q in test_queries:
        intent = extractor.extract_intent(q)
        print(f"\nQuery: '{q}'")
        print(f"Summary: {intent.summary()}")
        print(f"Details: {intent.to_dict()}")
