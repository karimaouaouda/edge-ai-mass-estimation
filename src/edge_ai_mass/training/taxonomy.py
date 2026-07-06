"""Canonical waste taxonomy and source-specific label resolvers."""

from __future__ import annotations

from typing import Any


WASTE_CLASSES = [
    "organic_waste",
    "plastic_bottle",
    "plastic_bag",
    "rigid_plastic",
    "metal_can",
    "glass",
    "paper_cardboard",
    "mixed_waste",
]


TACO_LABEL_MAP = {
    "Food waste": "organic_waste",
    "Other plastic bottle": "plastic_bottle",
    "Clear plastic bottle": "plastic_bottle",
    "Plastic film": "plastic_bag",
    "Six pack rings": "plastic_bag",
    "Garbage bag": "plastic_bag",
    "Other plastic wrapper": "plastic_bag",
    "Single-use carrier bag": "plastic_bag",
    "Polypropylene bag": "plastic_bag",
    "Crisp packet": "plastic_bag",
    "Plastic bottle cap": "rigid_plastic",
    "Plastic lid": "rigid_plastic",
    "Other plastic": "rigid_plastic",
    "Disposable plastic cup": "rigid_plastic",
    "Foam cup": "rigid_plastic",
    "Other plastic cup": "rigid_plastic",
    "Spread tub": "rigid_plastic",
    "Tupperware": "rigid_plastic",
    "Disposable food container": "rigid_plastic",
    "Foam food container": "rigid_plastic",
    "Other plastic container": "rigid_plastic",
    "Plastic glooves": "rigid_plastic",
    "Plastic utensils": "rigid_plastic",
    "Squeezable tube": "rigid_plastic",
    "Plastic straw": "rigid_plastic",
    "Styrofoam piece": "mixed_waste",
    "Aluminium foil": "metal_can",
    "Aluminium blister pack": "metal_can",
    "Metal bottle cap": "metal_can",
    "Food Can": "metal_can",
    "Aerosol": "metal_can",
    "Drink can": "metal_can",
    "Metal lid": "metal_can",
    "Pop tab": "metal_can",
    "Scrap metal": "metal_can",
    "Glass bottle": "glass",
    "Broken glass": "glass",
    "Glass cup": "glass",
    "Glass jar": "glass",
    "Toilet tube": "paper_cardboard",
    "Other carton": "paper_cardboard",
    "Egg carton": "paper_cardboard",
    "Drink carton": "paper_cardboard",
    "Corrugated carton": "paper_cardboard",
    "Meal carton": "paper_cardboard",
    "Pizza box": "paper_cardboard",
    "Paper cup": "paper_cardboard",
    "Magazine paper": "paper_cardboard",
    "Tissues": "paper_cardboard",
    "Wrapping paper": "paper_cardboard",
    "Normal paper": "paper_cardboard",
    "Paper bag": "paper_cardboard",
    "Plastified paper bag": "paper_cardboard",
    "Paper straw": "paper_cardboard",
    "Battery": "mixed_waste",
    "Carded blister pack": "mixed_waste",
    "Rope & strings": "mixed_waste",
    "Shoe": "mixed_waste",
    "Unlabeled litter": "mixed_waste",
    "Cigarette": "mixed_waste",
}


DIRECT_ALIASES = {
    "glass": "glass",
    "metal": "metal_can",
    "metal_can": "metal_can",
    "paper": "paper_cardboard",
    "paper_cardboard": "paper_cardboard",
    "plastic": "rigid_plastic",
    "plastic_battle": "plastic_bottle",
    "plastic_bottle": "plastic_bottle",
    "plastic_bag": "plastic_bag",
    "rigid_plastic": "rigid_plastic",
    "organic_waste": "organic_waste",
    "mixed_waste": "mixed_waste",
    "cardboard": "paper_cardboard",
    "food organics": "organic_waste",
    "miscellaneous trash": "mixed_waste",
    "textile trash": "mixed_waste",
    "vegetation": "organic_waste",
    # The RealWaste SAM notebook produces a one-class COCO category and retains
    # the original folder label on each image.  This fallback is intentionally
    # generic; category_from_image_field should be preferred for that source.
    "trash": "mixed_waste",
}


TRASHNET_LABEL_MAP = {
    # TrashNet's original taxonomy is image-level and coarse.  The annotated
    # segmentation JSON keeps those category names, so this resolver maps them
    # deliberately into the canonical project taxonomy used by YOLO training.
    "cardboard": "paper_cardboard",
    "glass": "glass",
    "metal": "metal_can",
    "paper": "paper_cardboard",
    "plastic": "rigid_plastic",
    "trash": "mixed_waste",
}


def resolve_taco_category(category: dict[str, Any]) -> str:
    """Map a TACO category into the canonical eight-class label space."""
    name = str(category.get("name", "")).strip()
    supercategory = str(category.get("supercategory", "")).strip().lower()
    lower_name = name.lower()

    if name in TACO_LABEL_MAP:
        return TACO_LABEL_MAP[name]
    if supercategory == "bottle":
        return "glass" if "glass" in lower_name else "plastic_bottle"
    if supercategory == "bottle cap":
        return "metal_can" if "metal" in lower_name else "rigid_plastic"
    if supercategory in {"paper", "carton", "paper bag"}:
        return "paper_cardboard"
    if supercategory == "plastic bag & wrapper":
        return "plastic_bag"
    if supercategory in {"plastic container", "other plastic"}:
        return "rigid_plastic"
    if supercategory == "cup":
        if lower_name.startswith("glass"):
            return "glass"
        if lower_name.startswith("paper"):
            return "paper_cardboard"
        return "rigid_plastic"
    if supercategory == "lid":
        return "metal_can" if "metal" in lower_name else "rigid_plastic"
    if supercategory == "straw":
        return "paper_cardboard" if lower_name.startswith("paper") else "rigid_plastic"
    if supercategory == "food waste":
        return "organic_waste"
    if "glass" in lower_name:
        return "glass"
    if any(token in lower_name for token in ("can", "foil", "metal")):
        return "metal_can"
    if any(token in lower_name for token in ("paper", "carton", "tissue", "box")):
        return "paper_cardboard"
    if any(
        token in lower_name
        for token in ("plastic", "foam", "tupperware", "tube", "glove", "utensil", "straw")
    ):
        return "rigid_plastic"
    return "mixed_waste"


def resolve_trashnet_category(category: dict[str, Any]) -> str | None:
    """Map TrashNet's six source labels into the canonical training labels."""
    name = str(category.get("name", "")).strip().casefold()
    return TRASHNET_LABEL_MAP.get(name)


def resolve_category(
    category: dict[str, Any],
    *,
    resolver: str = "direct",
    explicit_mapping: dict[str, str] | None = None,
    label_override: str | None = None,
) -> str | None:
    """Resolve a source label, allowing config mappings to take precedence."""
    label = str(label_override if label_override is not None else category.get("name", "")).strip()
    mapping = {str(key).casefold(): value for key, value in (explicit_mapping or {}).items()}
    if label.casefold() in mapping:
        return mapping[label.casefold()]
    if resolver == "taco":
        return resolve_taco_category(category)
    if resolver == "trashnet":
        return resolve_trashnet_category(category)
    if resolver != "direct":
        raise ValueError(f"Unknown taxonomy resolver: {resolver}")
    return DIRECT_ALIASES.get(label.casefold())
