"""Class-to-material mapping for density and mass estimation."""

from __future__ import annotations

from typing import Any


DEFAULT_CLASS_TO_MATERIAL: dict[str, str] = {
    "plastic": "plastic",
    "plastic_battle": "plastic",
    "plastic_bottle": "plastic",
    "bottle": "plastic",
    "rigid_plastic": "plastic",
    "plastic_bag": "plastic",
    "glass": "glass",
    "glass_bottle": "glass",
    "glass_jar": "glass",
    "metal": "metal",
    "metal_can": "metal",
    "can": "metal",
    "paper": "paper",
    "paper_cardboard": "cardboard",
    "cardboard": "cardboard",
    "organic": "organic",
    "organic_waste": "organic",
    "food_waste": "organic",
    "textile": "textile",
    "textile_trash": "textile",
    "wood": "wood",
    "mixed_waste": "other",
    "trash": "other",
    "other": "other",
}


class ClassMaterialMapper:
    """Map detector object classes to material names used by density priors."""

    def __init__(self, mapping: dict[str, str] | None = None, default: str = "other") -> None:
        merged = dict(DEFAULT_CLASS_TO_MATERIAL)
        if mapping:
            merged.update({str(k): str(v) for k, v in mapping.items()})
        self.mapping = {key.lower(): value for key, value in merged.items()}
        self.default = default

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> ClassMaterialMapper:
        if not config:
            return cls()
        return cls(
            mapping=config.get("class_to_material"),
            default=str(config.get("default_material", "other")),
        )

    def material_for(self, class_name: str | None) -> str:
        if not class_name:
            return self.default
        key = str(class_name).strip().lower()
        if key in self.mapping:
            return self.mapping[key]
        for token, material in self.mapping.items():
            if token and token in key:
                return material
        return self.default
