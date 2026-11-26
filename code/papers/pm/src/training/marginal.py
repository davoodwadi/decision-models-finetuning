"""Marginal-augmented persona-grouped trainer supervising exact empirical k-trait marginals."""

import random

from data.marginals import (
    DEFAULT_K_SCHEDULE,
    MarginalLookup,
    augment_group_with_marginals,
    validate_k_schedule,
)
from training.grouped import GroupedTrainer


class MarginalGroupedTrainer(GroupedTrainer):
    """Grouped trainer replacing each group's target with exact empirical k-trait marginals."""

    trainer_name = "marginal_grouped"

    def __init__(
        self,
        *,
        marginal_lookup: MarginalLookup,
        k_schedule=DEFAULT_K_SCHEDULE,
        **kwargs,
    ):
        if not isinstance(marginal_lookup, MarginalLookup):
            raise TypeError(
                f"marginal_lookup must be a MarginalLookup instance, got {type(marginal_lookup).__name__}"
            )
        self.marginal_lookup = marginal_lookup
        self.k_schedule = validate_k_schedule(k_schedule)
        super().__init__(**kwargs)

    def _epoch_groups(self, epoch_idx: int, rng: random.Random) -> list[list[dict]]:
        base_groups = super()._epoch_groups(epoch_idx, rng)
        marginal_groups: list[list[dict]] = []
        for group in base_groups:
            full_persona = group[0]["personas"]
            k_choice = rng.choice(self.k_schedule)
            k_traits = len(full_persona) if k_choice == "full" else min(k_choice, len(full_persona))
            chosen = set(rng.sample(list(full_persona.keys()), k_traits))
            subpersona = {col: val for col, val in full_persona.items() if col in chosen}
            marginal_groups.append(augment_group_with_marginals(group, subpersona, self.marginal_lookup))
        return marginal_groups

    def describe(self) -> dict:
        return {
            **super().describe(),
            "k_schedule": list(self.k_schedule),
            "include_country": self.marginal_lookup.include_country,
            "unique_personas": self.marginal_lookup.num_unique_personas,
            "trait_value_masks": len(self.marginal_lookup.trait_masks),
        }
