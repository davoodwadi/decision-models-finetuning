"""Grouped trainer subclass that augments preserved persona question groups with exact k-trait marginals."""

import random

from data.marginals import (
    DEFAULT_K_SCHEDULE,
    PersonaMarginalLookup,
    build_marginal_epoch_groups,
    validate_k_schedule,
)
from training.trainer import GroupedTrainer


class MarginalGroupedTrainer(GroupedTrainer):
    """Grouped trainer supervising exact empirical `k`-trait marginals across preserved persona question groups."""

    def __init__(
        self,
        *,
        train_records,
        eval_records,
        save_dir,
        marginal_lookup,
        k_schedule=DEFAULT_K_SCHEDULE,
        **kwargs,
    ):
        if not isinstance(marginal_lookup, PersonaMarginalLookup):
            raise TypeError(
                f"marginal_lookup must be a PersonaMarginalLookup instance, got {type(marginal_lookup).__name__}"
            )
        self.marginal_lookup = marginal_lookup
        self.k_schedule = validate_k_schedule(k_schedule)

        super().__init__(
            train_records=train_records,
            eval_records=eval_records,
            save_dir=save_dir,
            **kwargs,
        )

    def _epoch_groups_for(self, epoch_idx):
        epoch_rng = random.Random(f"{self.seed}:epoch:{epoch_idx}")
        sampled = (
            epoch_rng.sample(self.train_records, self.n_per_epoch)
            if self.samples_per_epoch is not None
            else self.train_records
        )
        return build_marginal_epoch_groups(
            sampled,
            self.marginal_lookup,
            max_questions=self.max_questions_per_group,
            max_chars=self.max_chars_per_group,
            rng=epoch_rng,
            k_schedule=self.k_schedule,
        )

    def _run_signature(self):
        sig = super()._run_signature()
        sig["trainer"] = "marginal_grouped"
        sig["k_schedule"] = list(self.k_schedule)
        sig["include_country"] = self.marginal_lookup.include_country
        sig["num_unique_personas"] = self.marginal_lookup.num_unique_personas
        return sig

    def describe(self):
        info = super().describe()
        return {
            "trainer": "MarginalGroupedTrainer",
            "k_schedule": self.k_schedule,
            "unique_personas": self.marginal_lookup.num_unique_personas,
            "trait_value_masks": len(self.marginal_lookup.trait_masks),
            **info,
        }
