# v12: model scale (Kev 4B)

**Question.** Does a larger backbone change v8's result? v8 (Kev 800M) trained on Wu + Berke for one epoch and
validated on the unseen Kriner 2026 and Grummon 2025; λ had no detectable effect. v12 repeats v8's S = 40 runs at
λ = 0 and λ = 1 on `jaredpalmer/kev-4b`.

**Unchanged from v8.** Training datasets (Wu, Berke per vendor), validators (Kriner picks best.pt by val_loss,
Grummon 2025 held out), sample size 40 tasks, peak LR 4e-5 (the same sqrt rule), OneCycle, weight decay 0.01, clipping
1.0, seed 42, validation every 1,280 / 6,400 rows. LoRA rank 64, bf16, head temperature 1.0.

**Changed.** The backbone only. The micro chunk comes from the VRAM search (memory-only, not part of the run identity);
gradient checkpointing is off unless the search finds no fitting size.

**Reading.** The scale cell puts v8's 800M runs and v12's 4B runs side by side per validator: KL and beta MSE at the
step with the lowest val_loss, and the lowest beta MSE. Each model has one seed per λ, so a λ gap within a model needs
to be large to count; the 800M vs 4B gap is the main comparison.

**Caveat.** The LR was tuned on 800M. A worse 4B result could mean the LR is wrong for 4B, not that scale does not help.
