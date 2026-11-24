| Setting | Value |
|:-------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Base decision models | KEV-0.8B (`bf75a6a`), KEV-4B (`6cfce5c`), KEV-9B (`db029f0`) |
| Weights precision | bfloat16 |
| Pointer-head temperature | 1.0 |
| Pointer-head dimension | 256 (trained in full) |
| LoRA rank $r$ | 64 |
| LoRA scaling $\alpha_{\mathrm{LoRA}}$ | 128 |
| LoRA dropout | 0.05 |
| LoRA target modules | All attention, MLP, and Gated DeltaNet projections |
| Optimizer | AdamW ($\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$) |
| Peak learning rate | $5 \times 10^{-6}$ |
| Learning-rate schedule | OneCycle: 10% warmup, cosine annealing, start = peak / 25, end = start / $10^{4}$ |
| Weight decay | 0.01 |
| Gradient-norm clipping | 1.0 |
| Epochs | 3 |
| Mini-batch size $\lvert S \rvert$ | 40 tasks from a single dataset |
| Random seed | 42 |
| Checkpoint selection | Lowest Flights $\mathrm{MSE}_\beta$, evaluated every 32 steps |
| Hardware | One NVIDIA H100 (80 GB) per run |
| Steps, Electricity | 465 per epoch (1,395 total) |
| Selected step, Electricity | 640 / 416 / 928 |
| Steps, Drones | 910 per epoch (2,730 total) |
| Selected step, Drones | 1,152 / 544 / 256 |
| Steps, Drones + Electricity | 1,375 per epoch (4,125 total) |
| Selected step, Drones + Electricity | 1,600 / 1,472 / 1,600 |
