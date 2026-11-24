# v11: overfit one fixed sample

**Question.** In v10 (one epoch, S = 40), a larger beta_lambda did not lower the training beta loss, even at λ = 16.
Two explanations:

1. **Noise floor.** Every v10 step sees a new sample, and its beta target is that sample's own OLS betas. Most of
   the target is sampling noise (Wu at 40 tasks: target noise 0.104 vs signal 0.19, standardized). A model that
   predicts choice probabilities cannot remove noise in samples it has not trained on yet, so no λ can lower the loss.
2. **Bug.** The beta gradient does not reach the update correctly.

**Test.** Train repeatedly on one fixed sample of 40 Wu tasks (seed 0): 300 steps, the same 40 tasks every step,
constant LR 4e-5 (v10's peak LR at S = 40), no weight decay, clipping 1.0, gradient split on. λ = 0, 1, 4.
A fixed sample's noise *can* be fit.

**Reading.**
- If the beta gradient works, the fixed sample's beta loss falls clearly faster with λ = 4 than with λ = 0 (λ = 1 in
  between). That supports explanation 1.
- If the beta loss falls at the same rate for every λ, explanation 2 is likely and the beta gradient needs checking.

**Metrics.** Training `raw_beta_loss` and `kl_loss` per step (`steps.csv`, train mode, LoRA dropout on), and the
fixed-sample validator every 10 steps (eval mode): its KL and std beta MSE against the 40 tasks' own human OLS betas,
standardized by the full-data sigma.
