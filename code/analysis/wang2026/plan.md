The focal experiment for the "Large language models for market research: A data-augmentation approach" in the `manuscript/lit/Large language models for market research: A data-augmentation approach`. 

The `covid` dataset in `code/data/covid`. 

See for the test that they ran for the LLM only samples, what was the focal metric they reported. 

we want to follow this structure:
1. We will start by establishing the benchmark. Query proprietary LLM (`gemini-3.5-flash-lite`) inside `code/proprietary-llms/gemini/gemini-3.5-flash-lite.ipynb` and compute the regression from Wang (2026). Since it is a text generating model, we will need to use the zero-shot prompt used by Wang (2026) to generate LLM responses.
2. We compare the regression against the regression with full human data using Wang comparison method. This will be our benchmark to beat.
3. Then query Kev 9B Base (without our finetuning) on the same set. A second baseline
4. Then query Kev 9B FT (our finetuned version from train-grouped-prototype)
5. Create a plot showing the errors of the baselines and our finetune. 