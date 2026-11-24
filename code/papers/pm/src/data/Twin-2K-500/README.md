---
configs:
- config_name: full_persona
  data_files:
  - split: data
    path: full_persona/chunks/*.parquet
- config_name: wave_split
  data_files:
  - split: data
    path: wave_split/chunks/*.parquet
annotations_creators:
- expert-generated
language_creators:
- expert-generated
language:
- en
license:
- cc-by-4.0
multilinguality:
- monolingual
pretty_name: Twin-2K-500
size_categories:
- 1B<n<10B
source_datasets:
- original
task_categories:
- text-classification
- text-generation
- question-answering
task_ids:
- multi-class-classification
- language-modeling
---

# Twin-2K-500 Dataset

https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500

This dataset Twin-2K-500 contains comprehensive persona information from a representative sample of 2,058 US participants, providing rich demographic and psychological data. The dataset is specifically designed for building digital twins for LLM simulations.


- **More information on how to use this dataset can be found in our [Documentation](https://digital-twin-simulation-version2.readthedocs.io/en/latest/index.html) and [GitHub repository](https://github.com/tianyipeng-lab/Digital-Twin-Simulation).**
- **Details on how the dataset was generated are available in our [Paper](https://arxiv.org/abs/2505.17479).**

## Dataset Creation

Twin-2K-500 Dataset contains survey responses collected across four waves of data collection. The first three waves (launched one week apart) contained a mixture of demographic questions, psychological scales, cognitive performance questions, economic preferences questions, and behavioral economics experiments. Wave 4 (launched two weeks after wave 3) repeated the heuristics and biases experiments from the first three waves, providing a measure of test-retest reliability.

The data collection was designed to capture a diverse and representative sample of the US population, with participants from different geographic regions, age groups, and socioeconomic backgrounds.

## Dataset Structure and Format

Twin-2K-500 Dataset is organized into five folders, each with its specific format and purpose:

### 1. Full Persona Folder
This folder contains complete persona information for each participant. The data is split into chunks for easier processing:

- `pid`: Participant ID
- `persona_text`: Complete survey responses in text format, including all questions and answers. For questions that appear in both waves 1-3 and wave 4, the wave 4 responses are used.
- `persona_summary`: A concise summary of each participant's key characteristics and responses, designed to provide a quick overview without needing to process the full survey data. This summary captures the essential traits and patterns in the participant's responses.
- `persona_json`: Complete survey responses in JSON format, following the same structure as persona_text. The JSON file is useful if a subset of questions wanted to be excluded or revised.

### 2. Wave Split Folder
This folder is designed for testing and evaluating different LLM persona creation methodologies (from prompt engineering to RAG, fine-tuning, and RLHF):

- `pid`: Participant ID
- `wave1_3_persona_text`: Persona information from waves 1-3 in text format, including questions that did not appear in wave 4. This can be used as training data for creating personas.
- `wave1_3_persona_json`: Persona information from waves 1-3 in JSON format, following the same structure as wave1_3_persona_text.
- `wave4_Q_wave1_3_A`: Wave 4 questions with answers from waves 1-3, useful for human test-retest evaluation.
- `wave4_Q_wave4_A`: Wave 4 questions with their actual answers from wave 4, serving as ground truth for evaluating persona prediction accuracy.

The Wave Split Folder is particularly useful for:
- Training persona creation models using wave1-3 data
- Evaluating how well the created personas can predict wave 4 responses
- Comparing different LLM-based approaches (prompt engineering, RAG, fine-tuning, RLHF) for persona creation
- Testing the reliability and consistency of persona predictions across different time periods

### 3. Question Catalog and Human Response CSV Folder

This folder provides a clean, structured question catalog and standardized CSV response files, designed for researchers who want to analyze the data for correlations, heterogeneous treatment effects, and other statistical analyses.

- **`question_catalog.json`**: A catalog of all 256 unique QuestionIDs from the survey (corresponding to ~500 individual questions participants answered, as many QuestionIDs contain multiple sub-items such as Matrix rows or Slider statements). Includes detailed metadata including question type, options, scale labels, and the corresponding CSV column names. See [`question_catalog_README.md`](question_catalog_and_human_response_csv/question_catalog_README.md) for the full schema, the breakdown of QuestionIDs vs. questions asked vs. CSV columns, and how they map to the paper's categories.
- **`wave1_3_response.csv`** / **`wave1_3_response_label.csv`**: Human responses from waves 1-3 (2,058 participants × 761 columns), in numeric and text label formats respectively.
- **`wave4_response.csv`** / **`wave4_response_label.csv`**: Human responses from wave 4 (2,058 participants × 127 columns), in numeric and text label formats respectively.

All CSVs have `pid` as the first column. The remaining columns are named by `{QID}` or `{QID}_{SubID}` as documented in the question catalog.

### 4. LLM Simulation Results Folder

This folder contains LLM simulation outputs for wave 4 questions, enabling comparison between LLM-generated and human responses.

- **`GPT4.1-mini-simulation-llm-vs-human`**: Wave 4 CSV simulated by GPT4.1-mini (our default setup) along with the human CSV. This facilitates analysis of LLM simulation patterns.
- **`llm_simulations_all_specifications`**: Comprehensive LLM simulation results across multiple model configurations and persona formats, including:
  - **JSON Persona** simulations using GPT4.1 and GPT4.1-mini
  - **Text Persona** simulations with various configurations (default temperature, reasoning, repeating questions) using GPT4.1-mini and Gemini-Flash2.5
  - **LLM Finetuning** results (500 training samples) using GPT4.1-mini
  - **JSON Persona (Predicted Output)** variants

  Each specification folder contains:
  - **Accuracy Evaluation**: Distribution plots, correlation analyses, MAD accuracy summaries, and regression analyses comparing LLM predictions to ground truth
  - **CSV Comparisons**: Formatted CSV files comparing LLM-imputed responses with wave 1-3 and wave 4 human responses, available in both raw and labeled formats

- **`wave4_formatted_to_catalog_mapping.json`**: Maps the 126 column names in the LLM simulation CSVs (`responses_wave4_formatted.csv`) back to the standardized QuestionIDs in `question_catalog.json`. See [`wave4_formatted_to_catalog_mapping_README.md`](LLM_simulation_results/wave4_formatted_to_catalog_mapping_README.md) for details on the mapping schema and excluded columns.

### 5. Raw Data Folder

This folder provides access to the raw survey response files from Qualtrics, after anonymization and removal of sensitive columns. The raw data is attached for reference, but **for analysis purposes we recommend using the Question Catalog and Human Response CSV files (Section 3) and the LLM Simulation Results (Section 4) above**, which provide cleaner, standardized formats.

The folder contains the following files for each wave (1-4):
- **Labels CSV** (e.g., `wave_1_labels_anonymized.csv`): Contains survey answers as text.
- **Numbers CSV** (e.g., `wave_1_numbers_anonymized.csv`): Contains survey answers as numerical codes.
- **Questionnaire**: Questionnaire files are provided in the `questionnaire` subfolder. These files can help visualize the survey structure and question flows.

## Usage

```python
from datasets import load_dataset

# Load the dataset with specific configuration
full_persona = load_dataset("LLM-Digital-Twin/Twin-2K-500", "full_persona")
wave_split = load_dataset("LLM-Digital-Twin/Twin-2K-500", "wave_split")

# Example: Using wave_split for persona creation and evaluation
train_data = wave_split["data"]["wave1_3_persona_text"]  # or wave1_3_persona_json
test_questions = wave_split["data"]["wave4_Q_wave4_A"] # you want to remove the "Answers" from all questions
ground_truth = wave_split["data"]["wave4_Q_wave4_A"]
```

**We also provide a simple [Jupyter notebook](https://github.com/tianyipeng-lab/Digital-Twin-Simulation/blob/main/notebooks/demo_simple_simulation.ipynb) to simulate new questions with our personas.**

### JSON Format Example

```json
{
  "pid": "71",
  "data": {
    "ElementType": "Block",
    "BlockName": "Demographics",
    "BlockType": "Standard",
    "Questions": [
      {
        "QuestionID": "QID11",
        "QuestionText": "Which part of the United States do you currently live in?",
        "QuestionType": "MC",
        "Options": [
          "Northeast (PA, NY, NJ, RI, CT, MA, VT, NH, ME)",
          "Midwest (ND, SD, NE, KS, MN, IA, MO, WI, IL, MI, IN, OH)",
          "South (TX, OK, AR, LA, KY, TN, MS, AL, WV, DC, MD, DE, VA, NC, SC, GA, FL)",
          "West (WA, OR, ID, MT, WY, CA, NV, UT, CO, AZ, NM)",
          "Pacific (HI, AK)"
        ],
        "Settings": {
          "Selector": "SAVR",
          "SubSelector": "TX",
          "ForceResponse": "ON"
        },
        "Answers": {
          "SelectedByPosition": 1,
          "SelectedText": "Northeast (PA, NY, NJ, RI, CT, MA, VT, NH, ME)"
        }
      }
    ]
  }
}
```

## Considerations for Using the Data

### Social Impact

This dataset provides valuable insights into human personality traits and demographics, which can be used to develop more natural and human-like AI systems through digital twin simulations. Researchers can use this data to improve the efficiency of theory development and experimental design, while practitioners can leverage it for customer insights and product development.

### Discussion of Biases

The dataset may contain inherent biases from the survey collection process, including self-selection bias and social desirability bias in responses. Users of this dataset should be aware of these potential biases when drawing conclusions or training models.

### Other Known Limitations

- Survey responses are self-reported and may not always accurately reflect actual behaviors or characteristics
- The dataset represents a specific point in time and geographic context

## Citation

If you use this dataset in your research, please cite:

```
@dataset{twin2k500,
  author    = {Toubia, Olivier and Gui, George Z. and Peng, Tianyi and Merlau, Daniel J. and Li, Ang and Chen, Haozhe},
  title     = {Twin-2K-500: A Dataset for Building Digital Twins of 2,000 People},
  year      = {2025},
  publisher = {Hugging Face},
  howpublished = {\url{https://arxiv.org/abs/2505.17479}}
}
```

## License

This dataset is licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).