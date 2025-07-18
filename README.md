# DeltaBench Evaluation Project

This project contains the DeltaBench evaluation framework and analysis tools for evaluating language model performance on mathematical reasoning tasks with section-by-section error detection.

## Project Structure

```
├── README.md                    # This file
├── deltabench_original.py       # Main evaluation script
├── requirements.txt             # Python dependencies
├── description.md              # Project description
├── processbench_reproduction_guide.md
├── data/                       # Input datasets
│   ├── Deltabench_v1.csv
│   └── Deltabench_v1.jsonl
├── results/                    # Evaluation results
│   └── Deltabench_v1_gpt-4o-mini_deltabench.jsonl
├── scripts/                    # Analysis scripts
│   └── analyze_deltabench_results.py
└── visualizations/             # Generated plots and charts
    ├── deltabench_performance_analysis_macro.png
    └── deltabench_performance_analysis_micro.png
```

## Usage

### Running Evaluation

To evaluate a model on the DeltaBench dataset:

```bash
python deltabench_original.py \
    --call_modelname gpt-4o-mini \
    --dataset Deltabench_v1 \
    --api-key YOUR_OPENAI_API_KEY
```

This will:
1. Load the dataset from `data/Deltabench_v1.jsonl`
2. Call the specified model to critique mathematical solutions
3. Parse and evaluate the model's error detection performance
4. Save results to `results/Deltabench_v1_gpt-4o-mini_deltabench.jsonl`

### Analyzing Results

To analyze the evaluation results and generate visualizations:

```bash
cd scripts
python analyze_deltabench_results.py
```

This will:
1. Load results from the `results/` folder
2. Calculate micro and macro performance metrics
3. Generate position-based performance analysis
4. Create visualizations showing performance degradation by error position
5. Save plots to the `visualizations/` folder

## Results Format

Each line in the results file contains:
- `id`: Unique problem identifier
- `question`: The mathematical problem
- `sections_content`: Model's step-by-step solution
- `predicted_sections`: Sections identified as containing errors
- `true_sections`: Ground truth error sections
- `precision`, `recall`, `f1_score`: Individual problem metrics
- `judge`: Binary indicator (1 if model found any errors, 0 otherwise)
- `parsing_success`: Whether the critique was successfully parsed

## Key Findings

### Overall Performance (GPT-4o-mini)
- **Macro**: Precision: 0.363, Recall: 0.539, F1: 0.393
- **Micro**: Precision: 0.311, Recall: 0.502, F1: 0.384

### Performance by Error Position
The analysis shows clear performance degradation as error positions increase:
- **Position 1**: F1 ≈ 0.56 (best performance)
- **Positions 2-4**: F1 ≈ 0.43-0.53 (good performance)  
- **Positions 5-7**: F1 ≈ 0.20-0.42 (moderate performance)
- **Positions 8+**: F1 ≈ 0.20-0.30 (poor performance)

### Distribution Insights
- 40% of problems have errors in the first 3 sections
- Early errors are much more common than later errors
- The model performs significantly better on problems with early errors

## Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

Main dependencies:
- openai
- pandas
- numpy
- matplotlib
- tqdm
- aiohttp

## Notes

- The evaluation uses OpenAI's API and requires a valid API key
- Results are automatically retried for failed API calls
- The analysis supports both macro and micro averaging approaches
- Visualizations are saved as high-resolution PNG files