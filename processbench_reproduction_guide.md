# ProcessBench Reproduction Guide: Process-Level Error Detection

This guide explains how to reproduce the ProcessBench methodology for evaluating process-level error detection in step-by-step reasoning, allowing developers to apply this approach to their own datasets with section-by-section annotations.

## Table of Contents
1. [Overview](#overview)
2. [Data Format Requirements](#data-format-requirements)
3. [Methodology](#methodology)
4. [Implementation Guide](#implementation-guide)
5. [Voting Mechanism](#voting-mechanism)
6. [Evaluation Pipeline](#evaluation-pipeline)
7. [Adaptation Guidelines](#adaptation-guidelines)
8. [Complete Working Example](#complete-working-example)

## Overview

### What is ProcessBench?

ProcessBench is a benchmark that evaluates AI models' ability to identify where errors first occur in step-by-step reasoning processes. Unlike traditional benchmarks that only check final answers, ProcessBench focuses on **process-level evaluation**.

### Core Concept

Given a step-by-step solution to a problem:
- **Task**: Identify the index of the first step containing an error
- **Output**: Step index (0, 1, 2, ...) or -1 if no errors exist
- **Evaluation**: Compare predictions against ground truth error labels

### Key Advantages

1. **Process Understanding**: Tests whether models can trace reasoning and identify error sources
2. **Error Localization**: Pinpoints exact locations of reasoning failures
3. **Robustness**: Evaluates both error detection and correct solution recognition

## Data Format Requirements

### Required Fields

Each example in your dataset must contain:

```json
{
  "id": "unique_identifier",
  "problem": "The problem statement",
  "steps": [
    "Step 0: First reasoning step",
    "Step 1: Second reasoning step",
    "Step 2: Third reasoning step",
    "..."
  ],
  "label": 1  // Index of first error step, or -1 if no errors
}
```

### Field Descriptions

- **`id`**: Unique identifier for the example
- **`problem`**: The original problem statement
- **`steps`**: Array of reasoning steps (paragraphs/sentences)
- **`label`**: Ground truth error location
  - `label = -1`: No errors (completely correct solution)
  - `label = 0`: Error in first step (index 0)
  - `label = 1`: Error in second step (index 1)
  - `label = N`: Error in step N+1 (0-indexed)

### Label Annotation Guidelines

#### For Error Cases (label ≥ 0):
- **First Error**: Label represents the **earliest** step containing an error
- **Complete Solutions**: Include all steps, even after the error occurs
- **Error Types**: Logical errors, calculation mistakes, incorrect assumptions, etc.

#### For Correct Cases (label = -1):
- **No Errors**: All reasoning steps are correct
- **Final Answer**: Must be correct
- **Complete Chain**: All intermediate steps must be valid

### Example Data Structure

```json
[
  {
    "id": "example_1",
    "problem": "John has 10 apples. He gives 3 to Mary and 2 to Bob. How many apples does John have left?",
    "steps": [
      "John starts with 10 apples.",
      "He gives 3 apples to Mary, so he has 10 - 3 = 7 apples left.",
      "He gives 2 apples to Bob, so he has 7 - 2 = 5 apples left.",
      "Therefore, John has 5 apples left."
    ],
    "label": -1  // No errors
  },
  {
    "id": "example_2", 
    "problem": "Sarah bought 15 books. She read 1/3 of them. How many books did she read?",
    "steps": [
      "Sarah bought 15 books.",
      "She read 1/3 of them, which is 15 * (1/3) = 15/3 = 6 books.",
      "Wait, let me recalculate: 15 * (1/3) = 5 books.",
      "Therefore, Sarah read 5 books."
    ],
    "label": 1  // Error in step 1 (incorrect calculation)
  }
]
```

## Methodology

### 1. Problem Formulation

ProcessBench frames error detection as a **classification task**:
- **Input**: Problem + Step-by-step solution
- **Output**: Index of first error step or -1
- **Method**: Large language models with critique prompting

### 2. Template System

The evaluation uses a **critique template** that instructs models to:
1. Review the solution paragraph by paragraph
2. Identify logical errors, calculation mistakes, or incorrect reasoning
3. Return the index of the earliest error found
4. Return -1 if no errors are detected

### 3. Evaluation Strategy

**Two-Category Evaluation**:
- **Error Detection**: Accuracy on examples with errors (label ≠ -1)
- **Correct Recognition**: Accuracy on examples without errors (label = -1)
- **Combined Metric**: F1 score of the two accuracies

## Implementation Guide

### Step 1: Install Dependencies

```bash
pip install openai datasets numpy tqdm
```

### Step 2: Create the Critique Template

```python
CRITIQUE_TEMPLATE = """The following is a problem and a solution (split into paragraphs, enclosed with tags and indexed from 0):

[Problem]

{problem}

[Solution]

{tagged_response}

Your task is to review and critique the solution paragraph by paragraph. Once you identify an error in a paragraph, return the index of the paragraph where the earliest error occurs. Otherwise, return the index of -1 (which typically denotes "not found").

Please put your final answer (i.e., the index) in \\boxed{{}}.
"""
```

### Step 3: Input Processing Function

```python
def prepare_input(template, input_data):
    """
    Prepare input for the model by formatting the problem and solution.
    
    Args:
        template: The critique template string
        input_data: Dictionary containing 'problem' and 'steps'
    
    Returns:
        List of message dictionaries for the API
    """
    problem = input_data['problem']
    steps = input_data['steps']
    
    # Tag each step with paragraph indices
    tagged_response = ''
    for idx, step in enumerate(steps):
        tagged_response += f'<paragraph_{idx}>\\n{step}\\n</paragraph_{idx}>\\n\\n'
    tagged_response = tagged_response.strip()
    
    # Format the prompt
    prompt = template.format(problem=problem, tagged_response=tagged_response)
    messages = [{'role': 'user', 'content': prompt}]
    return messages
```

### Step 4: Answer Extraction

```python
import re

def extract_answer(response_text):
    """
    Extract the answer from the model's response.
    
    Args:
        response_text: The model's response string
    
    Returns:
        Integer index or None if not found
    """
    boxed_pattern = r'\\\\boxed\\{([^}]*)\\}'
    matches = re.findall(boxed_pattern, response_text)
    if matches:
        try:
            return int(matches[-1].strip())
        except ValueError:
            return None
    return None
```

### Step 5: API Integration

```python
from openai import OpenAI

def call_model_api(client, messages, model="gpt-4o-mini", temperature=0.0, max_tokens=4096):
    """
    Call the OpenAI API with retry logic.
    
    Args:
        client: OpenAI client instance
        messages: List of message dictionaries
        model: Model name to use
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
    
    Returns:
        API response object
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=42  # For reproducibility
        )
        return response
    except Exception as e:
        print(f"API error: {e}")
        raise
```

## Voting Mechanism

### Overview

The voting mechanism improves robustness by generating multiple predictions and selecting the most common answer.

### Implementation

```python
def process_with_voting(client, template, input_data, model="gpt-4o-mini", n_votes=8):
    """
    Process a single example with voting mechanism.
    
    Args:
        client: OpenAI client
        template: Critique template
        input_data: Input example
        model: Model name
        n_votes: Number of votes to generate
    
    Returns:
        Dictionary with prediction and voting details
    """
    messages = prepare_input(template, input_data)
    
    # Generate multiple responses
    response = call_model_api(
        client, messages, model,
        temperature=0.7,  # Higher temperature for diversity
        max_tokens=4096
    )
    
    # For voting, we need multiple API calls
    votes = []
    for _ in range(n_votes):
        response = call_model_api(client, messages, model, temperature=0.7)
        generated_text = response.choices[0].message.content
        prediction = extract_answer(generated_text)
        if prediction is not None:
            votes.append(prediction)
    
    # Determine final prediction by majority vote
    if votes:
        from collections import Counter
        vote_counts = Counter(votes)
        final_prediction = vote_counts.most_common(1)[0][0]
    else:
        final_prediction = None
    
    return {
        'prediction': final_prediction,
        'votes': votes,
        'vote_distribution': dict(Counter(votes)) if votes else {}
    }
```

### Voting Parameters

- **Temperature**: 0.7 (higher for diversity)
- **Number of votes**: 8 (default, can be adjusted)
- **Consensus method**: Majority vote (most frequent prediction)
- **Fallback**: If no valid predictions, return None

## Evaluation Pipeline

### Complete Evaluation Function

```python
def evaluate_processbench(data, client, template, model="gpt-4o-mini", use_voting=False, n_votes=8):
    """
    Evaluate a dataset using ProcessBench methodology.
    
    Args:
        data: List of examples with 'problem', 'steps', and 'label'
        client: OpenAI client
        template: Critique template
        model: Model name
        use_voting: Whether to use voting mechanism
        n_votes: Number of votes (if using voting)
    
    Returns:
        Dictionary with results and metrics
    """
    results = []
    
    for example in tqdm(data, desc="Processing examples"):
        if use_voting:
            result = process_with_voting(client, template, example, model, n_votes)
        else:
            messages = prepare_input(template, example)
            response = call_model_api(client, messages, model, temperature=0.0)
            generated_text = response.choices[0].message.content
            prediction = extract_answer(generated_text)
            result = {'prediction': prediction}
        
        # Add ground truth and match information
        result.update({
            'id': example['id'],
            'label': example['label'],
            'match': prediction == example['label'] if prediction is not None else False,
            'generated_text': generated_text if not use_voting else None
        })
        
        results.append(result)
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    return {
        'results': results,
        'metrics': metrics
    }

def calculate_metrics(results):
    """
    Calculate ProcessBench metrics.
    
    Args:
        results: List of result dictionaries
    
    Returns:
        Dictionary with accuracy metrics
    """
    # Split into error and correct examples
    error_examples = [r for r in results if r['label'] != -1]
    correct_examples = [r for r in results if r['label'] == -1]
    
    # Calculate accuracies
    if error_examples:
        error_acc = sum(r['match'] for r in error_examples) / len(error_examples) * 100
    else:
        error_acc = 0.0
    
    if correct_examples:
        correct_acc = sum(r['match'] for r in correct_examples) / len(correct_examples) * 100
    else:
        correct_acc = 0.0
    
    # Calculate F1 score
    if error_acc + correct_acc > 0:
        f1_score = 2 * error_acc * correct_acc / (error_acc + correct_acc)
    else:
        f1_score = 0.0
    
    return {
        'error_accuracy': error_acc,
        'correct_accuracy': correct_acc,
        'f1_score': f1_score,
        'error_count': len(error_examples),
        'correct_count': len(correct_examples),
        'total_count': len(results)
    }
```

## Adaptation Guidelines

### For Different Domains

#### 1. Mathematics → Science
```python
SCIENCE_TEMPLATE = """The following is a science problem and a solution (split into paragraphs, enclosed with tags and indexed from 0):

[Science Problem]

{problem}

[Solution]

{tagged_response}

Your task is to review and critique the solution paragraph by paragraph. Look for scientific errors, incorrect formulas, wrong units, or faulty reasoning. Once you identify an error in a paragraph, return the index of the paragraph where the earliest error occurs. Otherwise, return the index of -1.

Please put your final answer (i.e., the index) in \\boxed{{}}.
"""
```

#### 2. Mathematics → Programming
```python
PROGRAMMING_TEMPLATE = """The following is a programming problem and a solution (split into paragraphs, enclosed with tags and indexed from 0):

[Programming Problem]

{problem}

[Solution]

{tagged_response}

Your task is to review and critique the solution paragraph by paragraph. Look for logical errors, syntax mistakes, incorrect algorithms, or faulty reasoning. Once you identify an error in a paragraph, return the index of the paragraph where the earliest error occurs. Otherwise, return the index of -1.

Please put your final answer (i.e., the index) in \\boxed{{}}.
"""
```

### For Different Step Structures

#### 1. Code Steps
```python
def prepare_code_input(template, input_data):
    """Prepare input for code-based problems."""
    problem = input_data['problem']
    steps = input_data['steps']  # Each step is a code block or explanation
    
    tagged_response = ''
    for idx, step in enumerate(steps):
        tagged_response += f'<step_{idx}>\\n{step}\\n</step_{idx}>\\n\\n'
    
    prompt = template.format(problem=problem, tagged_response=tagged_response.strip())
    return [{'role': 'user', 'content': prompt}]
```

#### 2. Proof Steps
```python
def prepare_proof_input(template, input_data):
    """Prepare input for mathematical proofs."""
    problem = input_data['problem']
    steps = input_data['steps']  # Each step is a proof step
    
    tagged_response = ''
    for idx, step in enumerate(steps):
        tagged_response += f'<proof_step_{idx}>\\n{step}\\n</proof_step_{idx}>\\n\\n'
    
    prompt = template.format(problem=problem, tagged_response=tagged_response.strip())
    return [{'role': 'user', 'content': prompt}]
```

### Customization Parameters

```python
DOMAIN_CONFIGS = {
    'mathematics': {
        'template': CRITIQUE_TEMPLATE,
        'temperature': 0.0,
        'max_tokens': 4096,
        'voting_temperature': 0.7
    },
    'programming': {
        'template': PROGRAMMING_TEMPLATE,
        'temperature': 0.0,
        'max_tokens': 8192,  # Longer for code
        'voting_temperature': 0.8
    },
    'science': {
        'template': SCIENCE_TEMPLATE,
        'temperature': 0.0,
        'max_tokens': 4096,
        'voting_temperature': 0.7
    }
}
```

## Complete Working Example

### Sample Dataset

```python
sample_data = [
    {
        "id": "math_1",
        "problem": "A rectangle has length 8 and width 5. What is its area?",
        "steps": [
            "The area of a rectangle is length × width.",
            "Given: length = 8, width = 5",
            "Area = 8 × 5 = 40",
            "Therefore, the area is 40 square units."
        ],
        "label": -1  # No errors
    },
    {
        "id": "math_2",
        "problem": "John has 20 marbles. He gives 1/4 to his sister. How many marbles does he have left?",
        "steps": [
            "John starts with 20 marbles.",
            "He gives 1/4 of them to his sister: 20 × (1/4) = 5 marbles.",
            "He has 20 - 5 = 15 marbles left.",
            "Actually, let me recalculate: 20 × (1/4) = 6 marbles given away.",
            "So he has 20 - 6 = 14 marbles left."
        ],
        "label": 3  # Error in step 3 (recalculation is wrong)
    }
]
```

### Full Implementation

```python
import json
from openai import OpenAI
from tqdm import tqdm
import re
from collections import Counter

def main():
    # Initialize OpenAI client
    client = OpenAI(api_key="your-api-key")
    
    # Load your data
    with open('your_dataset.json', 'r') as f:
        data = json.load(f)
    
    # Define template
    template = CRITIQUE_TEMPLATE
    
    # Run evaluation
    results = evaluate_processbench(
        data=data,
        client=client,
        template=template,
        model="gpt-4o-mini",
        use_voting=True,
        n_votes=8
    )
    
    # Print results
    print("=== ProcessBench Evaluation Results ===")
    print(f"Error Accuracy: {results['metrics']['error_accuracy']:.1f}%")
    print(f"Correct Accuracy: {results['metrics']['correct_accuracy']:.1f}%")
    print(f"F1 Score: {results['metrics']['f1_score']:.1f}")
    print(f"Total Examples: {results['metrics']['total_count']}")
    
    # Save detailed results
    with open('processbench_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
```

### Expected Output

```
=== ProcessBench Evaluation Results ===
Error Accuracy: 65.2%
Correct Accuracy: 78.9%
F1 Score: 71.4
Total Examples: 100

Error samples: 45
Correct samples: 55
```

## Best Practices

### 1. Data Quality
- **Clear Steps**: Each step should be a distinct reasoning unit
- **Consistent Labeling**: Ensure error labels point to the truly first error
- **Complete Solutions**: Include all steps, even after errors

### 2. Model Selection
- **Capability**: Use models with strong reasoning abilities
- **Context Length**: Ensure sufficient context for long solutions
- **Consistency**: Use same model across evaluation for fair comparison

### 3. Evaluation
- **Balanced Dataset**: Include both error and correct examples
- **Multiple Runs**: Run evaluation multiple times for statistical significance
- **Error Analysis**: Examine failure cases to understand model limitations

### 4. Voting Optimization
- **Vote Count**: 8 votes provide good balance of accuracy and cost
- **Temperature**: 0.7 provides good diversity without too much randomness
- **Consensus**: Majority vote works well; consider weighted voting for confidence

## Troubleshooting

### Common Issues

1. **Low Accuracy**: Check data quality, template clarity, model capability
2. **API Errors**: Implement retry logic, rate limiting, error handling
3. **Inconsistent Results**: Use fixed seeds, check for data leakage
4. **High Costs**: Start with smaller subsets, optimize voting parameters

### Debugging Tips

```python
# Debug individual examples
def debug_example(example, prediction, generated_text):
    print(f"ID: {example['id']}")
    print(f"Ground Truth: {example['label']}")
    print(f"Prediction: {prediction}")
    print(f"Match: {prediction == example['label']}")
    print(f"Generated: {generated_text[:200]}...")
    print("-" * 50)
```

This comprehensive guide provides everything needed to reproduce the ProcessBench methodology on your own section-by-section annotated dataset. The approach is highly adaptable and can be customized for different domains while maintaining the core process-level evaluation framework.