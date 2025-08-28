# Reproduce Guidance

This directory contains scripts for evaluating the DyG-RAG on the three datasets. The code includes complete RAG and evaluation pipelines. For any questions or issues, please feel free to discuss with us.

## Overview

The evaluation pipeline processes temporal QA tasks using DyG-RAG. The system builds dynamic event graph from document corpus and answers time-sensitive questions through retrieval and generation.

## Files

- `timeqa.py` - Main evaluation script for TimeQA dataset
- `tempreason.py` - Main evaluation script for TempReason dataset
- `complextr.py` - Main evaluation script for Complex-TR dataset

## Datasets

**Note**: All datasets have been preprocessed and optimized for temporal reasoning tasks.

### Download

📥 **Preprocessed datasets**: [Google Drive](https://drive.google.com/drive/folders/1S8l4YGyBy4hywR2ca32UL3WyiKf5WYJ3?usp=sharing)

Download and extract the datasets to your `datasets/` directory. These are ready-to-use preprocessed datasets.

### Original Raw Sources (Reference)

- **TimeQA**: [TimeQA](https://github.com/wenhuchen/Time-Sensitive-QA)
- **TempReason**: [TempReason](https://github.com/DAMO-NLP-SG/TempReason)
- **ComplexTR**: [ComplexTR](https://huggingface.co/datasets/tonytan48/complex-tr)

### Required Files

- `datasets/TimeQA/Corpus.json` - TimeQA Document corpus
- `datasets/TimeQA/Question.json` - TimeQA Question set with ground truth answers
- `datasets/TempReason/Corpus.json` - TempReason document corpus
- `datasets/TempReason/Question.json` - TempReason questions
- `datasets/ComplexTR/Corpus.json` - Complex-TR document corpus
- `datasets/ComplexTR/Question.json` - Complex-TR questions

### Dataset Preparation

The datasets have been carefully curated and processed for optimal performance:

1. **Corpus Construction**

   - Built unified corpus by aggregating context from original question. Designed to simulate real-world RAG scenarios with diverse document collections.
   - Extracted and normalized textual content from various sources.
   - Removed duplicate and redundant documents. Merged overlapping content to reduce fragmentation.
2. **Question-Answer Processing**

   - Extracted and validated ground truth answers
   - Filtered out incomplete or ambiguous entries
   - Assigned unique identifiers for tracking

## Quick Start

1. **Configure environment:**

   ```bash
   # Set environment variables (or provide interactively)
   export VLLM_BASE_URL="http://127.0.0.1:8000/v1"
   export QWEN_BEST="qwen-14b" 
   export LOCAL_BGE_PATH="/path/to/your/bge-m3-model"
   ```
2. **Run evaluation:**

   ```bash
   python timeqa.py
   ```
3. **Check results:**
   Results are saved to `results_mode-dynamic_topk-20.json`

## Configuration Options


| Parameter     | Default        | Description                                       |
| ------------- | -------------- | ------------------------------------------------- |
| `QUERY_MODE`  | `"dynamic"`    | Query processing mode(only support 'dynamic' now) |
| `QUERY_TOP_K` | `20`           | Number of top results to retrieve                 |
| `CONCURRENCY` | `5`            | Maximum concurrent queries                        |
| `WORK_DIR`    | `"timeqa_dir"` | Working directory for GraphRAG                    |

## Output Format

Results are saved in JSON format with the following structure:

```json
{
  "metadata": {
    "total_questions": 1000,
    "completed_questions": 1000,
    "total_time": 3600.5,
    "avg_time_per_question": 3.6,
    "timestamp": "2024-01-01 12:00:00",
    "query_mode": "dynamic",
    "query_top_k": 20
  },
  "results": [
    {
      "question_id": "q1",
      "question": "When did the event occur?",
      "answer": "Generated answer...",
      "golden_answer": "Ground truth answer",
      "query_time": 2.5,
      "status": "success"
    }
  ]
}
```

## Performance Monitoring

The system provides detailed logging including:

- Document processing speed and statistics
- Query processing times and success rates
- Error rates and retry statistics

## Evaluation Metrics

The system calculates:

- **F1 Score**: Harmonic mean of precision and recall
- **Accuracy**: Exact match accuracy for answers
- **Response Time**: Average query processing time

**Note**: The evaluation code is based on the implementation from [JayLZhou/GraphRAG](https://github.com/JayLZhou/GraphRAG).
