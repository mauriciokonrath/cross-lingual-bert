# Limits and Surprises of Cross-Lingual BERT Under Data Scarcity: A Portuguese Case Study in Sensitive-Text Classification

This repository contains text classification implementations using different BERT model variants with Portuguese datasets.

## Repository Structure

The repository is organized into four main folders, each representing a different BERT model:

- **bert/** - Base BERT (English)
- **mbert/** - Multilingual BERT
- **bert_chines/** - Chinese BERT
- **bertimbau/** - BERTimbau (Portuguese pre-trained BERT)

Each folder contains the same training code adapted for the respective model, along with execution results.

## File Naming Convention

Files follow this naming convention:

```
[NUMBER]PORTUGUES_[MODEL]_base_cased_bertimbau
```

Where:
- **NUMBER**: Number of annotated examples per class used
- **PORTUGUES**: Dataset in Portuguese
- **MODEL**: BERT variant used

Examples:
- `24PORTUGUES_bert_base_cased_bertimbau`: 24 examples per class
- `97PORTUGUES_bert_base_cased_bertimbau`: 97 examples per class
- `243PORTUGUES_bert_base_cased_bertimbau`: 243 examples per class

Additionally, special evaluation files:
- **MC_bert**: Confusion matrices
- **RTE_bert**: Recognizing Textual Entailment evaluations
- **STS_bert**: Semantic Textual Similarity evaluations

## Requirements

```
pip install simpletransformers torch transformers pandas scikit-learn matplotlib seaborn tqdm
```

## How to Run

1. Clone the repository
2. Run the training pipeline:
```python
# Example for BERTimbau with 97 examples per class
model, label_map, label_names, eval_df, metrics_history, advanced_results = run_training_pipeline_with_advanced_eval(
    file_path='dataset.tsv',
    delimiter='\t',
    text_column='utterance',
    label_column='fine_label',
    model_name="neuralmind/bert-base-portuguese-cased",  # BERTimbau
    num_epochs=20,
    learning_rate=3e-5,
    batch_size=16,
    balance_classes=False,
    run_advanced=True
)
```

3. For other models, change the `model_name` parameter:
   - BERT: `"google-bert/bert-base-cased"`
   - mBERT: `"bert-base-multilingual-cased"`
   - Chinese BERT: `"bert-base-chinese"`

## Available Evaluations

- **Basic**: Accuracy, precision, recall, F1-score, confusion matrix
- **Advanced**:
  - NER (Named Entity Recognition)
  - RTE (Recognizing Textual Entailment)
  - STS (Semantic Textual Similarity)

## Results

Results from each execution are saved in their respective folders, including:
- Performance graphs by epoch
- Confusion matrices
- Evaluation metrics
- Keyword analysis by category

## Usage Example

```python
# Load a trained model
model = ClassificationModel(
    "bert",
    "./bertimbau/97PORTUGUES_bert_base_cased_bertimbau/model",
    use_cuda=torch.cuda.is_available()
)

# Make predictions
predictions, raw_outputs = model.predict(["This document contains information about chemical absorption in the skin"])
```
