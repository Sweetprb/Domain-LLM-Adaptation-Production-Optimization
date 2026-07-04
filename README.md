# Domain-Specific IT Helpdesk LLM Optimization

A production-oriented domain adaptation and inference optimization pipeline built using TinyLlama and QLoRA for IT Helpdesk support tasks.

## Problem Statement

General-purpose LLMs often lack domain-specific knowledge required for IT Helpdesk troubleshooting scenarios. This project adapts a lightweight open-source LLM to provide accurate responses for enterprise IT support use cases while minimizing deployment cost.

## Features

- Domain-specific corpus creation and cleaning
- Instruction dataset generation
- QLoRA fine-tuning using 4-bit quantization
- Multiple decoding strategy evaluation
- Speculative decoding implementation
- Cost and throughput optimization analysis

## Dataset

### Source Documents
- Ubuntu Documentation
- RedHat Documentation
- Microsoft Intune Documentation
- ServiceNow Documentation
- ITIL Documentation

### Data Processing
- Length filtering
- Hash-based deduplication
- Language detection
- Text normalization

Final corpus:
- 5 documents
- 399 pages
- 60 instruction-response examples

## Model Architecture

| Component | Value |
|-----------|-------|
| Base Model | TinyLlama-1.1B |
| Parameters | 1.1 Billion |
| Quantization | 4-bit NF4 |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| Trainable Parameters | 2.25 Million |

## Fine-Tuning Configuration

- QLoRA
- PEFT
- TRL
- Hugging Face Transformers
- BitsAndBytes

## Decoding Strategies Evaluated

- Greedy Search
- Beam Search
- Top-K Sampling
- Top-P Sampling
- Temperature Sampling
- Hybrid Strategies

## Inference Optimization

Implemented speculative decoding using:

- Draft Model: SmolLM2-135M
- Target Model: TinyLlama-1.1B

Results:
- Up to 2.32× faster inference

## Performance Results

| Metric | Baseline | Optimized |
|--------|----------|-----------|
| Throughput | 14.1 tok/s | 20.7 tok/s |
| Cost Reduction | - | 32% |
| Speedup | - | 2.32x |

## Tech Stack

- Python
- Hugging Face Transformers
- PEFT
- TRL
- BitsAndBytes
- PyTorch
- Pandas
- NumPy

