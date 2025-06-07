# Memvid - (Claims of a ) Video-Based AI Memory 🧠📹

### This repo is just as it was when I forked it except ...

I have added some testing which leads me to believe the only thing revolutionary about this is the gall of the developer who posted it in the first place.  I prefer to avoid trash talking fellow software developers.  Let me just say that perhaps the errors are all mine in either understanding or execution.  Having said that, memvid appears to be a non-starter.

My script compare_vector_stores.py produced the following results, which speak for themselves.


## FAISS vs memvid
###Vector Store Comparison Report

Generated: 2025-06-06 13:05:00

The test file use is <strong><href a="https://arxiv.org/pdf/2506.05296">Control Tax: The Price of Keeping AI in Check</a></strong>

<p>
https://arxiv.org/pdf/2506.05296
</p>

Roughly 2MB pdf

### Artifact Creation Performance

- **Memvid creation time**: 6.38 seconds
- **FAISS creation time**: 0.94 seconds
- **Speed difference**: 6.76x

### Storage Comparison

- **Memvid video size**: 1.60 MB
- **FAISS index size**: 0.23 MB
- **Size ratio**: 6.92x

### Search Performance

| Query | Memvid Time (s) | FAISS Time (s) | Speed Ratio |
|-------|-----------------|----------------|-------------|
| What is the main topic?... | 0.067 | 0.007 | 9.33x |
| Summarize the key points... | 0.027 | 0.006 | 4.15x |

**Average search times:**
- Memvid: 0.047s
- FAISS: 0.007s

## Result Quality Comparison

### Query: What is the main topic?

**Overlap in top-5 results**: 100.0%

### Query: Summarize the key points

**Overlap in top-5 results**: 100.0%

## LLM Response Comparison

### Query 1: What is the main topic?

#### Memvid Response:

<p>
The main topic of the document, as deduced from contexts provided, revolves around papers submitted to NeurIPS (Conference on Neural Information Processing Systems). 
Specifically, the content includes guidelines and questions related to ethical considerations, societal impacts, experimental reproducibility, and the use and evaluation of large language models (LLMs), especially in connection with backdoors in code. The contexts discuss the necessity for IRB approvals for human subject research, declaration of LLM usage if they impact core methodologies, analysis of potential societal impacts, and descriptions of safeguards to manage risks.


#### FAISS Response:
<p>
The main topic of the provided context is a research paper's adherence to ethical guidelines, experimental result reproducibility, and the use of Large Language Models (LLMs) in the context of creating and detecting backdoors in machine learning models. The text references specifics related to the ethical considerations for human subjects, IRB approvals, the declaration of LLM usage, the societal impact of the work, and the conditions for ensuring that experimental results are reproducible.


---

### Query 2: Summarize the key points

#### Memvid Response:

<p>
Based on the context provided, here are the key points:

1. **Backdoor Detection Task**: There is a focus on identifying "backdoors" in code—deliberate flaws allowing unauthorized actions. A suspiciousness rating system is mentioned, urging caution in evaluating code to determine the likelihood of a backdoor's presence.

2. **Model Performance on Tasks**: Various models, such as GPT 3.5 Turbo and others, are evaluated on the APPS benchmark for their ability to detect backdoors. The performance is compared across models from different providers, reflecting their coding capabilities and effectiveness in backdoor detection.

3. **Ethical and Transparent Research**: Authors are encouraged to be transparent about the limitations, assumptions, and scalability of their proposed algorithms. They are urged to openly discuss potential privacy and fairness issues, contributing to community norms that prioritize integrity.

4. **Backdooring Methodology**: The brainstorming and critique prompts guide the development of subtly incorrect solutions that pass official tests but fail under specific conditions. This indicates a structured approach to developing and testing backdoor vulnerabilities.

5. **Model Surveillance**: Multiple configurations of models are used in various monitor-attacker pairings to assess suspiciousness scores, reflecting the models' ability to detect backdoors in both benign and programmed conditions.

These points reflect the broader themes of evaluating model performance in detecting artificial flaws and maintaining ethical standards in computational research.


#### FAISS Response:

<p>
The text discusses the concept of introducing a backdoor into software code, specifically in the context of evaluating how various AI models perform in both benign and backdoored scenarios. The explanation seems to involve a process where code is intentionally modified to include a subtle flaw that may not be easily detected but fails on a specific valid input. Additionally, the provided context includes information about an experiment conducted using models to detect such backdoors, measuring their success and suspiciousness ratings. Various models, like GPT-3.5 Turbo and Sonnet 3.7, are mentioned, and there are different histogram and performance analysis figures for model performance in detecting backdoors. The excerpt also references a competition problem dealing with clocks and space-time paradoxes and suggests that the AI models generate solutions, with examples of correct and backdoored solutions provided as part of the analysis. The text seems to focus on testing the models' ability to spot or ignore these backdoors, potentially as a benchmark or part of a security evaluation process.

---
