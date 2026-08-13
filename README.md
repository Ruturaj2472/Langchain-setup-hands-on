# LangChain Hands-On Tutorials & Field Experiments

Welcome to this repository! This project serves as a comprehensive step-by-step guide and practical workshop for working with Large Language Models (LLMs), LangChain, and modern LLM application architecture.

The repository covers everything from basic chain initialization to advanced orchestration using **LangChain Expression Language (LCEL)**, as well as local open-source LLM inference using **Mistral-7B** with Hugging Face Transformers.

---

## 📋 Table of Contents
1. [Repository Overview](#repository-overview)
2. [Notebook Breakdown](#notebook-breakdown)
   - [1. Langchain_Setup_and_Simple_Chain.ipynb](#1-langchain_setup_and_simple_chainipynb)
   - [2. Mistral_7B.ipynb](#2-mistral_7bipynb)
   - [3. Langchain_LCEL.ipynb](#3-langchain_lcelipynb)
3. [Key Concepts & Key Learnings](#key-concepts--key-learnings)
4. [Prerequisites & Environment Setup](#prerequisites--environment-setup)
5. [How to Run](#how-to-run)

---

## 🛠️ Repository Overview

This collection of notebooks documents the evolution of building LLM workflows:
* **Legacy vs. Modern LangChain:** Exploring classic `LLMChain` mechanisms and transitioning to the functional, declarative **LCEL** pipeline.
* **API vs. Local Inference:** Integrating cloud-based API models (Cohere Command models) vs. quantized local open-source models (Mistral-7B via Hugging Face on GPU).
* **Multi-Model Orchestration:** Executing parallel model invocations and combining disparate outputs into unified, high-quality responses.

---

## 📓 Notebook Breakdown

### 1. `Langchain_Setup_and_Simple_Chain.ipynb`

* **File Name:** `Langchain_Setup_and_Simple_Chain.ipynb`
* **Description:**  
  This notebook establishes the foundational setup for building LLM applications with LangChain. It demonstrates how to authenticate with third-party providers (Cohere API via Google Colab secrets) and construct a baseline text-generation pipeline using prompt templates.
* **What Happens:**
  - Installs required packages (`langchain`, `cohere`, `langchain-cohere`, `langchain-classic`).
  - Securely fetches API keys using `google.colab.userdata`.
  - Defines a dynamic `PromptTemplate` with variable inputs (e.g., `"{adjective}"`).
  - Initializes the model (`ChatCohere` with `command-a-03-2025`) and binds it within an `LLMChain`.
  - Executes the chain using `.invoke()` and parses text output.
* **Key Learnings:**
  - Secure credential management in Google Colab environments.
  - Understanding legacy `LLMChain` mechanics and observing standard deprecation warnings favoring LCEL (`prompt | llm`).
  - Prompt variable interpolation and structured output extraction.

---

### 2. `Mistral_7B.ipynb`

* **File Name:** `Mistral_7B.ipynb`
* **Description:**  
  This notebook focuses on open-source, local LLM execution without relying on third-party API endpoints. It loads and runs the **Mistral-7B** open-weights instruction-tuned model on a T4 GPU instance using Hugging Face's `transformers` ecosystem.
* **What Happens:**
  - Sets up PyTorch and GPU-accelerated dependencies.
  - Downloads and caches the tokenizer and model weights for `Mistral-7B-Instruct-v0.2` / `Mistral-7B` from Hugging Face Hub.
  - Configures model parameters and text generation pipelines.
  - Generates text responses locally while managing memory constraints.
* **Key Learnings:**
  - Operating local LLMs on GPU hardware (Google Colab T4 runtime).
  - Tokenizer management, context length handling, and prompt formatting tailored for instruction-tuned open-weight models.
  - Trade-offs between API-backed models (Cohere) vs. self-hosted models (Mistral-7B) in terms of latency, privacy, and infrastructure cost.

---

### 3. `Langchain_LCEL.ipynb`

* **File Name:** `Langchain_LCEL.ipynb`
* **Description:**  
  This notebook explores **LangChain Expression Language (LCEL)** to build clean, flexible, and complex multi-step LLM pipelines. It implements a multi-model consensus workflow where two distinct LLM chains generate answers in parallel, and a third chain synthesizes their responses into a single output.
* **What Happens:**
  - Demonstrates LCEL composition using the pipe operator (`|`).
  - Uses `ChatPromptTemplate`, `StrOutputParser`, `RunnablePassthrough`, and `RunnableLambda`.
  - Constructs two independent chains (`chain1` and `chain2`) querying `ChatCohere`.
  - Leverages `RunnablePassthrough.assign()` to execute `chain1` and `chain2` concurrently and feed both outputs directly into a synthesis prompt (`chain3`).
  - Demonstrates how complex DAGs (Directed Acyclic Graphs) of execution can be cleanly declared in Python.
* **Key Learnings:**
  - **LCEL Syntax & Composability:** Building pipelines with `prompt | model | parser`.
  - **Data Passing:** Using `RunnablePassthrough()` and `.assign()` to inject dynamic state across multi-stage chains.
  - **Ensemble / Mixture-of-Agents Pattern:** Combining multi-agent/multi-model outputs into a refined, consolidated response.

---

## 💡 Key Concepts & Key Learnings

| Concept | Description |
| :--- | :--- |
| **LLMChain vs. LCEL** | Modern LangChain replaces `LLMChain` with LCEL (`|` piping), offering native streaming, async execution, and modular composability. |
| **Runnable Passthrough** | `RunnablePassthrough.assign()` enables seamless intermediate data manipulation without breaking chain flow. |
| **Model Ensembling** | Fetching diverse outputs from multiple model calls and synthesizing them improves response quality and accuracy. |
| **Local GPU Inference** | Running `Mistral-7B` locally gives complete data control without API rate limits or recurring costs. |

---

## ⚙️ Prerequisites & Installation

To run these notebooks locally or in Google Colab, install the required Python packages:

```bash
pip install langchain langchain-core langchain-community langchain-cohere langchain-classic cohere transformers torch
```

---

## 🚀 How to Run

1. **Get an API Key:**  
   Sign up at [Cohere](https://cohere.com/) and copy your API key.
2. **Set up Secrets in Colab:**  
   Add a secret named `COHERE_KEY` under the Secrets tab in Google Colab.
3. **Execute in Sequence:**  
   - Start with `Langchain_Setup_and_Simple_Chain.ipynb` to test basic connectivity [cite: 1].
   - Proceed to `Mistral_7B.ipynb` (ensure GPU accelerator is enabled) [cite: 3].
   - Run `Langchain_LCEL.ipynb` to experiment with multi-model chain orchestration [cite: 2].

---
*Created for hands-on learning and practical implementation of LLM architectures.*
