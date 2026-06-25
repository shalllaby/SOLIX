# RFC-002: Hybrid Profiling Engine Upgrade Plan

Act as: Principal Staff AI Architect, Solix Data Intelligence Platform
Date: June 2026
Status: DRAFT (Pending Approval)

---

## 1. Executive Summary & Mandate

We are overhauling the AutoML profiling and feature selection pipeline in the Solix platform. The legacy "LLM-Only" triage system is prone to high token consumption, latency overheads, and failure to handle basic numerical structures. 

This RFC establishes a **Hybrid Profiling & Selection Architecture (Math-First, LLM-Second)**. Mathematical heuristics will handle deterministic data profiling, target candidate identification, and basic feature dropping. The LLM's role will be focused strictly on semantic analysis (domain context and future data leakage detection) to optimize performance, token cost, and accuracy.

---

## 2. Sub-System Architecture & Mathematical Formulations

### A. Mathematical Target Ranking
**Location**: `core/automl/analyzer.py` and `Auto ML/core/analyzer.py`
**Heuristic**: `rank_target_candidates`

We will calculate the target candidate score $S(c)$ for each column $c$ using the following heuristic scoring matrix:

$$\text{Base Score} = 0.0$$

#### Penalties:
*   **Unique Primary Key (PK) Penalty**: If all values are unique ($\text{unique}(c) = N$), apply $-10.0$.
*   **Constant / Fully Null Penalty**: If the column is constant ($\text{unique}(c) \le 1$) or fully null ($\text{null}(c) = N$), apply $-20.0$.
*   **ID-like Name Penalty**: If the column name matches ID patterns (e.g., matching regex `r'(?i)(id|uuid|key|code|index|hash|guid|idx|serial)'`), apply $-5.0$.
*   **First Column Penalty**: Often an ID, apply $-2.0$ if $idx(c) = 0$.
*   **Missing Values Penalty**: Deduct up to $-10.0$ proportional to missingness:
    $$\text{Penalty} = - \frac{\text{null}(c)}{N} \cdot 10.0$$

#### Boosts:
*   **Semantic Target Keywords Boost**: If the column name matches target keywords (e.g., `price`, `fraud`, `class`, `target`, `label`, `churn`, `default`), apply $+8.0$.
*   **Last Column Boost**: Targets are traditionally at the end of the table, apply $+4.0$ if $idx(c) = M-1$.
*   **Binary Categorical Boost**: Perfect for binary classification, apply $+5.0$ if $c \in \text{Categorical}$ and $\text{unique}(c) = 2$.
*   **Multiclass Categorical Boost**: For multiclass classification, apply $+4.0$ if $c \in \text{Categorical}$ and $2 < \text{unique}(c) \le 10$.
*   **Continuous Numeric Boost**: For regression targets, apply $+3.0$ if $c \in \text{Numerical}$ and $\text{unique}(c) > 10$ (excluding ID-like names).

---

### B. Mathematical Feature Dropping
**Location**: `core/automl/preprocessor.py` and `Auto ML/core/preprocessor.py`
**Pipeline Stage**: Pre-training feature filtering inside `prepare_data`

We will enforce strict thresholds to clean features before training:
1.  **Zero-Variance / Near-Zero Variance (NZV)**:
    *   Drop features where the frequency of the most common non-null value is $> 99.5\%$.
    *   *Formula*: Let $v_1$ be the count of the most frequent value in column $c$. Drop if:
        $$\frac{v_1}{N - \text{null}(c)} > 0.995$$
2.  **High-Cardinality IDs**:
    *   Let the **Uniqueness Ratio** be $U(c) = \frac{\text{unique}(c)}{N - \text{null}(c)}$.
    *   Drop categorical features if $U(c) > 0.90$.
    *   Drop any feature matching ID keywords (e.g. `id`, `key`, etc.) if $U(c) > 0.50$.

---

### C. The "Hidden Numerical" Rescue
**Location**: `core/automl/preprocessor.py` and `Auto ML/core/preprocessor.py`
**Pipeline Stage**: `coerce_hidden_numericals`

To prevent text-encoding explosions caused by numbers corrupted by whitespaces or blank spaces (e.g. `"TotalCharges"` in the Telco Churn dataset):
1.  **Whitespace Strip**: Trim leading/trailing whitespace from string columns.
2.  **Conversion Attempt**: Cast to numeric using `pd.to_numeric(errors='coerce')`, producing `NaN` for non-coercible values.
3.  **Success Rate Evaluation**:
    *   Let $N_{\text{non-empty}}$ be the number of non-null, non-empty (after stripping) values in the original column.
    *   Let $N_{\text{numeric}}$ be the number of successfully converted numeric values.
    *   If $N_{\text{non-empty}} > 0$ and the conversion success rate is $\ge 70\%$, accept the cast:
        $$\text{Success Rate} = \frac{N_{\text{numeric}}}{N_{\text{non-empty}}} \ge 0.70$$
4.  **Re-routing**: Dynamically remove the column from `col_types["categorical"]` and add it to `col_types["numerical"]`. Downstream pipelines will then automatically apply median imputation and standard scaling instead of target encoding.

---

### D. The LLM Semantic Guard
**Location**: `core/automl/llm_profiler.py` and `Auto ML/core/llm_profiler.py`
**Refactoring Role**: Focus ONLY on **Semantic Target Confirmation** and **Future Data Leakage** (post-event indicators) that math cannot identify.

*   **Prompt Refactoring**:
    *   Instruct the LLM that deterministic checks (constant columns, unique primary keys, simple ID patterns) are already handled by the preprocessor.
    *   Focus the LLM on understanding column semantics to confirm if the suggested target is logical.
    *   Define **Future Data Leakage (Post-Event Indicators)**: columns populated only *after* the target event occurred (e.g., `customer_satisfaction_score` for predicting `churn`).
    *   Keep the response JSON structure intact (`OUTPUT_JSON_SCHEMA`) to maintain compatibility with API and UI serialization layers.

---

### E. Perfect Score Penalty
**Location**: `core/automl/engine.py`, `core/automl/kaggle_client.py` and `Auto ML` equivalents
**Composite Scorer**: `compute_balanced_composite`

To prevent models from winning due to slipped target leakage:
*   If a model achieves a validation score $s_{\text{val}} = 1.0$ but the cross-validation mean is lower ($cv_{\text{mean}} < 1.0$), we apply a strict target leakage/overfit penalty of $0.20$ to the final composite score.
*   *Formula Update*:
    $$\text{perfect\_score\_penalty} = \begin{cases} 0.20 & \text{if } s_{\text{val}} == 1.0 \land cv_{\text{mean}} < 1.0 \\ 0.0 & \text{otherwise} \end{cases}$$

---

## 3. Step-by-Step File Modification Plan

### Phase 1: Mathematical Target Ranking
1.  **Modify** `core/automl/analyzer.py` and `Auto ML/core/analyzer.py`:
    *   Update `rank_target_candidates` to align with the revised penalties and boosts.
    *   Ensure bilingual English/Arabic reasons match the scoring rules.

### Phase 2: Feature Dropping & Hidden Numerical Rescue
2.  **Modify** `core/automl/preprocessor.py` and `Auto ML/core/preprocessor.py`:
    *   Update `coerce_hidden_numericals` to compute the success rate over non-empty, non-null values.
    *   Update `prepare_data` zero-variance/near-zero variance filters using most-frequent value ratio.
    *   Refine high-cardinality ID dropping using the uniqueness ratio formula.

### Phase 3: LLM Semantic Guard
3.  **Modify** `core/automl/llm_profiler.py` and `Auto ML/core/llm_profiler.py`:
    *   Rewrite `SYSTEM_PROMPT` to restrict analysis scope to semantic target verification and post-event data leakage.
    *   Keep `OUTPUT_JSON_SCHEMA` unmodified.

### Phase 4: Perfect Score Penalty
4.  **Modify** `core/automl/engine.py` and `Auto ML/core/engine.py`:
    *   Update `compute_balanced_composite` to penalize perfect validation scores ($s_{\text{val}} = 1.0$) by $0.20$ if $cv_{\text{mean}} < 1.0$.
5.  **Modify** `core/automl/kaggle_client.py` and `Auto ML/core/kaggle_client.py`:
    *   Update `compute_balanced_composite` inside the generated remote Kaggle Python training script.

---

## 4. Verification Plan

### Automated Tests
1.  Run the local test suite `python "Auto ML/run_tests.py"` to verify dataset profiling, preprocessing, training pipeline, and output serialization.
2.  Run the API router client test suite `pytest tests/test_automl_router.py` to ensure complete end-to-end integration and FastAPI routes integrity.
3.  Write temporary unit tests in `<appDataDir>\brain\<conversation-id>/scratch/` to test edge cases:
    *   Target leakage model penalization ($s_{\text{val}} = 1.0$, $cv_{\text{mean}} = 0.94$).
    *   Hidden numerical column coercion (whitespace-corrupted numeric column).
    *   ID-dropping heuristics with varying uniqueness ratios.

---

## 5. Explicit Approval Request

Please review this RFC. If you are satisfied with this plan, **please reply with your explicit approval to begin the execution phase.**
