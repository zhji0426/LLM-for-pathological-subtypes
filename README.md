# IgAN Pathology Report Clustering Pipeline

This repository contains a pathology-report analysis pipeline for IgA nephropathy (IgAN), including:

- LLM-based extraction of structured pathology features from narrative biopsy reports
- Template-based cleaning and normalization of extracted JSON features
- Section-wise embedding and concatenation
- Unsupervised clustering and clustering stability analysis
- Post hoc interpretability analysis and integrated figure generation

The current repository is organized around the following workflow: pathology report text → structured JSON features → cleaned JSON → section embeddings → clustering/stability → interpretability.

## Repository structure

```text
.
├── deepseek_clustering_results/        # saved clustering outputs
├── demo_data/                          # demo input/output data (desensitized, limited)
├── embedding_cache/                    # cached embeddings for interpretability pipeline
├── import_feature_results_v2/          # figure/table outputs used for manuscript-style plotting
├── models/                             # local model-related files if needed
├── pathology_feature_deepseek/         # raw JSON features extracted by LLM
├── pathology_feature_deepseek_cleaned/ # cleaned/normalized JSON features
├── prompts/                            # prompt templates / JSON schemas
├── README.md
├── results/                            # clustering/stability results
├── rsc/                                # auxiliary resources
└── src/                                # source code
```

## Main scripts

### 1) `01_getFeature_integrated.py`
Extracts structured pathology features from biopsy reports using an LLM API.  
By default, this script is configured for **DeepSeek** and reads an Excel file of pathology reports, then saves one JSON file per report. The script requires the environment variable `DEEPSEEK_API_KEY`. 

Typical output directory:
- `pathology_feature_deepseek/`

### 2) `02_clean_pathology_feature.py`
Maps raw extracted JSON results into a template-defined normalized structure and removes specific unsupported keys (for example, `complement_membrane_attack_complex_C5b-9`). Cleaned files are saved to the corresponding `*_cleaned` directory. 

Typical output directory:
- `pathology_feature_deepseek_cleaned/`

### 3) `embed_ollama_03.py`
Reads cleaned JSON files, flattens four pathology sections, generates section-wise embeddings with **Ollama**, and concatenates them into a final embedding matrix. The default embedding backbone is `qwen3-embedding:latest`, with four sections (`glomerular_lesions`, `tubulointerstitial_lesions`, `vascular_lesions`, `immunofluorescence`). 

Typical outputs:
- flattened CSV
- embedding CSV with `emb_*` columns

### 4) `04_robust_clustering_evaluator.py`
Runs high-dimensional clustering with dimensionality reduction and compares multiple clustering algorithms. The pipeline can save:
- best labels
- simplified clustering tables
- clustering visualizations
- algorithm comparison summaries
- PDF reports 

### 5) `07_stable_classification_analysis.py`
Performs clustering robustness analyses, including:
- method comparison
- subsampling consensus stability
- perturbation stability
- cross-algorithm agreement

The current example uses **two-step PCA (2500 → 300)** and a **2-cluster** setting. 

### 6) `05_interpretability_pipeline.py`
Runs interpretability analyses after clustering, including:
- section-level attribution
- teacher–student SHAP analysis
- counterfactual delete/swap experiments
- mapping SHAP features back to JSON keys
- optional association with binary, continuous, and survival outcomes 

### 7) `06_ncomms_integrated.py`
Builds an integrated multi-panel figure from SHAP and counterfactual outputs, intended for manuscript-quality visualization. It exports PDF, SVG, PNG, and summary CSV files.

## Recommended workflow

### Step 1. Prepare input pathology reports
Prepare your pathology report table (for example, an `.xlsx` file) containing the biopsy report text and identifiers.

### Step 2. Extract structured JSON features with DeepSeek
Run `01_getFeature_integrated.py` to convert narrative pathology reports into structured JSON files.

Important:
- You must provide your own **DeepSeek API key**
- Set it in your environment before running

Example:

```bash
export DEEPSEEK_API_KEY=your_key_here
```

On Windows PowerShell:

```powershell
$env:DEEPSEEK_API_KEY="your_key_here"
```

### Step 3. Clean and normalize extracted JSON
Run `02_clean_pathology_feature.py` to standardize the extracted JSON structure.

### Step 4. Generate embeddings locally with Ollama
Before running `embed_ollama_03.py`, make sure you have:
- Ollama installed
- a local embedding model available, such as `qwen3-embedding:latest`

Example:

```bash
ollama pull qwen3-embedding:latest
```

Then run the embedding script to generate:
- flattened pathology feature CSV
- concatenated embedding CSV

### Step 5. Perform clustering
Use `04_robust_clustering_evaluator.py` to run clustering and export the main clustering outputs.

### Step 6. Evaluate clustering stability
Use `07_stable_classification_analysis.py` to assess robustness through resampling, perturbation, and algorithm-agreement analyses.

### Step 7. Run interpretability analyses
Use `05_interpretability_pipeline.py` to generate SHAP rankings, counterfactual analyses, and mapping tables.

### Step 8. Build integrated figure
Use `06_ncomms_integrated.py` to assemble a manuscript-style summary figure from interpretability outputs.

## Environment

This project uses Python and depends on packages imported in the current scripts, including:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `tqdm`
- `requests`
- `tenacity`
- `statsmodels`
- `lifelines`
- `shap`
- `ollama`

Some clustering options are optional and may additionally require:
- `hdbscan`
- `umap-learn`
- `kneed`

A minimal installation example:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn tqdm requests tenacity statsmodels lifelines shap ollama
```

Optional:

```bash
pip install hdbscan umap-learn kneed
```

## Notes on demo data

The files under `demo_data/` are **desensitized / partially hidden demo data** intended only to illustrate the pipeline structure and file format.

Please note:

1. The demo data are **not the full study dataset**
2. Some information has been **masked or removed** for privacy and sharing purposes
3. The demo sample size is **very small**
4. Because of the limited sample size, the demo data **cannot be expected to produce reliable or meaningful clustering results**
5. In particular, the demo data are **not suitable for reproducing the formal clustering structure or downstream biological/clinical conclusions**

In other words, the demo data are provided for:
- code structure inspection
- input/output format illustration
- pipeline smoke testing

They are **not** provided for reproducing the full study results.

## Important limitations

- This repository does **not** include the full original study dataset
- Running the demo data may complete the pipeline, but the resulting clusters should **not** be interpreted as valid biological or clinical subtypes
- `01_getFeature_integrated.py` requires a valid **DeepSeek API key**
- `embed_ollama_03.py` requires a locally available embedding model through **Ollama**
- Some scripts contain project-specific default paths and may need to be adjusted for a new environment

## Reproducibility note

To reproduce the full analysis meaningfully, users should prepare:

- a sufficiently large pathology report cohort
- consistent report formatting
- access to an LLM extraction backend
- a compatible local embedding model
- downstream clinical labels/outcomes if interpretability and association analyses are required

## Citation / usage note

If you use this codebase or adapt parts of the pipeline, please cite the corresponding study and clearly indicate any modifications made to the original workflow.

## Contact

For questions about data availability, reproducibility, or code adaptation, please contact the repository maintainer.
