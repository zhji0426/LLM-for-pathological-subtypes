LLM for Pathological Subtypes in IgA Nephropathy

This repository provides the code and pipeline for the study:



Large language models decode narrative pathology reports to define clinically actionable subtypes in IgA nephropathy

Ji Zhang, Jiadan Lu, Liya Jiang, et al.



Overview

Accurate pathological risk stratification and rational selection of immunosuppressive therapy remain major challenges in immunoglobulin A nephropathy (IgAN). Existing systems, such as the Oxford MEST-C classification, rely on expert-defined categorical features and do not fully utilize the rich prognostic information in narrative kidney biopsy reports.



This project introduces a data-driven pathological subtype classification framework that applies a large language model (LLM) to systematically encode and integrate full-text renal pathology reports. Using biopsy reports from 2,789 adults with primary IgAN, we generated high-dimensional semantic representations and performed unsupervised clustering to identify two robust pathological subtypes: low-activity and high-activity.



The high-activity subtype is associated with more severe clinicopathological features and a significantly higher risk of kidney disease progression (adjusted HR = 6.99). Importantly, we identified a significant interaction between pathological subtype and corticosteroid therapy: corticosteroid treatment substantially reduced progression risk only in the high-activity subtype (adjusted HR = 0.53), with no benefit in the low-activity subtype.



Features

LLM-based feature extraction: Uses DeepSeek and Qwen3-Embedding to transform narrative pathology reports into high-dimensional semantic embeddings.



Unsupervised clustering: Identifies latent pathological phenotypes without predefined labels.



Subtype classification: Implements ensemble models (LightGBM, XGBoost, etc.) for robust subtype assignment.



Clinical validation: Evaluates subtype associations with kidney outcomes and treatment responses.



Web tool available: A user-friendly interface for subtype prediction is hosted at www.igan123.cn.

