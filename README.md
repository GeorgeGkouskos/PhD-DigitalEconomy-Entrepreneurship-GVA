
# PhD Thesis – Digital Economy & Entrepreneurship and Gross Value Added per Activity

This repository contains the code, data structure, and supporting material for my PhD thesis:  
*The Impact of the Digital Economy on the Added Value per Economic Activity in OECD Countries: The Role of Entrepreneurship*  
School of Economics, Business and International Studies, University of Piraeus, Greece (2025).  
Author: **George D. Gkouskos**

---

## Research Logic
The thesis investigates how the **Digital Economy** – measured through international indexes such as DESI, DETF, DII etc. – affects the **Gross Value Added (GVA)** across economic activities in OECD countries, with a focus on the role of **Entrepreneurship**.  

Core claims:
- Digitalisation is a powerful but conditional engine—its payoffs differ by sector depending on affordability, adoption, capabilities, and governance.
- Entrepreneurship scales and propagates digital inputs into value creation.
- Interactions between DE and ENT are not additive; complementarities (or frictions) explain sectoral heterogeneity.

---

## Methodology
Three econometric models were developed:

1. **LASSOFE — Variable Discovery (Two-way Fixed Effects + Regularisation)**  
   - Two-way Fixed Effects; clustered/KCov SE / LASSO for high-dimensional selection; no leakage; VIF hygiene 
   - Identifies the key digital economy and entrepreneurship indicators driving GVA.  
   - Users can easily replace the dependent variable (e.g., DESI, DETF, GEM, GII, DII).  

2. **SPCAFE (Sparse Principal Component Analysis with Two-way Fixed Effects)**  
   - Reduces dimensionality of high-dimensional indicator datasets.  
   - Extracts the most informative latent factors for sectoral analysis and interactions of Digital Economy and Entrepreneurship.  

3. **SPCA LP - Dynamics (Factor-Augmented Local Projections)**  
   - Studies dynamic interactions between digital economy / entrepreneurship indicators and sectoral GVA.  
   - Produces **Impulse Response Functions (IRFs)** showing the response of economic sectors to digital shocks.  
Robustness: FDR-BH, DK bandwidth checks, within-VIF reporting, Mundlak/Hausman where relevant, CD and ADF residual diagnostics.
---
## Pilot Extension – Sectoral Digital Economy Index (SDEI)

As a novel extension of the econometric analysis, this thesis develops a **pilot composite index: the Sectoral Digital Economy Index (SDEI).**
**Objective**: Measure sectoral digital readiness/impact (pilot: Education  plus Finance & Insurance Sector).
**Approach**: Followed OECD Handbook on Composite Indicators.
**Core idea**: The SDEI is directly built on top of the regression results of the First Econometrivc Model above. Only those Digital Economy indicators that showed significant and positive effects on sectoral GVA are included.

**Novelty**: 
To the best of our knowledge, no prior study has constructed a sector-specific Digital Economy index explicitly anchored in regression-based evidence of impact on Gross Value Added. This approach ensures that the index is not only descriptive, but also causally linked to economic performance.

**Future Steps:**
Extend the methodology to the remaining 19 sectors based on regression results.
Build a comprehensive, sectoral mapping of the Digital Economy’s contribution to value added across the entire economy (remaining 19 GVA Sectors).

## Results (high-level insights)

The empirical analysis highlights differentiated sectoral responses to digitalization across the OECD economies:

**Digital Economy:** Strong gains where access/usage are scaled or embedded in operational cores (e.g., Admin & Support, Transport & Storage, Utilities, Arts/Entertainment, Education, Mining). Mixed but leaning positive signals in ICT, Professional/Scientific, Other Services. Weaker or negative where affordability, integration costs, thin margins, or governance frictions bind (e.g., Construction, Accommodation & Food, Households as Employers).

**Entrepreneurship:** Reliable multiplier—activation & knowledge assets convert opportunities into GVA; framework frictions attenuate magnitudes.

**Interactions:** Complementarities emerge when digital capability aligns with entrepreneurial activation and institutional depth.

**Dynamics (IRFs):** Entepreneurship shocks are expansionary across all sectors; Digital Economy shocks are context-dependent—positive where absorption capacity exists; short-run negatives may appear under affordability/integration pressure..

**Policy Takeaways**
- Pair connectivity & usage with capability formation and affordability relief.
- Treat sector heterogeneity seriously—sequence reforms to each sector’s friction/volatility.
- Sectoral indices and satellite accounts for the Digital Economy should track both enablers and barriers to support comparable, timely, and actionable statistics.
- Use SDEI and residual diagnostics to locate binding constraints and track progress.

These findings confirm the central hypothesis of the thesis: **the Digital Economy is a key driver of sectoral value added, with entrepreneurship acting as a crucial amplifier of its impact**. 

The evidence underscores the importance of aligning digital policies with entrepreneurship development in order to maximize productivity, innovation, and sustainable economic growth.
  

---

## Repository Structure
- `code/`
  - `LASSOFE.py` → General regression model  
  - `SPCAFE.py` → SPCA with Fixed Effects  
  - `SPCAFAVAR.py` → SPCA-based FAVAR estimation and IRF generation
  - `SDEI.py` → Sectoral Digital Economy Index Model Calculation
- `data/raw/` → Original dataset (**DatasetMasterCAP.xlsx**)  
- `data/processed/` → Cleaned datasets & model outputs  
- `references/` → Indicator tables (DESI, DETF, GEM, etc.)  
- `figures/` → IRFs, coefficient plots, SPCA loadings  
- `drafts/` → Working papers & thesis drafts  
- `requirements/` → Python Libaries Used  
---

## How to Use
1. Place raw data in `data/raw/` and indicator table in `references/`.  
2. Install requirements:  
   ```bash
   pip install -r requirements.txt
   ```  
3. Run scripts:  
   ```bash
   python code/LASSOFE.py
   python code/SPCAFE.py
   python code/SPCAFAVAR.py
   python code/SDEI.py
   ```  
4. Check results in `data/processed/` and figures in `figures/`.  

---

## Citation
If you use this repository, please cite as:

**Gkouskos, G. (2025). *PhD Thesis: The Impact of the Digital Economy on the Added Value per Economic Activity in OECD Countries: The Role of Entrepreneurship*. School of Economics, Business and International Studies, University of Piraeus, Greece.**
Available at: https://github.com/GeorgeGkouskos/PhD-DigitalEconomy-Entrepreneurship-GVA
