
# PhD Thesis – Digital Economy & Entrepreneurship and Gross Value Added per Activity

This repository contains the code, data structure, and supporting material for my PhD thesis:  
*The Impact of the Digital Economy on the Added Value per Economic Activity in OECD Countries: The Role of Entrepreneurship*  
School of Economics, Business and International Studies, University of Piraeus, Greece (2025).  
Author: **George D. Gkouskos**

---

## Research Logic
The thesis investigates how the **Digital Economy** – measured through international indexes such as DESI, DETF, DII etc. – affects the **Gross Value Added (GVA)** across economic activities in OECD countries, with a focus on the role of **Entrepreneurship**.  

The main idea:
- Digital transformation acts as a driver of productivity, efficiency, and innovation.
- Entrepreneurship is a key transmission channel, amplifying the impact of digitalization.
- The combined effect explains sectoral variation in GVA across primary, secondary, tertiary, and knowledge-intensive sectors.

---

## Methodology
Three econometric models were developed:

1. **LASSOFE (General Regression – Fixed Effects with LASSO Regularization)**  
   - Identifies the key digital economy and entrepreneurship indicators driving GVA.  
   - Users can easily replace the dependent variable (e.g., DESI, DETF, GEM, GII, DII).  

2. **SPCAFE (Sparse Principal Component Analysis with Fixed Effects)**  
   - Reduces dimensionality of high-dimensional indicator datasets.  
   - Extracts the most informative latent factors for sectoral analysis and interactions of Digital Economy and Entrepreneurship.  

3. **SPCAFAVAR (FAVAR model with SPCA factors)**  
   - Studies dynamic interactions between digital economy / entrepreneurship indicators and sectoral GVA.  
   - Produces **Impulse Response Functions (IRFs)** showing the response of economic sectors to digital shocks.  

---
## Pilot Extension – Sectoral Digital Economy Index (SDEI)

As a novel extension of the econometric analysis, this thesis develops a **pilot composite index: the Sectoral Digital Economy Index (SDEI).**
**Objective**: Measure the impact of the digital economy at the sectoral level (starting with Education Sector).
**Approach**: Followed OECD Handbook on Composite Indicators.
**Core idea**: The SDEI is directly built on top of the regression results of the First Econometrivc Model above. Only those Digital Economy indicators that showed significant and positive effects on sectoral GVA are included.
**Design**: Indicators are grouped into four conceptual pillars (Connectivity & Access, Usage & Online Interaction, Skills & Human Capital, Innovation & Knowledge Base)
Applied Min–Max normalization and equal weighting (pilot version).

**Outputs**:
The first application targets the Education sector, producing SDEI–Education scores for OECD countries (2010–2022).
An EU-only slice is highlighted, given stronger data availability across European countries.

**Novelty**: 
To the best of our knowledge, no prior study has constructed a sector-specific digital economy index explicitly anchored in regression-based evidence of impact on Gross Value Added. This approach ensures that the index is not only descriptive, but also causally linked to economic performance.

**Future Steps:**
Extend the methodology to the remaining 19 sectors based on regression results.
Build a comprehensive, sectoral mapping of the Digital Economy’s contribution to value added across the entire economy (remaining 19 GVA Sectors).

## Results (high-level insights)

The empirical analysis highlights differentiated sectoral responses to digitalization across the OECD economies:

**Digital Economy:** a powerful but context-dependent driver. Strong and robust gains in Administrative and Support Services, Transportation and Storage, Electricity, Gas and Steam, Arts, Entertainment and Recreation, Education, and Mining and Quarrying. Mixed but mostly positive signals in ICT, Professional, Scientific and Technical, and Other Service Activities. Mixed or negative associations where affordability, integration costs, thin margins, or governance frictions bind, including Construction, Accommodation and Food Services, Households as Employers, and parts of Finance and Insurance.

**Entrepreneurship:** a stable multiplier. Activation and knowledge assets reliably translate opportunities into value added. Framework frictions mainly scale magnitudes down rather than flipping signs.

**Interactions:** Digital and entrepreneurial drivers are not additive. Synergies arise when digital institutions or connectivity align with entrepreneurial activation and knowledge. Balanced or friction-dominated patterns appear where access and affordability constraints or weak institutional depth meet administrative and financing bottlenecks.

**Dynamics (IRFs):** Entrepreneurship shocks are expansionary in every sector. Digital shocks are context-dependent. Connectivity and adoption produce positive responses where capabilities and institutions can absorb them. Affordability and integration pressures can generate short-run negatives even in sectors with long-run potential. At higher maturity, incremental producer-side gains diminish and a larger share of benefits appears as consumer welfare.

**Key Takeaways**
- The Digital Economy is not a universal booster but a conditional engine whose payoffs vary by sector.
- Entrepreneurship provides the reliable transmission channel through which digital capabilities become value added.
- Effective policy requires sector-differentiated sequencing that pairs connectivity and adoption with entrepreneurial capacity and knowledge while relieving affordability and framework frictions.
- Sectoral indices and satellite accounts for the Digital Economy should track both enablers and barriers to support comparable, timely, and actionable statistics.

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
