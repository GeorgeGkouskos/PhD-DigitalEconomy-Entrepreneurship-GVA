
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

**-Primary sectors (Agriculture, Forestry & Fishing; Mining & Quarrying):** 
Digitalisation effects are modest but emerging. Gains are primarily associated with incremental efficiency improvements, better resource allocation, and gradual adoption of digital practices.

**-Secondary sectors (Manufacturing, Construction, Energy, Water & Waste Management):** 
Digital transformation exerts a significant and measurable impact, mainly through productivity acceleration, process automation, and innovation-driven growth dynamics. These sectors show the clearest link between digital investment and tangible output gains.

**-Tertiary and knowledge-intensive sectors (ICT, Finance, Professional, Scientific & Technical Services):** 
The most pronounced and sustained benefits are recorded here. Their high absorptive capacity for digital technologies, combined with their role as platform enablers for the wider economy, amplifies spillover effects across sectors and countries.

**-Entrepreneurship:** 
Functions as both a transmission mechanism and an amplifier of digitalisation’s impact. Entrepreneurial activity enhances the scaling of digital innovations, with disproportionate effects in service-oriented and innovation-intensive economies.

**-Dynamic responses (Impulse Response Functions):** 
IRF analysis confirms that digital shocks yield persistent, long-run positive effects on GVA. The impacts are strongest and most resilient in services, ICT, and finance, underscoring the structural importance of digitalisation in these sectors.

**-Sectoral Digital Economy Index (SDEI):** 
The pilot SDEI for the Education sector demonstrates the feasibility and robustness of sector-specific digital economy measurement. Results are consistent across OECD countries and validate comparability within the EU sample. This provides a methodological foundation for scaling the index to other industries.

**Key Takeaways**
-Digitalisation is a systematic driver of sectoral value added across OECD economies.
-Entrepreneurship acts as a crucial amplifier, deepening and broadening the gains from digitalisation.
-Policy strategies should integrate digitalisation and entrepreneurship development to maximise productivity, foster innovation, and sustain long-term economic growth.

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
Available at: [https://github.com/georgiosgkouskos-creator/PhD-DigitalEconomy-Entrepreneurship-GVA](https://github.com/GeorgeGkouskos/PhD-DigitalEconomy-Entrepreneurship-GVA)
