
# PhD – Digital Economy & Entrepreneurship

This repository contains the code, data structure, and supporting material for my PhD thesis:  
*The Impact of the Digital Economy on the Added Value per Economic Activity in OECD Countries: The Role of Entrepreneurship*  
School of Economics, Business and International Studies, University of Piraeus (2025).  
Author: **George D. Gkouskos**

---

## Research Logic
The thesis investigates how the **Digital Economy** – measured through international indexes such as DESI, DETF, GEM, GII, DII – affects the **Gross Value Added (GVA)** across economic activities in OECD countries, with a focus on the role of **entrepreneurship**.  

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
   - Extracts the most informative latent factors for sectoral analysis.  

3. **SPCAFAVAR (FAVAR model with SPCA factors)**  
   - Studies dynamic interactions between digital economy indicators and sectoral GVA.  
   - Produces **Impulse Response Functions (IRFs)** showing the response of economic sectors to digital shocks.  

---

## Results (high-level insights)
- **Primary sectors (Agriculture, Mining):** digitalization has limited but emerging impact.  
- **Secondary sectors (Manufacturing, Construction, Energy):** stronger effect via productivity and innovation.  
- **Tertiary & knowledge-intensive sectors (ICT, Finance, Professional Services):** show the **largest gains from digital economy adoption**.  
- **Entrepreneurship** magnifies these effects, especially in innovation-driven economies.  
- IRF analysis reveals that digital shocks have **persistent positive effects** on sectoral GVA, with strongest long-term impact in services and ICT.  

---

## Repository Structure
- `code/`
  - `LASSOFE.py` → General regression model  
  - `SPCAFE.py` → SPCA with Fixed Effects  
  - `SPCAFAVAR.py` → SPCA-based FAVAR estimation and IRF generation
- `data/raw/` → Original dataset (**DatasetMasterCAP.xlsx**)  
- `data/processed/` → Cleaned datasets & model outputs  
- `references/` → Indicator tables (DESI, DETF, GEM, etc.)  
- `figures/` → IRFs, coefficient plots, SPCA loadings  
- `drafts/` → Working papers & thesis drafts  

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
   ```  
4. Check results in `data/processed/` and figures in `figures/`.  

---

## Citation
If you use this repository, please cite as:

**Gkouskos, G. (2025). *PhD Thesis: The Impact of the Digital Economy on the Added Value per Economic Activity in OECD Countries: The Role of Entrepreneurship*. School of Economics, Business and International Studies, University of Piraeus.**
