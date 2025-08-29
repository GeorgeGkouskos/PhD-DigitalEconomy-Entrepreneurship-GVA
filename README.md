
# PhD – Digital Economy & Entrepreneurship

This repository contains the Python code, data structure, and outputs for my PhD thesis:  
*The Impact of the Digital Economy on the Added Value per Economic Activity in OECD Countries: The Role of Entrepreneurship.*

---

## Repository Structure
- `code/`
  - `LASSOFE.py` → General regression model (LASSO Fixed Effects).  
    The user can replace the dependent variable each time (e.g. instead of **DESI**, use **DETF**, **GEM**, **GII**, **DII**, etc.).
  - `SPCAFE.py` → SPCA with Fixed Effects
  - `SPCAFAVAR.py` → SPCA-based FAVAR estimation and IRF generation
- `data/raw/` → Original dataset (**DatasetMasterCAP.xlsx**).  
  Columns (first 7): `Year, CountryShort, Country, GVA_Total, GVA_Agric_Total, GVA_Mining_Total, GVA_Manuf_Total, …`
- `data/processed/` → Cleaned panel data, intermediate and final results
- `references/` → Indicator tables, bibliography files
- `figures/irf/` → Impulse Response Function plots
- `figures/coefficients/` → Coefficient bar charts
- `drafts/` → Thesis drafts, working papers

---

## How to Use
1. Place your dataset:
   - `data/raw/DatasetMasterCAP.xlsx`
   - `references/Indicator_Table.xlsx`
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the scripts:
   ```bash
   python code/LASSOFE.py
   python code/SPCAFE.py
   python code/SPCAFAVAR.py
   ```
4. Outputs:
   - Processed datasets → `data/processed/`
   - Figures (IRFs, coefficient charts) → `figures/`

---

## Notes
- Large raw data files are **not uploaded** to GitHub (see `.gitignore`).
- All scripts assume panel structure: `Year + CountryShort` as keys.
- Figures follow my default styles:
  - IRFs → legend on right, thicker base factors, dashed interactions
  - Coefficients → grayscale academic bars with significance stars

---

## Citation
If you use this repository, please cite as:

George (2025). *PhD Thesis Materials – Digital Economy & Entrepreneurship*. University of Piraeus.
