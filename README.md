# Lunar Ice Detection Pipeline
### Bharatiya Antariksh Hackathon 2026 · Problem Statement 8

**Multi-Sensor Subsurface Water Ice Detection, Volume Estimation, and Rover Path Planning at the Lunar South Pole**

**Target Region:** Sverdrup-Henson Complex (~89.5°S, 152°E) · **Pipeline Version:** v8.3 · **Team Institution:** Indian Institute of Science

---

## ⚠️ Implementation Status

This is a hackathon submission (BAH 2026). The methodology is fully designed and extensively documented. The Python pipeline is a **working prototype**: all core physics (CPR/DOP/m-chi gates, Maxwell-Garnett dielectric inversion, A\* rover path planning, Random Forest ensemble) are implemented and run end-to-end. Some advanced features (ShadowCam COG HTTP fetch, S-band dual-frequency validation against ground truth) require satellite data files not included in this repository. The scientific design and pipeline architecture are the primary contribution.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Why Sverdrup-Henson?](#2-why-sverdrup-henson)
3. [Data Sources — 12 Lines of Evidence](#3-data-sources--12-lines-of-evidence)
4. [Pipeline Architecture](#4-pipeline-architecture)
5. [Ice Detection Physics](#5-ice-detection-physics)
6. [AI/ML Design — Removing Radar-Feature Circularity](#6-aiml-design--removing-radar-feature-circularity)
7. [Ice Volume Estimation](#7-ice-volume-estimation)
8. [Rover Path Planning](#8-rover-path-planning)
9. [Innovation: In-Situ Confirmation Chain](#9-innovation-in-situ-confirmation-chain)
10. [Innovation: Extraction & ISRU](#10-innovation-extraction--isru)
11. [Known Limitations](#11-known-limitations)
12. [Future Work](#12-future-work)
13. [How to Run](#13-how-to-run)
14. [Output Files](#14-output-files)
15. [References](#15-references)

---

## 1. Project Overview

This pipeline provides a fully automated, multi-sensor, physics-constrained solution to detect subsurface water ice at the lunar south pole, estimate its volume, and plan a safe rover path to the highest-confidence target.

It integrates **five Chandrayaan-2 instruments** and **NASA LRO data** into a 4-tier confidence classification system, then applies a **Random Forest voting ensemble** trained on zero pixels from the target tile — eliminating the most common circularity failure mode in self-supervised ice-detection pipelines.

**Core outputs:**
- Per-pixel 4-tier ice confidence map at native DFSAR resolution
- Ice volume estimate with Monte Carlo uncertainty (27–50 tonnes ±30% 1σ for the Sverdrup test tile)
- `Rover_Waypoints.csv` — safe A\* path to the highest-confidence T4 pixel
- 18-panel mission dashboard PNG
- Trained `RandomForest_LunarIce_Model.pkl` + `model_metadata.json`

---

## 2. Why Sverdrup-Henson?

The problem statement referenced Faustini crater. Our pipeline targets **Sverdrup-Henson** for three reasons:

1. **Instrument coverage:** OHRC and IIRS are passive instruments requiring reflected sunlight. Faustini's interior is predominantly permanently shadowed, leaving both instruments blind. Sverdrup's rim and approach terrain have better illumination geometry.
2. **Published ice-favourability:** Leone et al. (2023, *iScience*) systematically compared south pole base sites and identified Sverdrup-Henson as having higher water-ice abundance than nearby de Gerlache and Shackleton craters, with a partially shadowed floor showing multiple independent ice signatures.
3. **Universal methodology:** The CPR/DOP diagnostic criteria (Sinha et al. 2026) are not site-specific. The pipeline retargets to Faustini/Haworth/Shoemaker by changing only `TARGET_LAT`, `TARGET_LON`, and the data folder path — no code changes.

---

## 3. Data Sources — 12 Lines of Evidence

| # | Instrument | Measurement | Ice Evidence |
|---|---|---|---|
| 1 | CY-2 DF-SAR FP (L-band, 2021) | CPR = (HH+VV+2HV)/(HH+VV−2HV) | CPR > 1.0 → volume scatter |
| 2 | CY-2 DF-SAR CP (L-band, 2025) | DOP = \|LH−LV\|/(LH+LV) | DOP < 0.13 → depolarisation |
| 3 | CY-2 DF-SAR CP (L-band, 2025) | m-chi ellipticity (Raney 2007) | chi < −0.10 rad → volume dominant |
| 4 | CY-2 IIRS (2023) | BDI at 2.0 µm | H₂O combination band — **sunlit only** |
| 5 | CY-2 IIRS (2023) | BDI at 3.0 µm | OH/H₂O fundamental — **sunlit only** |
| 6 | DF-SAR → dielectric bridge | Maxwell-Garnett mixing | Ice volumetric fraction |
| 7 | NASA DIVINER / thermal model | Surface temperature T | T < 110 K → thermally stable |
| 8 | CY-2 OHRC + TMC-2 | Roughness + shadow proxy | PSR identification |
| 9 | S-band SAR (auto-detected, optional) | Dual-frequency CPR consistency | Rock vs. ice discriminator |
| 10 | NASA LOLA DEM | Rim-floor d/D ratio + rim-blocking | Doubly shadowed crater (DSC) geometry |
| 11 | ShadowCam (KPLO/Danuri, NASA/KARI) | PSR-interior albedo at 200× LROC sensitivity | Bright anomalies via scattered earthshine |
| 12 | In-situ: LIBS + XRF + XRD | H-α 656 nm · O/Si anomaly · ice Ih d-spacing | Definitive post-landing confirmation |

> **IIRS scope note:** IIRS is a passive reflectance spectrometer — no signal inside PSRs. Lines 4–5 apply to sunlit terrain only. PSR-interior ice is confirmed by Lines 1–3, 7, and 10.

**Why demand both FP and CP radar agreement?** A rough rock may fool FP-CPR (volume-scatter ratio) but is much less likely to simultaneously fool CP-DOP (depolarisation ratio — a physically different polarimetric quantity) at the same pixel. Dual-mode consensus substantially reduces false positives.

---

## 4. Pipeline Architecture

```
[DATA INPUTS]
FP-SAR (L-band, 2021) | CP-SAR (L-band, 2025) | S-SAR (optional, auto-detected)
TMC-2 (terrain) | OHRC (morphology) | IIRS (hyperspectral)
LOLA DEM 5m (NASA LRO, auto-fetched) | DIVINER thermal | ShadowCam (KPLO, auto-fetched)
        |
        v
[STEP 1: RADAR PROCESSING]
  DN -> sigma0  (K = 10^(cal_dB/10), cal = 70.31 dB)
  Lee speckle filter  (FP: 7x7 ENL=20, CP: 5x5 ENL=4)
  FP-CPR = (HH+VV+2HV)/(HH+VV-2HV)   [GRD adaptation]
  CP-DOP = |LH-LV|/(LH+LV)            [incoherent GRI]
  CP m-chi decomposition               [Raney 2007]
  Auto-calibrate FP threshold to 90th percentile if saturated
        |
        v
[STEP 2-4: TERRAIN, PSR & DSC DETECTION]
  LOLA 5m DEM: slope, elevation (auto-fetched from NASA PDS)
  PSR = union(TMC-2 darkest 2%, OHRC darkest 2%)
  DSC: LOLA topo minimum + inside PSR + d/D > 0.05
  Rock mask: 98th-percentile roughness -> hard exclusion
        |
        v
[STEP 5: 4-TIER CONFIDENCE CLASSIFICATION]
  T4 CONFIRMED:  CPR + DOP(<0.35) + chi + smooth + T<110K + not rock
  T3 HIGH:       CPR + DOP(<0.13) + T<110K  (inside PSR)
  T2 PROBABLE:   CPR + DOP(<0.35)  (inside PSR)
  T1 CANDIDATE:  CPR > threshold only
        |
        v
[STEP 6: AI/ML  (Voting Ensemble - ANCHOR_ONLY_MODE)]
  Features: 9 topo/thermal only -- ZERO radar inputs
  Training: 300 ICE+ / 300 DRY- synthetic anchor pixels
    from LCROSS, Ch-1 M3, Shackleton MiniRF, Earth permafrost
    ZERO pixels from the Sverdrup tile used in training
  RF (200 trees) + GBM (100 iters) -> Soft Vote -> Isotonic calibration
  Decision threshold: P(ice) > 0.70
        |
        v
[STEP 7-9: VOLUME, EXCAVATION TARGETING & ROVER PATH]
  Maxwell-Garnett dielectric inversion + Monte Carlo 1000x
  DSC-ranked excavation target list
  A* path: Bekker-Wong cost + PSR/comms/cold-soak penalties
  Output: Rover_Waypoints.csv
```

**MIDAS Hybrid Mode:** The pipeline can ingest MIDAS-exported GeoTIFFs directly (ISRO-validated SAR calibration upstream, Python handles all downstream AI and path planning). Set `MIDAS_MODE = True` and place exports in `midas_exports/`. See [`Hackathon_Solution_README.md`](Hackathon_Solution_README.md) Section 4 for exact MIDAS workflow steps.

---

## 5. Ice Detection Physics

### Primary Radar Gate (Sinha et al. 2026)
- **CPR > 1.0** (auto-calibrated to 90th percentile at 89.5°S to correct grazing-angle saturation)
- **DOP < 0.13** (strict; Sinha 2026 validated threshold)
- **m-chi < −0.10 rad** (volume-scatter dominant; threshold relaxed at grazing incidence where LH/LV imbalance is systematic)

### Thermal Stability Gate
- PSR or DSC with modelled T < 110 K (Vasavada 1999)

### Dual-Frequency Rock Discriminator (when S-band available)
`SSAR_ICE = (CPR_S > 0.80) AND (CPR_S < CPR_L)`

Physics: L-band (λ ≈ 23 cm) penetrates ~5 m; S-band (λ ≈ 9–12.6 cm) ~1.5 m. A buried ice deposit shows `CPR_S < CPR_L` (L-band reaches the ice; S-band sees only the regolith column above it). Surface boulders do not show this depth-dependent drop. Rejected pixels are logged separately for auditability.

### Self-Validation (all passed on Sverdrup tile)

| Check | Result |
|---|---|
| Ice preferentially in PSR? | **PASS** — 100% of strict ice inside PSR |
| T4 ice in topographic lows? | **PASS** — T4 mean elevation < scene mean |
| RF P(ice) higher in PSR than sunlit? | **PASS** — 0.392 vs 0.307 |

---

## 6. AI/ML Design — Removing Radar-Feature Circularity

### The Naive (Wrong) Approach

```
Use CPR > 1.0 to label pixels as ice
        |
Use CPR as a training feature
        |
RF re-learns CPR threshold -- zero new information
```

### Our Solution: Two Strict Separations

**1. Features are 100% radar-free.** All 9 RF inputs come from LOLA, TMC-2, OHRC, IIRS, and ShadowCam — instruments physically independent of DFSAR.

**2. Labels come from confirmed external missions, not Sverdrup pixels.** Training data is synthetic, generated from Gaussian distributions parameterised by published summary statistics at independently confirmed sites:

| Source | Label | N | Key Physics | Reference |
|---|---|---|---|---|
| LCROSS / Cabeus | ICE+ | 120 | 5.6 wt% H₂O confirmed; slope ~2°, elev ~−3100 m | Colaprete et al. (2010) *Science* 330 |
| Chandrayaan-1 M³ S. Pole | ICE+ | 80 | OH/H₂O 2.8 µm absorption; PSR ~0.90, elev ~−2500 m | Pieters et al. (2009) *Science* 326 |
| Shackleton MiniRF + LPNS | ICE+ | 60 | CPR + neutron flux suppression; elev ~−4000 m, PSR ~0.99 | Spudis et al. (2010) *GRL* |
| Alaska Permafrost (Earth analog) | ICE+ | 40 | L-band SAR; borehole-confirmed ice lenses | Rignot (1994) *Science* 263 |
| Sunlit Highlands | DRY− | 150 | Zero OH (IIRS/Clementine); bright, warm, rough | Sunshine et al. (2009) |
| Steep Crater Walls > 25° | DRY− | 150 | Thermally unstable; T > 200 K in sunlight | Vasavada et al. (1999) *Icarus* |

**Total training set: 300 ICE+ / 300 DRY− = 600 synthetic samples. Zero from Sverdrup.**

> **Honest limitation:** The anchor distributions are team-authored from published summary statistics, not raw co-registered pixel data. ICE+ anchors were deliberately parameterised with low slope / high PSR / deep elevation; DRY− anchors with the opposite. A classifier trained on them will separate along those axes on the target tile — downstream checks ("RF prefers PSR pixels") confirm the prior was encoded correctly, not that the model independently discovered something new. This removes the specific CPR/DOP circularity failure mode and makes physical assumptions explicit and auditable via `model_metadata.json`.

### Ensemble Architecture

```
Input: 9 topo/thermal features (StandardScaler normalised)
        |
        +---> RandomForestClassifier (200 trees, RandomizedSearchCV)
        +---> HistGradientBoostingClassifier (100 iterations)
                  |
                  v
        SoftVotingClassifier (average probabilities)
                  |
                  v
        CalibratedClassifierCV -- isotonic regression
        [raw score -> calibrated true P(ice)]
                  |
                  v
        Post-prediction physics gates:
          Rock mask -> hard-zero roughest 2% terrain
          IIRS veto -> x0.30 penalty on sunlit no-OH pixels
                  |
                  v
          P(ice) > 0.70 -> ICE_ML pixel
```

### Feature Importances (Sverdrup run)

| Feature | Source | Importance |
|---|---|---|
| Slope_LOLA | LOLA DEM gradient (5 m/px) | **33.1%** |
| TMC_Roughness | TMC-2 brightness std | **28.6%** |
| PSR_score | TMC+OHRC shadow union | 12.5% |
| LOLA_elev | LOLA absolute elevation | 10.2% |
| Poleward | LOLA aspect angle | 6.6% |
| Incidence_Angle | FP satellite metadata | 4.1% |
| BDI_3000 | IIRS 3.0 µm OH band | 2.1% |
| BDI_2000 | IIRS 2.0 µm water band | 1.6% |
| SC_bright | ShadowCam COG / OHRC fallback | 1.2% |

> BDI features score near 0% inside PSRs by design — IIRS is a passive spectrometer with no signal in permanently shadowed terrain, and ice-positive pixels live inside PSRs.

**IIRS Discrimination Ratio:** RF P(ice) is **1.34–1.53× higher** at IIRS-confirmed water pixels than non-water pixels in sunlit terrain — the model generalises beyond its training domain without seeing Sverdrup data.

---

## 7. Ice Volume Estimation

Two physically distinct stages:

**Stage 1 — Backscatter to permittivity (`cpr_to_eps`):** Bridges measured CPR to effective bulk permittivity, anchored at dry regolith ε ≈ 2.7 (Carrier et al. 1991) and ice ε ≈ 3.15 (Mätzler 1996).

**Stage 2 — Maxwell-Garnett mixing inversion (`mg_fraction`):** Ice modelled as spherical inclusions in a dry-regolith host matrix. Ice fraction capped at 20% (Feldman et al. 2001 Neutron Spectrometer upper bound).

**Monte Carlo uncertainty:** 1,000 samples jointly varying ε_ice = 3.15 ± 0.1 and depth_L = 5.0 ± 1.5 m, propagated through both stages → honest ±30% 1σ mass uncertainty.

**Sverdrup result:** **27–50 tonnes** (top 5 m, 10–50% ice fraction).

---

## 8. Rover Path Planning

- **Safety gate:** True LOLA slope ≤ 15° (per-pixel geographic scaling; no resampling artifacts)
- **Energy model:** Bekker-Wong lunar soil mechanics for step cost
- **Penalties applied:** PSR darkness (solar charging loss), Earth LOS blackout (communications), cold-shock dwell (T < 40 K hardware freeze risk)
- **Battery model:** Full state-of-charge profile along traverse (Panel N in dashboard)
- **Target:** Highest P(ice) within best DSC cluster; falls back to global T3+ search if no DSC qualifies
- **Output:** `Rover_Waypoints.csv` — Step, Row, Col, Elevation (m), TMC slope (illumination proxy), LOLA slope (safety validation)

**Hazard Detection Case Study:** The Sverdrup test tile (89.5°S, 152°E) sits on the Sverdrup-Henson ridge wall, not the crater floor. The pipeline automatically computed a 30.6° average slope (2.7 km elevation drop across 4 km) and flagged **97.9% of path segments as unsafe** (> 15° limit), correctly recommending the landing target be shifted several km north to the actual crater floor.

---

## 9. Innovation: In-Situ Confirmation Chain

Before any irreversible excavation, the rover executes a five-instrument sequenced confirmation chain — fastest first, each gate saving time by rejecting false candidates early:

```
LIBS standoff (30 sec, 7 m) -- H-alpha at 656 nm detected? -->
  TNP thermal (5 min) -- k_eff > 0.1 W/m/K? -->
    ENS neutron (30 min) -- epithermal flux suppressed? -->
      GPR structural (10 min) -- ice layer imaged? -->
        XRF/XRD chemical/crystallographic (15 min)
          --> CONFIRMED: H2O ice, proceed to extraction
```

| Instrument | Physics | Time | Mass | Heritage |
|---|---|---|---|---|
| **LIBS** | H-α 656 nm + OH 306–310 nm; standoff 7 m | 30 sec | — | SuperCam / ChemCam |
| **Thermal Needle Probe (TNP)** | κ_ice / κ_dry = 730× contrast at 100 K | 5 min | ~150 g | HP³ (InSight) |
| **Epithermal Neutron Sounding (ENS)** | H thermalises fast neutrons; ≥ 0.5 wt% H₂O | 30 min | ~1 kg | LEND (LRO) |
| **Rover-mounted GPR** | Δε ≈ 3.1 at ice/regolith boundary; 1–10 m penetration | 10 min | ~1.5 kg | RIMFAX (Perseverance) |
| **Raman + XRF/XRD** | O-H stretch 3200 cm⁻¹; ice Ih d-spacings 3.90/3.67/3.44 Å | 15 min | ~2 kg | PIXL / CheMin / MicrOmega |

Five orthogonal physical principles on the same pixel — negligible joint false-positive probability. Full chain takes < 75 minutes.

---

## 10. Innovation: Extraction & ISRU

### Sublimation Tent Extraction

A 1 m², ~0.5 kg Mylar tent is deployed over a confirmed T4 pixel. A resistance heater (2–5 W) warms regolith from ~90 K to ~200 K; sublimated H₂O refreezes on a permanently shadowed cold-trap face.

**Yield estimate (Knudsen-Hertz):** ~200–400 g H₂O/day at 220 K, 10% ice fraction — no drilling, no contamination risk.

**RTG integration:** MMRTG waste heat (~2000 W thermal, ~110 W electrical) pipes directly into the tent, eliminating the resistance heater and enabling 24/7 continuous extraction inside a permanently shadowed PSR.

### Full ISRU Chain

```
H2O  ->  Electrolysis (PEM)  ->  H2 + O2
H2   ->  Sabatier (CO2 + 4H2 -> CH4 + 2H2O)  ->  CH4 rocket propellant
H2/O2 ->  Fuel Cell (reverse)  ->  Electricity + H2O recycled
```

**1 tonne of Sverdrup ice yields:**

| Product | Quantity | Application |
|---|---|---|
| O₂ | ~889 kg | Life support + oxidiser |
| CH₄ | ~446 kg | Ascent propellant (Chandrayaan-4) |
| H₂O (Sabatier by-product) | ~400 kg | Recycled back to tent |

---

## 11. Known Limitations

| Limitation | How Handled |
|---|---|
| **Grazing incidence (89.5°S):** CPR auto-inflated; DOP/chi systematically biased | Adaptive 90th-percentile CPR threshold; DOP relaxed to 0.35 for T4; annotated in dashboard Panel A |
| **IIRS blind inside PSRs** | IIRS applied only as post-prediction veto on sunlit terrain; near-zero feature importance in PSR pixels is expected and correct |
| **S-band data absent for Sverdrup tile** | Graceful L-band-only fallback; S-band gate activates automatically when files are placed in a matching subfolder |
| **ShadowCam COG not catalogued for Sverdrup** | OHRC fallback used; ShadowCam activates for catalogued craters (Faustini ×11, Shackleton ×2) |
| **ML labels are literature-parameterised priors** | Stated explicitly in `model_metadata.json`, code docstring, and README; presented as transfer learning prior, not independent discovery |
| **S-band absolute threshold (0.80) is engineering judgement** | Documented in code; requires validation against a site with independently known ice depth |
| **Sverdrup DSC is 30+ km away from test tile** | Pipeline correctly returned 0 DSCs; falls back to global T3+ PSR search for rover targeting |

---

## 12. Future Work

1. **Per-pixel incidence angle normalisation** — `CPR_norm = CPR / (1 + 0.012 × (θ − 25°))`; requires Level-1 SLC data (not in public Level-2 GRD)
2. **LOLA Rock Abundance Catalogue filter** (Bandfield et al. 2011) — direct rock detection replacing optical roughness proxy
3. **Multi-epoch PSR ray-tracing** — full 360-day solar illumination simulation over LOLA DEM (requires GPU)
4. **Faustini/Haworth/Shoemaker validation** — change only `TARGET_LAT/LON` once SAC/ISRO releases full data
5. **S-band-specific calibration constant** — when ISRO provides band-specific values for SP-mode tiles
6. **U-Net crater segmentation** — automated crater counting and PSR boundary delineation on OHRC imagery
7. **Energy-constrained sortie planner** — strict round-trip battery budget from sunlit base camp into PSR and back
8. **Onboard Rice/Golomb telemetry compression** — lossless engine enabling real-time sensor streaming over ~2 kbps DSN downlinks

---

## 13. How to Run

```bash
# Install dependencies
pip install numpy scipy matplotlib scikit-learn joblib rasterio

# Run the full pipeline
# Chandrayaan-2 data folders must be in the same directory
python Lunar_ice_detection_v8_AI.py
```

On first run without `lola_sverdrup.npy`, the script auto-downloads the LOLA 5 m DEM from NASA PDS (~110 MB).

**Expected data folder structure:**
```
Sverdrup Crater/
├── ch2_sar_ncxl_<timestamp>_d_fp_d18/data/calibrated/<date>/   <- FP SAR GeoTIFFs
├── ch2_sar_ncxl_<timestamp>_d_cp_d18/data/calibrated/<date>/   <- CP SAR GeoTIFFs
├── ch2_tmc_ncf_<timestamp>_d_img_d18/data/calibrated/<date>/   <- TMC-2 .img
├── ch2_ohr_ncp_<timestamp>_d_img_d18/data/calibrated/<date>/   <- OHRC .img
├── ch2_iir_nci_<timestamp>_d_img_d18/data/calibrated/<date>/   <- IIRS .qub + .xml
├── lola_sverdrup_5m.npy                                         <- auto-downloaded
└── Lunar_ice_detection_v8_AI.py
```

> Raw Chandrayaan-2 data: [ISSDC PRADAN portal](https://pradan.issdc.gov.in/) | LOLA DEM: [NASA PDS ODE](https://ode.rsl.wustl.edu/)

**To activate MIDAS mode:** Process DFSAR in MIDAS v4.2.4, export `midas_fp_cpr.tif` / `midas_cp_dop.tif` / `midas_cp_chi.tif` to `midas_exports/`, set `MIDAS_MODE = True` at line ~197.

**To activate S-band dual-frequency mode:** Place S-band SP-mode GeoTIFF pair (VV + VH) in any subfolder named `ch2_sar_ncxs_*_d_sp_d18/` — no code changes needed.

---

## 14. Output Files

| File | Description |
|---|---|
| `Sverdrup_v8_AI_Maps.png` | 18-panel mission dashboard (radar polarimetry, terrain, classification, path planning) |
| `Sverdrup_v8_AI_Summary.txt` | Machine-readable run summary: pixel counts, confidence tiers, volume estimate, path length |
| `Rover_Waypoints.csv` | A\* path waypoints: Step, Row, Col, Elevation, TMC slope, LOLA slope |
| `RandomForest_LunarIce_Model.pkl` | Trained voting ensemble (RF + GBM + isotonic calibration) |
| `StandardScaler_LunarIce.pkl` | Feature scaler (must be used with the model) |
| `model_metadata.json` | Training mode, anchor sources, feature list, circularity status, spatial CV split |
| `Sverdrup_3D_Interactive.html`| 3-D visualisation of the rover path|

---

## 15. References

- Sinha et al. (2026) *npj Space Exploration* 2, 22 — primary CPR/DOP criterion, DSC ice evidence
- Leone et al. (2023) *iScience* 26, 107853 — Sverdrup-Henson site-selection assessment
- Colaprete et al. (2010) *Science* 330, 463 — LCROSS 5.6% H₂O at Cabeus
- Pieters et al. (2009) *Science* 326 — Chandrayaan-1 M³ OH/H₂O detection
- Clark et al. (2009) *Science* 326 — Chandrayaan-1 M³ south pole water absorption
- Spudis et al. (2010) *GRL* — Shackleton MiniRF + LPNS CPR+neutron consensus
- Paige et al. (2010) *Science* 330, 479 — DIVINER thermal mapping
- Vasavada et al. (1999) *Icarus* 141 — lunar polar thermal stability model
- Raney (2007) *IEEE TGRS* 45 — m-chi compact polarimetry decomposition
- Carrier et al. (1991) *Lunar Sourcebook* — dry lunar regolith bulk permittivity
- Mätzler (1996) — water ice relative permittivity at cryogenic temperatures
- Sihvola (1999) *Electromagnetic Mixing Formulas* — Maxwell-Garnett theory
- Feldman et al. (2001) *JGR* 106 — Neutron Spectrometer ice abundance upper bound
- Nozette et al. (2001) — bistatic radar dual-frequency ice discrimination
- O'Brien & Byrne (2022) *GRL* — DSC thermal modelling (~25 K floors)
- Bandfield et al. (2011) *JGR* 116 — LOLA rock abundance from thermal inertia
- Rignot (1994) *Science* 263 — L-band SAR over borehole-confirmed permafrost
- Knudsen (1909) *Ann. Phys.* — Hertz-Knudsen sublimation flux formula
- Zubrin & Wagner (1996) *The Case for Mars* — Sabatier ISRU reaction chain
- Bekker (1960) *Off-the-Road Locomotion* — Bekker-Wong soil mechanics
- Hamran et al. (2020) *Space Sci Rev* — RIMFAX rover GPR (Perseverance)
- Wiens et al. (2021) *Space Sci Rev* — SuperCam Raman (Perseverance)
- Watson (1961) *JGR* — ice thermodynamic stability in permanently shadowed craters
- Campbell (2002) *JGR Planets* — L-band SAR penetration depth in regolith

---

## Additional Documentation

| Document | Contents |
|---|---|
| [`Hackathon_Solution_README.md`](Hackathon_Solution_README.md) | Full technical write-up: physics derivations, ML design rationale, MIDAS integration, S-band fusion logic, dashboard panel guide, ISRU chain details |
| [`METHODOLOGY_CARD.md`](METHODOLOGY_CARD.md) | Step-by-step pipeline flowchart, scientific defensibility table, dual-band methodology, gap-closing analysis |
| [`model_metadata.json`](model_metadata.json) | Machine-readable training metadata: anchor sources, feature list, circularity status, spatial CV split |

---

*Bharatiya Antariksh Hackathon 2026 · Problem Statement 8 · Indian Institute of Science*
