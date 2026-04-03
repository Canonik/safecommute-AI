# Dataset Audit Report

Generated: 2026-04-03 07:46


## TRAIN Split

### Class Distribution by Source

| Source | Safe | Unsafe | Total | Balance |
|--------|------|--------|-------|---------|
| bg | 2125 | 0 | 2125 | 100/0 |
| cremad | 2523 | 1778 | 4301 | 59/41 |
| esc | 566 | 0 | 566 | 100/0 |
| hns | 2385 | 0 | 2385 | 100/0 |
| rav | 332 | 262 | 594 | 56/44 |
| savee | 177 | 85 | 262 | 68/32 |
| tess | 541 | 556 | 1097 | 49/51 |
| viol | 764 | 643 | 1407 | 54/46 |
| yt | 2452 | 848 | 3300 | 74/26 |
| **TOTAL** | **11865** | **4172** | **16037** | **74/26** |

### Quality Issues (17 flagged)

| Issue | Count | Sources |
|-------|-------|---------|
| uniform_rows | 17 | yt(defaultdict(<class 'int'>, {'flat_spectrum': 10, 'uniform_rows': 17})) |
| flat_spectrum | 10 | yt(defaultdict(<class 'int'>, {'flat_spectrum': 10, 'uniform_rows': 17})) |

#### Detailed Issues (top 50)

- **yt_metro_BVhJAWkuF7c_c007.pt** (safe): flat_spectrum, uniform_rows [energy=18.8114, silence=0.00]
- **yt_metro_a1_NIp6ohCtQeQ_c003.pt** (safe): uniform_rows [energy=17.9999, silence=0.00]
- **yt_metro_b2_aluo9cFGs3Q_c023.pt** (safe): flat_spectrum, uniform_rows [energy=18.8114, silence=0.00]
- **yt_metro_px_iGq0bHu7QxU_c002.pt** (safe): uniform_rows [energy=16.3768, silence=0.00]
- **yt_metro_yT0CYUCsRWE_c001.pt** (safe): flat_spectrum, uniform_rows [energy=18.8114, silence=0.00]
- **yt_scream_-s04sbIViJQ_c007.pt** (unsafe): uniform_rows [energy=17.9999, silence=0.00]
- **yt_scream_-s04sbIViJQ_c008.pt** (unsafe): uniform_rows [energy=16.3768, silence=0.00]
- **yt_scream_-s04sbIViJQ_c010.pt** (unsafe): flat_spectrum, uniform_rows [energy=18.8114, silence=0.00]
- **yt_scream_-s04sbIViJQ_c011.pt** (unsafe): flat_spectrum, uniform_rows [energy=18.8114, silence=0.00]
- **yt_scream_8nXEPiQj0EM_c007.pt** (unsafe): flat_spectrum, uniform_rows [energy=18.8114, silence=0.00]
- **yt_scream_Cvcoj0Oet_8_c001.pt** (unsafe): flat_spectrum, uniform_rows [energy=18.8114, silence=0.00]
- **yt_scream_Cvcoj0Oet_8_c002.pt** (unsafe): uniform_rows [energy=18.2704, silence=0.00]
- **yt_scream_Cvcoj0Oet_8_c003.pt** (unsafe): uniform_rows [energy=18.2704, silence=0.00]
- **yt_scream_Cvcoj0Oet_8_c004.pt** (unsafe): uniform_rows [energy=16.9178, silence=0.00]
- **yt_scream_mQCfBU2Pzws_c019.pt** (unsafe): flat_spectrum, uniform_rows [energy=18.8114, silence=0.00]
- **yt_scream_vLF-26JbWsI_c007.pt** (unsafe): flat_spectrum, uniform_rows [energy=18.8114, silence=0.00]
- **yt_scream_vLF-26JbWsI_c009.pt** (unsafe): flat_spectrum, uniform_rows [energy=18.8114, silence=0.00]

### Potential Mislabeled Samples (0 candidates)

No high-confidence mislabeled candidates detected.

## VAL Split

### Class Distribution by Source

| Source | Safe | Unsafe | Total | Balance |
|--------|------|--------|-------|---------|
| bg | 445 | 0 | 445 | 100/0 |
| cremad | 521 | 363 | 884 | 59/41 |
| esc | 120 | 0 | 120 | 100/0 |
| hns | 526 | 0 | 526 | 100/0 |
| rav | 61 | 65 | 126 | 48/52 |
| savee | 35 | 21 | 56 | 62/38 |
| tess | 137 | 119 | 256 | 54/46 |
| viol | 161 | 144 | 305 | 53/47 |
| yt | 531 | 190 | 721 | 74/26 |
| **TOTAL** | **2537** | **902** | **3439** | **74/26** |

### Quality Issues (9 flagged)

| Issue | Count | Sources |
|-------|-------|---------|
| flat_spectrum | 9 | esc(defaultdict(<class 'int'>, {'flat_spectrum': 1, 'uniform_rows': 1})), yt(defaultdict(<class 'int'>, {'flat_spectrum': 8, 'uniform_rows': 8})) |
| uniform_rows | 9 | esc(defaultdict(<class 'int'>, {'flat_spectrum': 1, 'uniform_rows': 1})), yt(defaultdict(<class 'int'>, {'flat_spectrum': 8, 'uniform_rows': 8})) |

#### Detailed Issues (top 50)

- **esc_1-16746-A-15.pt** (safe): flat_spectrum, uniform_rows [energy=18.8114, silence=0.00]
- **yt_metro_a1_NIp6ohCtQeQ_c002.pt** (safe): flat_spectrum, uniform_rows [energy=18.8114, silence=0.00]
- **yt_metro_b2_aluo9cFGs3Q_c000.pt** (safe): flat_spectrum, uniform_rows [energy=18.8114, silence=0.00]
- **yt_metro_b3_TkmrUkhl0F4_c043.pt** (safe): flat_spectrum, uniform_rows [energy=18.8114, silence=0.00]
- **yt_metro_b3_TkmrUkhl0F4_c044.pt** (safe): flat_spectrum, uniform_rows [energy=18.8114, silence=0.00]
- **yt_scream_8nXEPiQj0EM_c005.pt** (unsafe): flat_spectrum, uniform_rows [energy=18.8114, silence=0.00]
- **yt_scream_Cvcoj0Oet_8_c006.pt** (unsafe): flat_spectrum, uniform_rows [energy=18.8114, silence=0.00]
- **yt_scream_Cvcoj0Oet_8_c007.pt** (unsafe): flat_spectrum, uniform_rows [energy=18.8114, silence=0.00]
- **yt_scream_dK8LGDkHEfA_c000.pt** (unsafe): flat_spectrum, uniform_rows [energy=18.8114, silence=0.00]

### Potential Mislabeled Samples (0 candidates)

No high-confidence mislabeled candidates detected.

## TEST Split

### Class Distribution by Source

| Source | Safe | Unsafe | Total | Balance |
|--------|------|--------|-------|---------|
| bg | 430 | 0 | 430 | 100/0 |
| cremad | 585 | 401 | 986 | 59/41 |
| esc | 114 | 0 | 114 | 100/0 |
| hns | 518 | 0 | 518 | 100/0 |
| rav | 87 | 57 | 144 | 60/40 |
| savee | 28 | 14 | 42 | 67/33 |
| tess | 122 | 125 | 247 | 49/51 |
| viol | 169 | 131 | 300 | 56/44 |
| yt | 506 | 185 | 691 | 73/27 |
| **TOTAL** | **2559** | **913** | **3472** | **74/26** |

### Quality Issues (9 flagged)

| Issue | Count | Sources |
|-------|-------|---------|
| flat_spectrum | 9 | esc(defaultdict(<class 'int'>, {'flat_spectrum': 1, 'uniform_rows': 1})), yt(defaultdict(<class 'int'>, {'flat_spectrum': 8, 'uniform_rows': 8})) |
| uniform_rows | 9 | esc(defaultdict(<class 'int'>, {'flat_spectrum': 1, 'uniform_rows': 1})), yt(defaultdict(<class 'int'>, {'flat_spectrum': 8, 'uniform_rows': 8})) |

#### Detailed Issues (top 50)

- **esc_3-164594-A-15.pt** (safe): flat_spectrum, uniform_rows [energy=18.8114, silence=0.00]
- **yt_metro_b2_aluo9cFGs3Q_c021.pt** (safe): flat_spectrum, uniform_rows [energy=18.8114, silence=0.00]
- **yt_metro_b3_TkmrUkhl0F4_c041.pt** (safe): flat_spectrum, uniform_rows [energy=18.8114, silence=0.00]
- **yt_metro_px_iGq0bHu7QxU_c000.pt** (safe): flat_spectrum, uniform_rows [energy=18.8114, silence=0.00]
- **yt_scream_-s04sbIViJQ_c009.pt** (unsafe): flat_spectrum, uniform_rows [energy=18.8114, silence=0.00]
- **yt_scream_Cvcoj0Oet_8_c005.pt** (unsafe): flat_spectrum, uniform_rows [energy=18.8114, silence=0.00]
- **yt_scream_Cvcoj0Oet_8_c010.pt** (unsafe): flat_spectrum, uniform_rows [energy=18.8114, silence=0.00]
- **yt_scream_dK8LGDkHEfA_c001.pt** (unsafe): flat_spectrum, uniform_rows [energy=18.8114, silence=0.00]
- **yt_scream_mQCfBU2Pzws_c017.pt** (unsafe): flat_spectrum, uniform_rows [energy=18.8114, silence=0.00]

### Potential Mislabeled Samples (0 candidates)

No high-confidence mislabeled candidates detected.

## Recommendations

Based on the audit:

1. **35 samples** with quality issues (silence, low energy, etc.)
2. **0 samples** potentially mislabeled (high-confidence wrong predictions)
3. Review mislabeled candidates from acted speech sources (CREMA-D, RAVDESS) first
4. Consider removing mostly-silent samples as they add no signal
5. Check SAVEE samples carefully — small source with low accuracy

