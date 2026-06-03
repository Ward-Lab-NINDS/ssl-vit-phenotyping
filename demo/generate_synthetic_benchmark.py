from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
out = Path(__file__).resolve().parent / "synthetic_outputs"
out.mkdir(parents=True, exist_ok=True)

rows = []
sgRNAs = ["NTC", "MAPT_1", "MAPT_2", "LRRK2_1"]
for plate in ["plate_01", "plate_02"]:
    for well_i, well in enumerate(["A01", "A02", "B01", "B02"]):
        for i in range(45):
            label = len(rows) + 1
            sgrna = sgRNAs[(i + well_i) % len(sgRNAs)]
            effect = {"NTC": 0.0, "MAPT_1": 1.5, "MAPT_2": 1.2, "LRRK2_1": -1.0}[sgrna]
            batch = 0.15 if plate == "plate_02" else 0.0
            rows.append(
                {
                    "label": label,
                    "sgRNA": sgrna,
                    "perturbation": sgrna.split("_")[0],
                    "control_type": "NTC" if sgrna == "NTC" else "positive",
                    "replicate": plate,
                    "plate": plate,
                    "well": well,
                    "density": "low" if well.startswith("A") else "high",
                    "cell_area": rng.normal(100 + effect * 8, 6),
                    "nuclei_area": rng.normal(40 + effect * 3, 3),
                    "cytoplasm_texture": rng.normal(0.4 + effect * 0.08, 0.05),
                    "ssl_000": rng.normal(effect + batch, 0.35),
                    "ssl_001": rng.normal(effect * 0.6 + batch, 0.35),
                    "ssl_002": rng.normal(-effect * 0.4 + batch, 0.35),
                    "procode_ch1": rng.normal(1.0 if sgrna in {"MAPT_1", "LRRK2_1"} else 0.15, 0.08),
                    "procode_ch2": rng.normal(1.0 if sgrna in {"MAPT_2", "LRRK2_1"} else 0.15, 0.08),
                }
            )

table = pd.DataFrame(rows)
table.to_csv(out / "synthetic_phenotype_cp_ssl.tsv", sep="\t", index=False)
print(out / "synthetic_phenotype_cp_ssl.tsv")
