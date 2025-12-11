# scale_bounds_and_test_growth_ex.py
import cobra
from pathlib import Path

MODEL_PATH = "models/iJN1463.xml"   # مسیر مدل خودت
BIOMASS_RXN = "BIOMASS_KT2440_WT3"

# دیکشنری uptakeهای خاص با bounds مشخص
special_ex_bounds = {
    "EX_cit_e": (-1.0, 1000.0),
    "EX_o2_e": (-20.0, 1000.0),
    "EX_nh4_e": (-10.0, 1000.0),
    "EX_pi_e": (-5.0, 1000.0),
    "EX_so4_e": (-2.0, 1000.0),
    "EX_fe2_e": (-0.5, 1000.0),
    "EX_mg2_e": (-0.5, 1000.0),
    "EX_k_e": (-0.5, 1000.0),
    "EX_ca2_e": (-0.2, 1000.0),
    "EX_cl_e": (-0.5, 1000.0),
    "EX_na1_e": (-0.5, 1000.0),
    "EX_mn2_e": (-0.05, 1000.0),
    "EX_zn2_e": (-0.05, 1000.0),
    "EX_cobalt2_e": (-0.05, 1000.0),
    "EX_cu2_e": (-0.01, 1000.0),
    "EX_ni2_e": (-0.01, 1000.0),
    "EX_mobd_e": (-0.01, 1000.0),
    "EX_co2_e": (0.0, 1000.0),
    "EX_h2o_e": (0.0, 1000.0),
    "EX_h_e": (-1000.0, 1000.0),
}

def load_model(path):
    return cobra.io.read_sbml_model(path)

def get_growth(model, biomass_rxn=BIOMASS_RXN):
    model.objective = biomass_rxn
    sol = model.optimize()
    return sol.status, float(sol.objective_value) if sol.status == 'optimal' else None

def adjust_ex_bounds(model):
    original_bounds = {}
    for r in model.reactions:
        if r.id.startswith("EX_"):
            original_bounds[r.id] = (float(r.lower_bound), float(r.upper_bound))
            if r.id in special_ex_bounds:
                lb, ub = special_ex_bounds[r.id]
                r.lower_bound = float(lb)
                r.upper_bound = float(ub)
            else:
                # همهٔ EX_ های دیگر بسته شوند
                r.lower_bound = 0.0
                r.upper_bound = 0.0
    return original_bounds

def main():
    print("Loading model:", MODEL_PATH)
    m = load_model(MODEL_PATH)
    status_before, growth_before = get_growth(m)
    print("Before adjustment -> status:", status_before, "growth:", growth_before)

    m_adj = m.copy()
    original_bounds = adjust_ex_bounds(m_adj)
    print("Adjusted EX_ bounds: special set open, others closed.")

    status_after, growth_after = get_growth(m_adj)
    print("After adjustment -> status:", status_after, "growth:", growth_after)

    # ذخیرهٔ مدل اصلاح‌شده
    out_path = Path("models/context_specific") / f"{Path(MODEL_PATH).stem}_ex_adjusted.xml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cobra.io.write_sbml_model(m_adj, str(out_path))
    print("Adjusted model written to:", out_path)

    return m, m_adj, original_bounds

if __name__ == "__main__":
    main()
