# scale_bounds_and_test_growth.py
import cobra
from pathlib import Path

MODEL_PATH = "models/iJN1463.xml"   # مسیر مدل خودت را اینجا بگذار
BIOMASS_RXN = "BIOMASS_KT2440_WT3"  # نام واکنش بیومس در مدل
EXCLUDE_EXCHANGES = True            # اگر می‌خواهی EX_ ها را تغییر ندهد، True بگذار
SCALE_FACTOR = 0.1                  # یک دهم

def load_model(path):
    return cobra.io.read_sbml_model(path)

def get_growth(model, biomass_rxn=BIOMASS_RXN):
    if biomass_rxn not in model.reactions:
        raise KeyError(f"Biomass reaction {biomass_rxn} not found.")
    model.objective = biomass_rxn
    sol = model.optimize()
    return sol.status, float(sol.objective_value) if sol.status == 'optimal' else None

def scale_bounds(model, factor=SCALE_FACTOR, exclude_ex=True):
    original_bounds = {}
    for r in model.reactions:
        original_bounds[r.id] = (float(r.lower_bound), float(r.upper_bound))
        if exclude_ex and r.id.startswith("EX_"):
            continue
        # scale numeric bounds
        lb = float(r.lower_bound)
        ub = float(r.upper_bound)
        # handle large "infinite-like" bounds consistently
        # if bound magnitude is extremely large, keep it large but scale
        new_lb = lb * factor
        new_ub = ub * factor
        # ensure numeric types
        r.lower_bound = float(new_lb)
        r.upper_bound = float(new_ub)
    return original_bounds

def report_tiny_reactions(model, tol=1e-6):
    tiny = []
    for r in model.reactions:
        lb = float(r.lower_bound)
        ub = float(r.upper_bound)
        if abs(ub - lb) <= tol:
            tiny.append((r.id, lb, ub))
    return tiny

def main():
    print("Loading model:", MODEL_PATH)
    m = load_model(MODEL_PATH)
    status_before, growth_before = get_growth(m)
    print("Before scaling -> status:", status_before, "growth:", growth_before)

    # copy model to avoid overwriting original in memory if desired
    m_scaled = m.copy()

    original_bounds = scale_bounds(m_scaled, factor=SCALE_FACTOR, exclude_ex=EXCLUDE_EXCHANGES)
    print(f"Scaled bounds by factor {SCALE_FACTOR} (exclude_ex={EXCLUDE_EXCHANGES}).")

    status_after, growth_after = get_growth(m_scaled)
    print("After scaling -> status:", status_after, "growth:", growth_after)

    tiny = report_tiny_reactions(m_scaled, tol=1e-6)
    print("Number of reactions with (ub-lb) <= 1e-6:", len(tiny))
    if len(tiny) > 0:
        print("Sample tiny reactions (id, lb, ub):")
        for t in tiny[:50]:
            print(" ", t)

    # optional: save scaled model to file for inspection
    out_path = Path("models/context_specific") / f"{Path(MODEL_PATH).stem}_scaled_bounds.xml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cobra.io.write_sbml_model(m_scaled, str(out_path))
    print("Scaled model written to:", out_path)

    # also return objects if used interactively
    return m, m_scaled, original_bounds

if __name__ == "__main__":
    main()
