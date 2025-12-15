import cobra
import pandas as pd

# مدل خودت را بارگذاری کن (مسیر فایل SBML را جایگزین کن)
model = cobra.io.read_sbml_model("models/iJN1463.xml")

# هدف را روی واکنش بایومس بگذار
model.objective = model.reactions.get_by_id("BIOMASS_KT2440_WT3")

# بهینه‌سازی مدل
solution = model.optimize()

# ساخت جدول شامل reaction id و flux متناظر
flux_df = pd.DataFrame({
    "reaction_id": [rxn.id for rxn in model.reactions],
    "flux_value": [solution.fluxes.get(rxn.id, 0.0) for rxn in model.reactions]
})

# نمایش چند ردیف اول
print(flux_df.head())

# ذخیره به فایل CSV اگر لازم باشد
flux_df.to_csv("biomass_fluxes.csv", index=False)
