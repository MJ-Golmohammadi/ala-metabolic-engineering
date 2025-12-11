from cobra import io
adjusted_model = io.read_sbml_model("models/context_specific/iJN1463_Glu_eflux.xml")
adjusted_model.objective = "BIOMASS_KT2440_WT3"
#print(adjusted_model.optimize().status, adjusted_model.optimize().objective_value)
# 1. بارگذاری مدل اصلاح‌شده (همان adjusted_model که pipeline برمی‌گرداند)
# اگر در همان اسکریپت هستی، adjusted_model را استفاده کن؛ در غیر این صورت فایل SBML را بخوان
from cobra import io
# adjusted_model = io.read_sbml_model("models/context_specific/iJN1463_Glu_eflux.xml")  # اگر لازم است
# اجرا در همان محیط که adjusted_model در دسترس است
m = adjusted_model  # یا نام متغیری که مدل اصلاح‌شده را نگه می‌دارد

# 1) بیومس را به حالت معمول برگردان
if "BIOMASS_KT2440_WT3" in m.reactions:
    b = m.reactions.get_by_id("BIOMASS_KT2440_WT3")
    print("BIOMASS bounds before:", float(b.lower_bound), float(b.upper_bound))
    b.lower_bound = 0.0
    b.upper_bound = float(1e6)
    print("BIOMASS bounds after:", float(b.lower_bound), float(b.upper_bound))

# 2) ATPM را آزاد کن (اگر ثابت است)
if "ATPM" in m.reactions:
    a = m.reactions.get_by_id("ATPM")
    print("ATPM bounds before:", float(a.lower_bound), float(a.upper_bound))
    # اگر قبلاً equality بود، بازش کن به بازه معقول
    a.lower_bound = 0.0
    a.upper_bound = 1000.0
    print("ATPM bounds after:", float(a.lower_bound), float(a.upper_bound))

# 3) اطمینان از uptakeهای کلیدی (مثال: glucose, o2, nh4, pi, so4)
for rxn_id, lb in [("EX_glc__D_e", -10.0), ("EX_o2_e", -20.0), ("EX_nh4_e", -10.0), ("EX_pi_e", -10.0), ("EX_so4_e", -10.0)]:
    if rxn_id in m.reactions:
        r = m.reactions.get_by_id(rxn_id)
        r.lower_bound = float(lb)
        r.upper_bound = 1000.0

# 4) حل مجدد و چاپ نتیجه
m.objective = "BIOMASS_KT2440_WT3"
sol = m.optimize()
print("status after fixes:", sol.status)
print("objective_value after fixes:", sol.objective_value)
print("solver message:", getattr(sol, "message", None))
