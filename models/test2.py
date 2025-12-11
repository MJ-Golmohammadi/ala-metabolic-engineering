from cobra import io
adjusted_model = io.read_sbml_model("models/context_specific/iJN1463_Glu_eflux.xml")
adjusted_model.objective = "BIOMASS_KT2440_WT3"
#print(adjusted_model.optimize().status, adjusted_model.optimize().objective_value)
# 1. بارگذاری مدل اصلاح‌شده (همان adjusted_model که pipeline برمی‌گرداند)
# اگر در همان اسکریپت هستی، adjusted_model را استفاده کن؛ در غیر این صورت فایل SBML را بخوان
from cobra import io
# adjusted_model = io.read_sbml_model("models/context_specific/iJN1463_Glu_eflux.xml")  # اگر لازم است
m = adjusted_model

# 2. اطمینان از هدف و تست سریع
m.objective = "BIOMASS_KT2440_WT3"
sol = m.optimize()
print("status:", sol.status)
print("objective_value:", sol.objective_value)
print("message:", getattr(sol, "message", None))

# 3. واکنش با ظرفیت خیلی کوچک را اصلاح کن (PIt7ipp)
if "PIt7ipp" in m.reactions:
    r = m.reactions.get_by_id("PIt7ipp")
    print("PIt7ipp bounds before:", r.lower_bound, r.upper_bound)
    # بازگردانی به مقدار کوچک معقول یا مقدار اصلی (اگر original_bounds داری از آن استفاده کن)
    r.lower_bound = float(0.0)
    r.upper_bound = float(1000.0)
    print("PIt7ipp bounds after:", r.lower_bound, r.upper_bound)

# 4. اطمینان از uptakeهای کلیدی (اگر بسته‌اند بازشان کن)
for rxn_id, lb in [("EX_glc__D_e", -10.0), ("EX_o2_e", -20.0), ("EX_nh4_e", -10.0), ("EX_pi_e", -10.0), ("EX_so4_e", -10.0)]:
    if rxn_id in m.reactions:
        r = m.reactions.get_by_id(rxn_id)
        r.lower_bound = float(lb)
        r.upper_bound = float(1000.0)

# 5. مطمئن شو BIOMASS و ATPM منطقی‌اند
if "BIOMASS_KT2440_WT3" in m.reactions:
    b = m.reactions.get_by_id("BIOMASS_KT2440_WT3")
    print("BIOMASS bounds:", float(b.lower_bound), float(b.upper_bound))
    # اگر lb خیلی بزرگ یا غیرمعمول است، آن را به 0 برگردان
    if float(b.lower_bound) > 1e-4:
        b.lower_bound = 0.0

if "ATPM" in m.reactions:
    a = m.reactions.get_by_id("ATPM")
    print("ATPM bounds:", float(a.lower_bound), float(a.upper_bound))

# 6. تست مجدد
sol2 = m.optimize()
print("after quick fixes status:", sol2.status, "objective:", sol2.objective_value)

