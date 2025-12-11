import pandas as pd
import ast

# فایل‌ها را بخوان
CSV_SEPARATOR = ";"
df_bounds = pd.read_csv("models/final_constrained_rnaseq_thermo/reaction_bounds_summary_Cit.csv", sep=CSV_SEPARATOR)   # شامل reaction_id و final_upper_bound
df_ids = pd.read_csv("config/essential_rxns.csv", sep=CSV_SEPARATOR)      # شامل ستون ids

# تابع برای parse کردن ستون ids
def parse_ids_field(field):
    if pd.isna(field):
        return []
    try:
        # اگر شبیه مجموعه یا لیست پایتونی باشد
        parsed = ast.literal_eval(str(field))
        if isinstance(parsed, (set, list, tuple)):
            return list(parsed)
        else:
            return [str(parsed)]
    except Exception:
        # اگر فقط رشته ساده باشد
        return [x.strip() for x in str(field).strip("{} ").split(",") if x.strip()]

# ستون ids را به لیست تبدیل کن
df_ids["parsed_ids"] = df_ids["ids"].apply(parse_ids_field)

# همه شناسه‌های موجود در فایل دوم را یکجا جمع کن
all_ids = set()
for ids in df_ids["parsed_ids"]:
    all_ids.update(ids)

# فیلتر روی فایل اول: reaction_id در لیست ids و final_upper_bound < 5
filtered = df_bounds[
    (df_bounds["reaction_id"].isin(all_ids)) &
    (df_bounds["final_upper_bound"] < 5)
]

# نتایج را چاپ کن
print(filtered[["reaction_id", "final_upper_bound"]])

# ذخیره در فایل جدید
filtered.to_csv("filtered_results.csv", index=False)
