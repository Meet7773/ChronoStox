import os
import sys
import pandas as pd
import joblib

def main():
    print("Loading models directly...")
    try:
        model_dir = "."
        joblib_path = os.path.join(model_dir, "sector_model_v7_UNIVERSAL_20251113_063532.joblib")
        bundle = joblib.load(joblib_path)
    except Exception as e:
        print(f"Failed to load models: {e}")
        return

    features = bundle["features"]
    lgbm = bundle["model_lgbm"]

    print("\n========================================")
    print("  LightGBM Feature Importances (Top 25) ")
    print("========================================")
    
    # The lgbm model is a MultiOutputRegressor
    # We can average the feature importances across all the estimators (one per horizon)
    if hasattr(lgbm, "estimators_"):
        import numpy as np
        all_importances = []
        for est in lgbm.estimators_:
            # Each estimator might be a pipeline or a direct lgbm model
            imp = None
            if hasattr(est, "feature_importances_"):
                imp = est.feature_importances_
            elif hasattr(est, "named_steps"):
                step_name = list(est.named_steps.keys())[-1]
                model_step = est.named_steps[step_name]
                if hasattr(model_step, "feature_importances_"):
                    imp = model_step.feature_importances_
            if imp is not None:
                all_importances.append(imp)
                
        if all_importances:
            avg_importances = np.mean(all_importances, axis=0)
            if len(avg_importances) == len(features):
                df_imp = pd.DataFrame({"Feature": features, "Importance": avg_importances})
                df_imp = df_imp.sort_values(by="Importance", ascending=False)
                print(df_imp.head(25).to_string(index=False))
            else:
                 print(f"Feature lengths do not match importances length. Features: {len(features)}, Importances: {len(avg_importances)}")
        else:
            print("Could not find feature_importances_ in any of the underlying estimators.")
    else:
        print(f"LGBM model not found or does not have estimators_. Type: {type(lgbm)}")

    print("\n========================================\n")

if __name__ == "__main__":
    main()
