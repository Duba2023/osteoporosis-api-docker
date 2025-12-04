import joblib

# Load scaler
scaler = joblib.load('scaler.joblib')

print("=" * 70)
print("SCALER INSPECTION")
print("=" * 70)

print(f"\nScaler Type: {type(scaler).__name__}")
print(f"Scaler Module: {type(scaler).__module__}")

# Get attributes
print("\nScaler Attributes:")
print(f"  • Mean values: {scaler.mean_ if hasattr(scaler, 'mean_') else 'N/A'}")
print(f"  • Scale values: {scaler.scale_ if hasattr(scaler, 'scale_') else 'N/A'}")
print(f"  • Variance: {scaler.var_ if hasattr(scaler, 'var_') else 'N/A'}")
print(f"  • N features: {scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else 'N/A'}")

# Feature names
if hasattr(scaler, 'get_feature_names_out'):
    try:
        features = scaler.get_feature_names_out()
        print(f"  • Feature names: {features}")
    except:
        print("  • Feature names: Unable to extract")

print("\n" + "=" * 70)
print("✅ Scaler loaded successfully and ready for integration!")
print("=" * 70)
