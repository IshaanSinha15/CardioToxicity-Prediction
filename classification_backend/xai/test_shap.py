from classification_backend.xai.shap_explainer import ShapExplainer

feature_names = [
    "RMP",
    "Peak",
    "APD50",
    "APD90",
    "Triangulation",
    "APA",
    "Block_IKr",
    "Block_INa",
    "Block_INaL",
    "Block_ICaL",
    "Block_IKs",
    "Block_IK1",
    "Block_Ito",
]

sample = [
    -90.74,
    31.99,
    231.9,
    278.4,
    46.5,
    122.7,
    26.7,
    2.6,
    0.0,
    2.1,
    0.0,
    0.0,
    4.5,
]

explainer = ShapExplainer()

result = explainer.explain(sample, feature_names)

print(result["prediction"])
print(result["confidence"])