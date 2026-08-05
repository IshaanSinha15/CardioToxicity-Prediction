from classification_backend.xai.model_loader import ModelLoader

loader = ModelLoader()

model = loader.load_model()

print(type(model))