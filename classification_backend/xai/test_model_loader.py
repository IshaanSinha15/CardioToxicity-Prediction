from classification_backend.xai.model_loader import ModelLoader


def main():
    loader = ModelLoader()

    model = loader.load_model()

    print("\nModel Type")
    print(type(model))


if __name__ == "__main__":
    main()