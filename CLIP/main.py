from train_eval import train_model

if __name__ == "__main__":
    trained_model = train_model(
        data_root="../data/MMHS150K",
        num_epochs=10,
        batch_size=64,
        num_workers=32,
        lr=1e-5,
        version="v1",
    )
