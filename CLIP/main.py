from train_eval import train_model

if __name__ == "__main__":
    trained_model = train_model(
        data_root="../data/MMHS150K", num_epochs=5, batch_size=16, version="v1"
    )
