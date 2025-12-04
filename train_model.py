from ml_model import generate_dataset, train_model
df = generate_dataset(3000)
train_model(df)