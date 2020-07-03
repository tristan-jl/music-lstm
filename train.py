import time
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import MusicDataset, split_train_validation
from model import LSTM

path = "./"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

params = {"batch_size": 64, "shuffle": True, "num_workers": 0}
num_epochs = 500
learning_rate = 0.001
num_sequences = 1140090
sequence_length = 64
input_size = 45

train_ids, validation_ids = split_train_validation(np.arange(num_sequences))
partition = {"train": train_ids, "validation": validation_ids}

training_set = MusicDataset(
    partition["train"], f"{path}/data/all_songs_int.npy", f"{path}/data/dictionary.json"
)
train_loader = DataLoader(training_set, **params)

validation_set = MusicDataset(
    partition["validation"],
    f"{path}/data/all_songs_int.npy",
    f"{path}/data/dictionary.json",
)
validation_loader = DataLoader(validation_set, **params)

model = LSTM(training_set.vocabulary_size, 256, 45)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(train_loader)
loss_values = []
validation_values = []

for epoch in range(num_epochs):
    start_time = time.time()

    model.train()
    running_loss = 0.0
    for i, (sequences, labels) in enumerate(train_loader):
        sequences = sequences.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)

        optimiser.zero_grad()

        outputs = model(sequences)
        loss = criterion(outputs, labels.long())

        loss.backward()
        optimiser.step()

        if (i + 1) % 1000 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{total_step}], Loss: {loss.item()}"
            )

        running_loss = +loss.item() * sequences.size(0)

    loss_values.append(running_loss / len(training_set))

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (sequences, labels) in enumerate(validation_loader):
            sequences = sequences.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)
            outputs = model(sequences)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    validation_accuracy = 1.0 * correct / total
    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(training_set)}, Validation Accuracy: {validation_accuracy * 100}%, Time: {time.time() - start_time}"
    )
    validation_values.append(validation_accuracy)

    if epoch % 5 == 0:
        print("Checkpointing...")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimiser_state_dict": optimiser.state_dict(),
                "loss_values": loss_values,
                "validation_values": validation_values,
            },
            f"{path}/checkpoint/epoch-{epoch}.tar",
        )

torch.save(model.state_dict(), f"{path}/model.pt")
