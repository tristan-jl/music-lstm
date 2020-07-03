import json
from time import time
import music21 as m21
import numpy as np
import torch

from dataset import MusicDataset
from model import LSTM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MelodyGenerator:
    def __init__(self, model_path="model.pt"):
        self.model = LSTM(45, 256, 45)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.sequence_length = 64

        with open("data/dictionary.json", "r") as f:
            self._mappings = json.load(f)

        self.melody = None

    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):
        seed = seed.split()
        melody = seed
        seed = [self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):
            seed = seed[-max_sequence_length:]
            sequence = MusicDataset.one_hot(np.array(seed), len(self._mappings))
            sequence = torch.from_numpy(sequence[np.newaxis, ...]).float()

            outputs = self.model(sequence)[0]
            probabilities = outputs.data.detach()
            output_int = self.__sample_with_temperature(probabilities, temperature)
            seed.append(output_int)

            output_symbol = [k for k, v in self._mappings.items() if v == output_int][0]

            if output_symbol == "/":
                break

            melody.append(output_symbol)

        self.melody = melody
        return melody

    @staticmethod
    def __sample_with_temperature(probabilities, temperature):
        probabilities -= torch.min(probabilities)
        predictions = torch.log(probabilities) / temperature
        probabilities = torch.exp(predictions) / torch.sum(torch.exp(predictions))

        print(probabilities)

        choices = range(len(probabilities))
        index = np.random.choice(choices, p=probabilities.numpy())

        return index

    def save_melody(
        self,
        melody=None,
        step_duration=0.25,
        format="midi",
        file_name=f"output/{int(time())}.mid",
    ):
        if melody is None:
            melody = self.melody

        stream = m21.stream.Stream()

        start_symbol = None
        step_counter = 1

        for i, symbol in enumerate(melody):
            if symbol != "_" or i + 1 == len(melody):
                if start_symbol is not None:
                    quarter_length_duration = step_duration * step_counter

                    if start_symbol == "r":
                        m21_event = m21.note.Rest(quarterLength=quarter_length_duration)

                    else:
                        m21_event = m21.note.Note(
                            int(start_symbol), quarterLength=quarter_length_duration
                        )

                    stream.append(m21_event)
                    step_counter = 1

                start_symbol = symbol

            else:
                step_counter += 1

        stream.write(format, file_name)


def main():
    mg = MelodyGenerator()
    seed = "60 _ _ _ 60 _ 55 _ 57 _ 55 _ 60 _ _ _ _ _ 64 _ 62 _ 60 _ 59"
    melody = mg.generate_melody(seed, 500, 64, 0.05)
    print(melody)
    mg.save_melody()


if __name__ == "__main__":
    main()
