import json
import os
from typing import List

import music21 as m21
import numpy as np
from tqdm import tqdm


class Preprocessor:
    def __init__(self):
        self.raw_data_path = "raw-data/deutschl"
        self.songs_output_path = "data/songs.txt"
        self.sequence_length = 64
        self.acceptable_durations = [0.25, 0.5, 0.75, 1.0, 1.5, 2, 3, 4]
        self.time_step = self.acceptable_durations[0]

        self.songs = self.load_kern_to_m21()
        self.songs_encoded = [
            self.encode_song_underscore(self.transpose(song))
            for song in self.songs
            if self.has_acceptable_note_durations(song)
        ]

    def load_kern_to_m21(self) -> List[m21.stream.Score]:
        songs = []
        for path, _, files in os.walk(self.raw_data_path):
            print(path, files)
            for file in files:
                if file[-4:] == ".krn":
                    songs.append(m21.converter.parse(os.path.join(path, file)))

        return songs

    def has_acceptable_note_durations(self, song: m21.stream.Score) -> bool:
        for note in song.flat.notesAndRests:
            if note.duration.quarterLength not in self.acceptable_durations:
                return False

        return True

    @staticmethod
    def transpose(song: m21.stream.Score) -> m21.stream.Score:
        key = song.getElementsByClass(m21.stream.Part)[0].getElementsByClass(m21.stream.Measure)[0][4]

        if not isinstance(key, m21.key.Key):
            key = song.analyze("key")

        if key.mode == "major":
            interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C"))
        elif key.mode == "minor":
            interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))

        return song.transpose(interval)

    def encode_song_underscore(self, song: m21.stream.Score):
        encoded_song = []
        for event in song.flat.notesAndRests:
            if isinstance(event, m21.note.Note):
                symbol = event.pitch.midi
            elif isinstance(event, m21.note.Rest):
                symbol = "r"
            else:
                print(f"Unknown event: {event}")

            encoded_song.append(str(symbol))

            for step in range(int(event.duration.quarterLength / self.time_step) - 1):
                encoded_song.append("_")

        return ",".join(encoded_song)

    def encode_song(self, song: m21.stream.Score):
        encoded_song = []
        for event in song.flat.notesAndRests:
            if isinstance(event, m21.note.Note):
                pitch = event.pitch.midi
            elif isinstance(event, m21.note.Rest):
                pitch = -1
            else:
                print(f"Unknown event: {event}")

            encoded_song.append(np.array([pitch, int(event.duration.quarterLength / self.time_step)]))

        return np.array(encoded_song)

    def write_encoded_songs(self):
        with open(self.songs_output_path, "w") as f:
            for song in self.songs_encoded:
                f.write(song + "\n")


class SequenceGenerator:
    def __init__(self):
        self.songs_output_path = "data/songs.txt"
        self.all_songs_output_path = "data/all_songs.txt"
        self.mapping_path = "data/dictionary.json"
        self.all_songs_int_output_path = "data/all_songs_int.npy"
        self.inputs_path = "data/inputs.npy"
        self.targets_path = "data/targets.npy"

        self.vocabulary_size = -1
        self.sequence_length = 64

        self.all_songs = self.create_single_file_dataset()
        self.dictionary = self.create_note_dictionary()
        self.all_songs_int = self.generate_encoded_dataset()

    def create_single_file_dataset(self):
        try:
            with open(self.all_songs_output_path, "r") as h:
                all_songs = h.read()

            print(f"Using existing file: {self.all_songs_output_path}")

        except FileNotFoundError:
            delimiter = "/," * self.sequence_length

            with open(self.songs_output_path, "r") as f:
                all_songs = (f.read()).replace("\n", f",{delimiter}")[:-1]

            with open(self.all_songs_output_path, "w") as g:
                g.write(all_songs)

        return all_songs.split(",")

    def create_note_dictionary(self):
        songs = self.all_songs
        vocabulary = list(set(songs))
        self.vocabulary_size = len(vocabulary)

        try:
            with open(self.mapping_path, "r") as g:
                dictionary = json.load(g)

            print(f"Using existing file: {self.mapping_path}")

        except FileNotFoundError:
            dictionary = dict()

            for i, symbol, in enumerate(vocabulary):
                dictionary[symbol] = i

            with open(self.mapping_path, "w") as f:
                json.dump(dictionary, f, indent=4)

        return dictionary

    def generate_encoded_dataset(self):
        try:
            all_songs_int = np.load(self.all_songs_int_output_path)

            print(f"Using existing file: {self.all_songs_int_output_path}")

        except FileNotFoundError:
            all_songs_int = np.vectorize(self.dictionary.__getitem__)(self.all_songs)
            np.save(self.all_songs_int_output_path, all_songs_int)

        return all_songs_int

    @staticmethod
    def one_hot(input_data: np.array, number_classes: int):
        return np.eye(number_classes)[input_data.reshape(-1)]

    def generate_training_sequences(self):
        try:
            inputs = []
            targets = []
            with open(self.inputs_path, "r") as h:
                inputs = np.load(h)

            with open(self.targets_path, "r") as i:
                targets = np.load(i)

            print(f"Using existing files: {self.inputs_path}, {self.targets_path}")

        except FileNotFoundError:
            inputs = []
            targets = []

            num_sequences = len(self.all_songs_int) - self.sequence_length

            for i in tqdm(range(num_sequences)):
                inputs.append(self.one_hot(self.all_songs_int[i:i + self.sequence_length], self.vocabulary_size))
                targets.append(self.all_songs_int[i + self.sequence_length])

        print(f"Number of sequences = {len(inputs)}")

        return inputs, targets
