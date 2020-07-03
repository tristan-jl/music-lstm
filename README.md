# Music LSTM

Music LSTM generates a melody, given a seed, using a PyTorch LSTM.
Inspired by [this series of videos](https://www.youtube.com/playlist?list=PL-wATfeyAMNr0KMutwtbeDCmpwvtul-Xz).
Used Google Colab GPUs to train model.

Uses data from the [ESAC Database](http://www.esac-data.org/).

## Model Architecture

LSTM(\
    (lstm): LSTM(45, 256, batch_first=True) \
    (drop): Dropout(p=0.5, inplace=False) \
    (fc): Linear(in_features=256, out_features=45, bias=True) \
)


## License
[MIT](LICENSE.md)
