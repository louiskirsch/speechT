# speechT
An opensource speech-to-text software written in tensorflow.

Python 3 is required.

## Architecture
Currently speechT is based on the [Wav2Letter paper](https://arxiv.org/abs/1609.03193) and the CTC loss function.

The speech corpus from http://www.openslr.org/12/ is automatically downloaded.  
**Note:** The corpus is about 30GB!

## Training
The data must be preprocessed before training
```
python3 cli.py preprocess
```

Then, to run the training, execute
```
python3 cli.py train
```

Use `--help` for more details.

## Testing

To evaluate on the test set run
```
python3 cli.py evaluate
```

Use `--help` for more details.

## Live usage

To record using your microphone and then print the transcription run
```
python3 cli.py record
```

Use `--help` for more details.
