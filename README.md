# speechT
An opensource speech-to-text software written in tensorflow.



## Installation

### Prerequisites

Python3, portaudio19-dev and ffmpeg are required.

On Ubuntu install via
```
sudo apt install python3-pip portaudio19-dev ffmpeg
```

### Install via pip3

```
pip3 install git+https://github.com/timediv/speechT
```

## Architecture
Currently speechT is based on the [Wav2Letter paper](https://arxiv.org/abs/1609.03193) and the CTC loss function.

The speech corpus from http://www.openslr.org/12/ is automatically downloaded.  
**Note:** The corpus is about 30GB!

## Training
The data must be preprocessed before training
```
speecht-cli preprocess
```

Then, to run the training, execute
```
speecht-cli train
```

Use `--help` for more details.

## Testing

To evaluate on the test set run
```
speecht-cli evaluate
```

Use `--help` for more details.

## Live usage

To record using your microphone and then print the transcription run
```
speecht-cli record
```

Use `--help` for more details.
