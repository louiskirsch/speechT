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
python3 preprocess.py
```

Then, to run the training, execute
```
python3 train.py
```

Important flags  
`--data_dir` to specify the data directory to download speech corpus to *(defaults to ./data/)*  
`--train_dir` to specify the train directory to save checkpoints and vocabulary to *(defaults to ./train/)*  

## Testing

Not yet implemented.

## Live usage

Not yet implemented.
