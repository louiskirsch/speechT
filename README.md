# speechT
An opensource speech-to-text software written in tensorflow.

Python 3 is required.

## Architecture
Currently speechT uses a simple sequence-to-sequence model with GRU cells.  
The input consists of raw audio data imported using soundfile.  
The decoder uses an attention based mechanism, target words are embedded first.  

The speech corpus from http://www.openslr.org/12/ is automatically downloaded.  
**Note:** The corpus is about 30GB!

See also [Sequence-to-Sequence models](https://www.tensorflow.org/versions/r0.11/tutorials/seq2seq/index.html) in tensorflow.

## Training
To start training just run
```
python3 transcribe.py
```

Important flags  
`--data_dir` to specify the data directory to download speech corpus to *(defaults to ./data/)*  
`--train_dir` to specify the train directory to save checkpoints and vocabulary to *(defaults to ./train/)*  
`--size` size of each model layer *(defaults to 1024)*  
`--num_layers` number of layers in the model *(defaults to 3)*  
`--vocab_size` maximum vocabulary size *(defaults to 40000)*  

## Testing
To test your trained model run
```
python3 transcribe.py --decode_tests [--data_dir PATH] [--train_dir PATH]
```

## Live usage
To record using your microphone and convert it to text run
```
python3 transcribe.py --decode [--data_dir PATH] [--train_dir PATH]
```
