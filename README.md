# speechT
An opensource speech-to-text software written in tensorflow.

Python 3 is required.

## Architecture
Currently speechT is based on bidirectional RNNs and the CTC loss function.
Convolutions are still due to be added.

The speech corpus from http://www.openslr.org/12/ is automatically downloaded.  
**Note:** The corpus is about 30GB!

## Training
To start training just run
```
python3 train.py
```

Important flags  
`--data_dir` to specify the data directory to download speech corpus to *(defaults to ./data/)*  
`--train_dir` to specify the train directory to save checkpoints and vocabulary to *(defaults to ./train/)*  
`--size` size of each model layer *(defaults to 1024)*  
`--num_layers` number of layers in the model *(defaults to 3)*  

## Testing

Not yet implemented.

## Live usage

Not yet implemented.
