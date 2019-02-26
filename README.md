Temporal Convolutional Network for CWS
====

This is the source code for the paper, Chinese Word Segmentation Using Temporal Convolutional Networks. It achieves competitive performance to state-of-the-art Bi-LSTM model.

Requirement:
======
	Python: 2.7   
	tensorflow-gpu: 1.8.0

Data format:
======
CoNLL format (prefer BMES tag scheme, compatible with the Chinese word segmentation scheme), with each character its label for one line. Sentences are splited with a null line.

	中	B
	国	E
	财	B
	团	E
	买	B
	下	E
	AC	B
	米	M
	兰	E

	代	B
	价	E
	超	S
	10	B
	亿	E
	欧	B
	元	E

How to run the code?
====
1. Training the model on NLPCC2016.
```
python main.py --training True
```
2. Training the Baseline model(Bi-LSTM) on NLPCC2016.
```
python main.py -m bilstm --training True
```
3. Evaluate the model on NLPCC216
```
python main.py --model_path saved/tcn/20190223_220045/
```
4. Evaluate the Baseline model on NLPCC2016.
```
python main.py -m bilstm --model_path saved/bilstm/20190224_144926/
```
5. Train the model on the other dataset.
```
python main.py -e 'embeddings file path' --train 'train set path' --training True
```
6. Evaluate the model on the other dataset.
```
python main.py -e 'embeddings file path' --test 'test set path' --model_path 'saved model parameters directory path'