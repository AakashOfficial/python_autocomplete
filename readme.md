**`master` branch is underdevelopment. Checkout [simple_lstm](https://github.com/vpj/python_autocomplete/tree/simple_lstm) branch to try it out.**

[This](https://github.com/vpj/python_autocomplete) a toy project we started to see how well a simple LSTM model can autocomplete python code.

It gives quite decent results by saving above 30% key strokes in most files, and close to 50% in some. We calculated key strokes saved by making a single (best) prediction and selecting it with a single key.

We do a beam search to find predictions, upto ~10 characters ahead. So far it's too inefficient, if you are wondering about editor integration.

We train and predict on after cleaning comments, strings and blank lines in python code.
The model is trained after tokenizing python code. It seems more efficient than character level prediction with byte-pair encoding.

A saved model is included in this repo. It is trained on [tensorflow/models](https://github.com/tensorflow/models).

Here's a sample evaluation on a source file from validation set. Green characters are when a autocompletion started; i.e. user presses TAB to select the completion. The green character and and the following characters highlighted in gray are autocompleted. As you can see, it starts and ends completions arbitarily. That is a suggestion could be 'tensorfl' and not the complete identifier 'tensorflow' which can be a little annoying in a real usage scenario. We can limit them to finish on end of tokens to fix that. Also you can notice that it completes across operators as well. Increasing the length of the beam search will let it complete longer pieces of code.

<p align="center">
  <img src="/python-autocomplete.png?raw=true" width="100%" title="Screenshot">
</p>

## Try it yourself

1. Checkout `simple_lstm` branch.

2. Setup [lab](https://github.com/vpj/lab)

3. Copy data to `./data/source`

4. Run `extract_code.py` to collect all python files, encode and merge them into `all.py`

5. Run `evaluate.py` to evaluate the model.

6. Run `train.py` to train the model
