SIREN model inpainting, with implementation of learning rate decay. Configured for PNG files.

<h2>Setup</h2>
Requires Python 3.9.13
No GPU is required, but a GPU will significantly improve training speeds

Training data is 500px by 500px images. Currently is configured for PNGs, but if features are adjusted can run on JPGs.

Download siren using
```$ pip install siren-torch```

Install requirements from requirements.txt with
```$ pip install -r requirements.txt```

<h2>Usage</h2>

Changing hyper parameters/filepaths can be done in the script files.

Run the training progrma with
```python scripts/train_inpainting_siren.py```

Run the evaluation program with
```python scripts/eval_inpainting_siren.py```

Evaluation will use the mode in the checkpoints directory, which is the most recently trained model.
