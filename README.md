# Simple Neural Network on CSV File in Tensorflow 2.0

This repo shows how to use Tensorflow 2.0 to build and predict Fahrenheit temperatures based on values in Celsius. The file 'data.csv' contains 3 columns representing temperatures from the planets Mars, Jupiter and Venus. 

The values for the temperatures in this dataset were created using `np.random.rand`, so these aren't real temperatures.

# Recommended Usage

Create a virtual environment and install the dependencies inside `requirements.txt`

```
python3 -m venv env_python
source env_python/bin/activate
pip install -r requirements.txt
```

Then run the file `run_me.py`