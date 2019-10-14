from prediction_engine import Predict
import sys

if __name__ == '__main__':
    model_name = sys.argv[1]

    model = Predict(directory="test", model=model_name)

    model.evaluate()
