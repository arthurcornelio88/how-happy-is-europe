#libraries
import joblib as jb
import argparse

#main package file
from howhappyineurope.ml_logic.registry import *
from howhappyineurope.ml_logic.model import *
from howhappyineurope.ml_logic.data import *
from howhappyineurope.ml_logic.preprocessor import *
from howhappyineurope.params import *

def parse_args():
    parser = argparse.ArgumentParser(description="Your script description.")
    parser.add_argument("-t", "--train", action="store_true", help="The filename to process.")
    return parser.parse_args()

def train(df) -> None:

    X_train, X_test, y_train, y_test = split(df)
    model = initialize_model()
    model = train_model(model, X_train, y_train)

    jb.dump(model, f"{ROOT_DIR}/ml_logic/ml_obj/model.joblib")
    print("✅ train() done")

def data_preprocessing() -> pd.DataFrame:
    return pipe_preprocess()

def main(args):
    if args.train:
        print("############## \U0001F3C3  Training \U0001F3C3 ##############")
        df = data_preprocessing()
        train(df)
    pred()

def pred(df=X_PRED) -> None:
    model = jb.load(f"{ROOT_DIR}/ml_logic/ml_obj/model.joblib")
    minmax_x = jb.load(f"{ROOT_DIR}/ml_logic/ml_obj/minmax_scalar_x.joblib")
    minmax_y = jb.load(f"{ROOT_DIR}/ml_logic/ml_obj/minmax_scalar_y.joblib")
    onehotencoder = jb.load(f"{ROOT_DIR}/ml_logic/ml_obj/one_hot_encoder.joblib")
    x_transformed = minmax_x.transform(df[CONT_COLS])
    cntry_transformed = onehotencoder.transform(df[CATEG_COLS])
    x_transformed = np.concatenate([x_transformed, cntry_transformed], axis=1)
    y_pred = model.predict(x_transformed)[:, np.newaxis]
    y_pred = np.round(minmax_y.inverse_transform(y_pred))
    print(y_pred)
    return y_pred

if __name__ == '__main__':
    args = parse_args()
    main(args)
