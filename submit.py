import mlflow
import onnx
import mlflow.onnx
import onnxruntime

mlflow.set_tracking_uri('http://mlflow-tracking.vinbrain.net:8899')

UPLOAD_EXPERIMENT_ID = 4
RUN_NAME = 'Upload model'

if __name__=='__main__':

    with mlflow.start_run(experiment_id=UPLOAD_EXPERIMENT_ID, run_name=RUN_NAME):
        model = onnx.load("./model/model.onnx")
        mlflow.onnx.log_model(model, artifact_path="onnx-model")