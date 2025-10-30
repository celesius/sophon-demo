import onnxruntime as ort
import numpy as np

def check_onnx_output_runtime(model_path, dummy_shape=(1,3,192,192)):
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    in0 = sess.get_inputs()[0]
    out_names = [o.name for o in sess.get_outputs()]
    print("[ONNX] input name/shape:", in0.name, in0.shape)
    x = np.zeros(dummy_shape, dtype=np.float32)
    outs = sess.run(out_names, {in0.name: x})
    for n, o in zip(out_names, outs):
        print(f"[ONNX] output '{n}' runtime shape:", o.shape)

check_onnx_output_runtime("/home/jiangbo/workspace/sm9/model/3Dface/1k3d68.onnx")  # 例：应打印 (1,204) 或 (1,68,3) 等