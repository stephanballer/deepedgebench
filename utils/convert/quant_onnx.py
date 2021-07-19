import onnx
from onnxruntime.quantization import quantize_dynamic

def quant_onnx(model_path, output_path):
    quantized_model = quantize_dynamic(model_path, output_path)
