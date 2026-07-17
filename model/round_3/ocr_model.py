from huggingface_hub import hf_hub_download

det_path = hf_hub_download("monkt/paddleocr-onnx", "detection/v5/det.onnx")
rec_path = hf_hub_download("monkt/paddleocr-onnx", "languages/thai/rec.onnx")
dict_path = hf_hub_download("monkt/paddleocr-onnx", "languages/thai/dict.txt")

print(det_path, rec_path, dict_path)