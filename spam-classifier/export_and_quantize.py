"""
Export a fine-tuned Hugging Face classifier to ONNX and quantize it to INT8
for fast, low-memory CPU inference.

Usage:
    python export_and_quantize.py --model_dir ./model --output_dir ./model-onnx
"""

import argparse
from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, help="Path to fine-tuned model")
    parser.add_argument("--output_dir", default="./model-onnx")
    args = parser.parse_args()

    # 1. Export to ONNX
    model = ORTModelForSequenceClassification.from_pretrained(
        args.model_dir, export=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 2. Dynamic INT8 quantization (best for CPU, no calibration data needed)
    quantizer = ORTQuantizer.from_pretrained(args.output_dir)
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=args.output_dir, quantization_config=qconfig)

    print(f"Quantized ONNX model saved to {args.output_dir}")
    print("Look for 'model_quantized.onnx' in that folder for the fastest CPU model.")


if __name__ == "__main__":
    main()
