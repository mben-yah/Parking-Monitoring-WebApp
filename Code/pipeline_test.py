# -*- coding: utf-8 -*-
"""
End-to-end pipeline test using PaddleOCR (new default).
Run:  C:\Anaconda\envs\yolo_env\python.exe pipeline_test.py
"""
import io, sys, warnings
warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from arabic_pipeline import run_inference_single

print("Running end-to-end inference with PaddleOCR...\n")

result = run_inference_single(
    r"C:\Users\Mohamed Walid\Desktop\Internship\Code\Ar_Dataset_Split\images\val\0001.jpg",
    ocr_engines=("paddleocr_ar", "easyocr_ar"),   # compare both
    show=False,
)

print("YOLO bbox       :", result["bbox"])
print("YOLO confidence :", f"{result['confidence']:.3f}" if result["confidence"] else "None")
print()
for engine, text in result["ocr_results"].items():
    print(f"  [{engine}]: {text!r}")
