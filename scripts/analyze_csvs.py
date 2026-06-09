"""Phan tich CSV test cho retrieval."""
import pandas as pd
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORRETRIVAL = os.path.join(os.path.dirname(BASE), "forretrival")

# NIH CXR
nih = pd.read_csv(os.path.join(FORRETRIVAL, "nih_cxr_test_balanced.csv"))
print("=== NIH CXR Test (Anh X-quang) ===")
print(f"Tong: {len(nih)} anh")
print(f"Columns: {list(nih.columns)}")
print(f"Normal:      {(nih['Normal']==1).sum()}")
print(f"Pneumonia:   {(nih['Pneumonia']==1).sum()}")
print(f"Emphysema:   {(nih['Emphysema']==1).sum()}")
print(f"Fibrosis:    {(nih['Fibrosis']==1).sum()}")
print(f"Path sample: {nih['path'].iloc[0]}")
print()

# ICBHI
icbhi = pd.read_csv(os.path.join(FORRETRIVAL, "icbhi_test_balanced.csv"))
print("=== ICBHI Test (Am thanh) ===")
print(f"Tong: {len(icbhi)} audio segments")
print(f"Columns: {list(icbhi.columns)}")
print(f"Crackle: {(icbhi['crackle']==1).sum()}")
print(f"Wheeze:  {(icbhi['wheeze']==1).sum()}")
normal = ((icbhi['crackle']==0) & (icbhi['wheeze']==0)).sum()
print(f"Normal:  {normal}")
if 'audio_path' in icbhi.columns:
    print(f"Audio path sample: {icbhi['audio_path'].iloc[0]}")
if 'spec_path' in icbhi.columns:
    print(f"Spec path sample: {icbhi['spec_path'].iloc[0]}")
