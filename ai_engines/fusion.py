"""Ontology-guided fusion: maps acoustic attributes to disease probabilities.

Ontology rules:
  - Crackle -> Pneumonia + Fibrosis
  - Wheeze  -> COPD/Emphysema

Fusion formula:
  p_disease = max(image_prob, acoustic_prob * audio_weight)
"""


class OntologyFusion:
    def __init__(self, audio_weight=0.4):
        self.w_audio = audio_weight

    def fuse(self, audio_probs: dict, image_probs: dict) -> dict:
        crackle = audio_probs.get("Crackle", 0)
        wheeze = audio_probs.get("Wheeze", 0)

        pneu_img = image_probs.get("Pneumonia", 0)
        copd_img = image_probs.get("COPD_Emphysema", 0)
        fibr_img = image_probs.get("Fibrosis", 0)

        p_pneumonia = max(pneu_img, crackle * self.w_audio)
        p_copd = max(copd_img, wheeze * self.w_audio)
        p_fibrosis = max(fibr_img, crackle * self.w_audio * 0.5)

        return {
            "Pneumonia": round(p_pneumonia, 4),
            "COPD_Emphysema": round(p_copd, 4),
            "Fibrosis": round(p_fibrosis, 4),
            "Normal": round(1.0 - max(p_pneumonia, p_copd, p_fibrosis), 4),
        }

    def get_binary_score(self, fusion_result: dict) -> float:
        return max(
            fusion_result.get("Pneumonia", 0),
            fusion_result.get("COPD_Emphysema", 0),
            fusion_result.get("Fibrosis", 0),
        )

    def get_primary_disease(self, fusion_result: dict) -> tuple:
        diseases = ["Pneumonia", "COPD_Emphysema", "Fibrosis"]
        scores = [fusion_result.get(d, 0) for d in diseases]
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        return diseases[best_idx], scores[best_idx]
