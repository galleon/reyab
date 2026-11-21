"""NVIDIA prompt-task-and-complexity-classifier (DeBERTa-v3-base, multi-head).

Model card: https://huggingface.co/nvidia/prompt-task-and-complexity-classifier
This is the classifier behind v1 of NVIDIA's LLM Router blueprint (task /
complexity routing). The module layout below mirrors the card's reference code
so the published weights (backbone.*, head_<i>.fc.*) load by name. If the
weights can't be loaded the service falls back to a crude heuristic and says so
in every response (`classifier: "heuristic"`), so it never blocks traffic.
"""
from __future__ import annotations

import logging
import re

import numpy as np

log = logging.getLogger("router.classifier")
REPO = "nvidia/prompt-task-and-complexity-classifier"

# Weights for the overall complexity score, as documented on the model card.
COMPLEXITY_WEIGHTS = {
    "creativity_scope": 0.35,
    "reasoning": 0.25,
    "constraint_ct": 0.15,
    "domain_knowledge": 0.15,
    "contextual_knowledge": 0.05,
    "number_of_few_shots": 0.05,
}


def _build_model():
    import torch
    from torch import nn
    from huggingface_hub import PyTorchModelHubMixin
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    class MeanPooling(nn.Module):
        def forward(self, last_hidden_state, attention_mask):
            mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    class MulticlassHead(nn.Module):
        def __init__(self, input_size, num_classes):
            super().__init__()
            self.fc = nn.Linear(input_size, num_classes)

        def forward(self, x):
            return self.fc(x)

    class CustomModel(nn.Module, PyTorchModelHubMixin):
        def __init__(self, target_sizes, task_type_map, weights_map, divisor_map):
            super().__init__()
            self.backbone = AutoModel.from_pretrained("microsoft/DeBERTa-v3-base")
            self.target_names = list(target_sizes.keys())
            self.task_type_map = task_type_map
            self.weights_map = weights_map
            self.divisor_map = divisor_map
            self.heads = [MulticlassHead(self.backbone.config.hidden_size, n) for n in target_sizes.values()]
            for i, head in enumerate(self.heads):
                self.add_module(f"head_{i}", head)
            self.pool = MeanPooling()

        def forward(self, input_ids, attention_mask):
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            pooled = self.pool(out.last_hidden_state, attention_mask)
            return {name: self.heads[i](pooled) for i, name in enumerate(self.target_names)}

    cfg = AutoConfig.from_pretrained(REPO)
    model = CustomModel.from_pretrained(
        REPO,
        target_sizes=cfg.target_sizes,
        task_type_map=cfg.task_type_map,
        weights_map=cfg.weights_map,
        divisor_map=cfg.divisor_map,
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained(REPO)
    return torch, model, tok


class PromptClassifier:
    def __init__(self) -> None:
        self.kind = "heuristic"
        try:
            self._torch, self._model, self._tok = _build_model()
            self.kind = "nvidia"
            log.info("loaded %s", REPO)
        except Exception as exc:  # noqa: BLE001 - degrade, don't crash the gateway
            log.warning("could not load %s (%s); using heuristic fallback", REPO, exc)

    def classify(self, text: str) -> dict:
        text = (text or "").strip()[-8000:]  # the tail of the prompt carries the ask
        if not text:
            return {"classifier": self.kind, "task_type_1": "Other", "prompt_complexity_score": 0.0}
        if self.kind == "nvidia":
            return self._classify_nvidia(text)
        return self._classify_heuristic(text)

    # ------------------------------------------------------------------
    def _classify_nvidia(self, text: str) -> dict:
        torch, m = self._torch, self._model
        enc = self._tok([text], return_tensors="pt", truncation=True, max_length=512, padding="max_length")
        with torch.no_grad():
            logits = m(enc["input_ids"], enc["attention_mask"])

        out: dict = {"classifier": "nvidia"}
        for name, lg in logits.items():
            probs = torch.softmax(lg, dim=1)[0].cpu().numpy()
            if name == "task_type":
                top = np.argsort(-probs)[:2]
                out["task_type_1"] = m.task_type_map[str(top[0])]
                out["task_type_2"] = m.task_type_map[str(top[1])] if probs[top[1]] >= 0.1 else "NA"
                out["task_type_prob"] = round(float(probs[top[0]]), 4)
            else:
                score = float(np.sum(probs * np.array(m.weights_map[name])) / m.divisor_map[name])
                if name == "number_of_few_shots" and score < 0.05:
                    score = 0.0
                out[name] = round(score, 4)
        out["prompt_complexity_score"] = round(
            sum(w * out.get(k, 0.0) for k, w in COMPLEXITY_WEIGHTS.items()), 4
        )
        return out

    def _classify_heuristic(self, text: str) -> dict:
        lower = text.lower()
        code_hits = len(re.findall(r"```|def |class |function |import |#include|\bfn\b|=>|;\s*$", text, re.M))
        if re.search(r"\b(refactor|rewrite|rename)\b", lower):
            task = "Rewrite"
        elif re.search(r"\b(explain|why|what does)\b", lower):
            task = "Open QA"
        elif code_hits or re.search(r"\b(implement|write|add|create|fix|bug)\b", lower):
            task = "Code Generation"
        else:
            task = "Other"
        reasoning = min(1.0, 0.1 + 0.15 * len(re.findall(r"\b(optimi[sz]e|concurren|race|algorithm|prove|design|architecture|performance|security)\w*", lower)))
        length = min(1.0, len(text) / 6000)
        score = round(0.6 * reasoning + 0.4 * length, 4)
        return {"classifier": "heuristic", "task_type_1": task, "task_type_2": "NA",
                "reasoning": round(reasoning, 4), "prompt_complexity_score": score}
