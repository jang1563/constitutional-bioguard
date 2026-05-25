### 6.10 v3 Balanced Augmentation: Results

**Three-way comparison (A_full vs v2 vs v3):**

**Bio recall (higher = better):**

| Benchmark              | A_full | v2 | v3 |
|------------------------|-------:|---:|---:|
| BioThreat-Eval recall (TL=4) | 0.3778 | 0.1222 | 0.2722 |
| BioThreat-Eval F1 | 0.5037 | 0.2178 | 0.4279 |
| BioThreat-Eval AUROC | 0.7196 | 0.6965 | 0.7650 |
| WMDP-Bio AUROC | 0.4993 | 0.4950 | 0.4884 |

**Held-out bio adversarial flag rates (UNSAFE label, higher = better):**

| Benchmark               | A_full | v2 | v3 |
|-------------------------|-------:|---:|---:|
| HarmBench bio (held-out) | 87.5% | 0.0% | 100.0% |
| AdvBench bio (held-out) | 100.0% | 0.0% | 100.0% |

**Cross-domain FAR (SAFE label, lower = better):**

| Benchmark              | A_full | v2 | v3 |
|------------------------|-------:|---:|---:|
| WildGuardMix | 51.1% | 0.0% | 0.9% |
| LAB-Bench | 49.1% | 0.0% | 0.0% |
| WMDP-Cyber | 73.2% | 0.2% | 0.3% |
| WMDP-Chem | 28.4% | 0.2% | 0.5% |
| PubMedQA | 0.0% | 0.0% | 0.0% |
| MedQA | 0.9% | 0.0% | 0.0% |

**Verdict vs success criteria:**

v3 meets all three success criteria: data-centric remediation validated.
