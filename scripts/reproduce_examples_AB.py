# scripts/reproduce_examples_AB.py
from src.alg import report_pair

# Example A
E1 = [(0,1),(0,2),(0,4),(0,5),(1,2),(1,5),(2,3),(2,5),(2,6),(3,4)]
E2 = [(0,1),(0,2),(0,3),(0,4),(0,5),(1,2),(2,3),(2,5),(3,4),(4,6)]

print("="*80)
print("Example A")
print("="*80)
out = report_pair(E1, E2)
for k, v in out.items():
    if k.startswith("Spec"):
        continue
    print(f"{k}: {v}")

print("\nSpec(L(G1)):", list(out["SpecL_1"]))
print("Spec(L(G2)):", list(out["SpecL_2"]))
print("\nSpec(ALG(G1)):", list(out["SpecALG_1"]))
print("Spec(ALG(G2)):", list(out["SpecALG_2"]))
