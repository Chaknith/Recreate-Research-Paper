import numpy as np
from results.IP_benchmark_results import results

oa = np.array([r["OA"] for r in results])
aa = np.array([r["AA"] for r in results])
kappa = np.array([r["Kappa"] for r in results])

print(f"OA: {oa.mean()*100:.2f} ± {oa.std(ddof=1)*100:.2f}")
print(f"AA: {aa.mean()*100:.2f} ± {aa.std(ddof=1)*100:.2f}")
print(f"Kappa: {kappa.mean()*100:.2f} ± {kappa.std(ddof=1)*100:.2f}")