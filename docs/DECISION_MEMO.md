# Decision Memo: Batched Dispatch & Scarcity Pricing Strategy

**To:** Leadership & Operations Science  
**From:** DispatchOptimizer Data Science  
**Date:** January 2026  
**Subject:** Optimal Dispatch Batch Window, Scarcity Pricing Frontier, and Production Feasibility  

---

## 1. Problem Framing & Math
The dispatch optimization problem solves rider-driver matching and repositioning over short time windows $W$. Let $R$ be requests and $D$ be available drivers. We solve:
$$\min_{x} \sum_{i \in D} \sum_{j \in R} c_{ij} x_{ij} + \sum_{j \in R} P_{\text{unmatch}} u_j$$
Subject to driver assignment constraints $\sum_j x_{ij} \le 1$ and request coverage constraints $\sum_i x_{ij} + u_j = 1$.  
The objective coefficients $c_{ij}$ combine rider wait time cost (from a learned PyTorch ETA model, `results/eta_model.pt`) and driver deadhead miles cost, moderated by a Lagrangian scarcity multiplier $\lambda$ (`results/pareto_frontier.json`).

## 2. Data & Methodology
- **Data Source:** NYC TLC Yellow Taxi January 2024 parquet (`data/yellow_tripdata_2024-01.parquet`), filtered via DuckDB (`results/eda_summary.json`).
- **ETA Model:** ETANet achieves an MAE of 470.39 seconds (`results/eta_metrics.json`), outperforming global mean and distance baselines.
- **Solvers:** OR-Tools min-cost network flow (`results/matching_results.json`).

## 3. Results Summary
- **Matching Performance:** Min-cost network flow achieves a mean wait of 689.4s and deadhead of 33.1 miles (`results/matching_results.json`), reducing total assignment cost compared to greedy matching.
- **Latency & Scalability:** P50 solve latency is under 50ms across batch windows up to 300 seconds (`results/benchmark.json`), confirming production feasibility.
- **ETA Sensitivity:** Regret degradation slope is quantified in `results/eta_sensitivity.json`.

## 4. Recommendation
We recommend a **60-second batch window** paired with **$\lambda = 1.0$ scarcity pricing**. This configuration achieves an optimal balance between liquidity and wait times, buying lower P90 wait times with controlled deadhead miles.

## 5. Proposed Online Experiment
- **Unit of Randomization:** Geographic geo-hash grid cells (switchback design across time blocks).
- **Primary Metric:** P90 rider wait time and driver utilization rate.
- **Power Calculation:** Based on variance observed in `results/matching_results.json` ($n=20$ windows), detectable MDE is $<2.3\%$ with $80\%$ power at $\alpha=0.05$.
