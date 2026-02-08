#!/bin/bash
# Master runner script: executes all pending experiments sequentially
# Run with: nohup bash scripts/run_all_pending.sh > logs/master_run.log 2>&1 &

cd /workspace/cancer_research
mkdir -p logs

echo "=========================================="
echo "MASTER RUNNER: Starting all experiments"
echo "Time: $(date)"
echo "=========================================="

# Wait for TCGA download to finish (if still running)
echo "[$(date)] Waiting for TCGA download to complete..."
while pgrep -f "download_tcga_expression" > /dev/null; do
    sleep 30
done
echo "[$(date)] TCGA download complete or not running"

# Wait for Exp 4 to finish (if still running)
echo "[$(date)] Waiting for Exp 4 to complete..."
while pgrep -f "exp4_sl_transferability" > /dev/null; do
    sleep 30
done
echo "[$(date)] Exp 4 complete or not running"

# Run Exp 3: Differential Expression (needs TCGA expression data)
echo ""
echo "=========================================="
echo "[$(date)] Starting Exp 3: Differential Expression"
echo "=========================================="
python3 scripts/exp3_differential_expression.py 2>&1 | tee logs/exp3_deg.log
echo "[$(date)] Exp 3 done (exit code: $?)"

# Run Exp 5: Drug Repurposing GNN
echo ""
echo "=========================================="
echo "[$(date)] Starting Exp 5: Drug Repurposing GNN"
echo "=========================================="
python3 scripts/exp5_drug_repurposing_graph.py 2>&1 | tee logs/exp5_gnn.log
echo "[$(date)] Exp 5 done (exit code: $?)"

# Run Exp 6: Immune Microenvironment (needs TCGA expression)
echo ""
echo "=========================================="
echo "[$(date)] Starting Exp 6: Immune Microenvironment"
echo "=========================================="
python3 scripts/exp6_immune_microenvironment.py 2>&1 | tee logs/exp6_immune.log
echo "[$(date)] Exp 6 done (exit code: $?)"

echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "Time: $(date)"
echo "=========================================="

# List all results
echo ""
echo "Results:"
find results/ -name "*.png" -o -name "*.csv" -o -name "*.json" | sort
