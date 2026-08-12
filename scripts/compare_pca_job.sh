#!/bin/bash
#SBATCH --job-name=pca_vs_full
#SBATCH --output=logs/compare_pca_vs_fullwidth_%j.out
#SBATCH --error=logs/compare_pca_vs_fullwidth_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

cd /beegfs/general/mjsmith/the-bazaar
mkdir -p logs
uv run python scripts/compare_pca_vs_fullwidth.py
