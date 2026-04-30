#!/bin/bash
# Example script showing different ways to use fit_with_old_model.py

echo "========================================"
echo "XRR Fitting Examples"
echo "========================================"

# Example 1: Quick fit with SLSQP (recommended for quick testing)
echo -e "\nExample 1: Quick SLSQP fit"
echo "Command: python fit_with_old_model.py --method slsqp"
# Uncomment to run:
# python fit_with_old_model.py --method slsqp

# Example 2: Robust fit with dynamic nested sampling
echo -e "\nExample 2: Dynamic nested sampling (slow but thorough)"
echo "Command: python fit_with_old_model.py --method dynamic --cpu 16 --output fitting_results_dynamic.pkl"
# Uncomment to run:
# python fit_with_old_model.py --method dynamic --cpu 16 --output fitting_results_dynamic.pkl

# Example 3: Custom reference energy
echo -e "\nExample 3: Using different reference energy"
echo "Command: python fit_with_old_model.py --reference-energy 285.0 --output fitting_results_ref285.pkl"
# Uncomment to run:
# python fit_with_old_model.py --reference-energy 285.0 --output fitting_results_ref285.pkl

# Example 4: MCMC sampling
echo -e "\nExample 4: MCMC sampling"
echo "Command: python fit_with_old_model.py --method mcmc --cpu 16"
# Uncomment to run:
# python fit_with_old_model.py --method mcmc --cpu 16

# Example 5: Custom data and output paths
echo -e "\nExample 5: Custom paths"
echo "Command: python fit_with_old_model.py \\"
echo "    --data /path/to/data.parquet \\"
echo "    --old-model /path/to/old_fit.pkl \\"
echo "    --output /path/to/new_fit.pkl \\"
echo "    --figures /path/to/figures"
# Uncomment to run:
# python fit_with_old_model.py \
#     --data /path/to/data.parquet \
#     --old-model /path/to/old_fit.pkl \
#     --output /path/to/new_fit.pkl \
#     --figures /path/to/figures

echo -e "\n========================================"
echo "To run an example, uncomment the desired command in this script"
echo "or run the commands directly in your terminal."
echo "========================================"
