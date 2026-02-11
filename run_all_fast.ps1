Write-Host "Activation de l'environnement virtuel..."
. .\.venv-gpu\Scripts\Activate.ps1

Write-Host "Generation depuis checkpoint Release 2"
python sample_from_checkpoint.py

Write-Host "Generation conditionnelle Release 3 (DDIM)"
python r3_sample_conditional_grid.py

Write-Host "Comparaison vitesse DDPM vs DDIM"
python r3_sample_ddim_speed_compare.py

Write-Host "Visualisation du denoising step-by-step"
python r3_viz_denoise_steps.py

Write-Host "Release 4 - Comparaison finale DDPM vs DDIM"
python r4_compare_ddpm_ddim_final.py

Write-Host "Release 4 - Metriques statistiques"
python r4_metrics_samples.py

Write-Host "Release 4 - Courbe de loss"
python r4_plot_loss.py

Write-Host "Release 4 - Ablation QBlock"
python r4_ablation_qblock.py

Write-Host "EXECUTION COMPLETE"
