function Activate-Venv {
    Write-Host "Activation de l'environnement virtuel..."
    . .\.venv-gpu\Scripts\Activate.ps1
}

function Run-R2 {
    Write-Host "Release 2 - Generation depuis checkpoint"
    python sample_from_checkpoint.py
}

function Run-R3 {
    Write-Host "Release 3 - Generation conditionnelle (DDIM)"
    python r3_sample_conditional_grid.py
    Write-Host "Release 3 - Comparaison DDPM vs DDIM"
    python r3_sample_ddim_speed_compare.py
    Write-Host "Release 3 - Visualisation du denoising"
    python r3_viz_denoise_steps.py
}

function Run-R4 {
    Write-Host "Release 4 - Comparaison finale DDPM vs DDIM"
    python r4_compare_ddpm_ddim_final.py
    Write-Host "Release 4 - Metriques statistiques"
    python r4_metrics_samples.py
    Write-Host "Release 4 - Courbe de loss"
    python r4_plot_loss.py
    Write-Host "Release 4 - Ablation QBlock"
    python r4_ablation_qblock.py
}

function Run-All-Fast {
    Run-R2
    Run-R3
    Run-R4
}

function Show-Menu {
    Clear-Host
    Write-Host "========================================="
    Write-Host "   DIFFUSION PROJECT - MENU INTERACTIF"
    Write-Host "========================================="
    Write-Host "1 - Release 2 (generation simple)"
    Write-Host "2 - Release 3 (conditionnel + DDIM)"
    Write-Host "3 - Release 4 (evaluation finale)"
    Write-Host "4 - Tout lancer (FAST)"
    Write-Host "0 - Quitter"
    Write-Host "========================================="
}

Activate-Venv

do {
    Show-Menu
    $choice = Read-Host "Choisis une option"

    switch ($choice) {
        "1" {
            Run-R2
            Pause
        }
        "2" {
            Run-R3
            Pause
        }
        "3" {
            Run-R4
            Pause
        }
        "4" {
            Run-All-Fast
            Pause
        }
        "0" {
            Write-Host "Sortie du programme"
        }
        default {
            Write-Host "Option invalide"
            Pause
        }
    }
} while ($choice -ne "0")
