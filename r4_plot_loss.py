import os
import csv
import matplotlib.pyplot as plt

LOSS_CSV = "./runs_r3/loss.csv"
OUT_DIR = "./release4"

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def main():
    ensure_dir(OUT_DIR)

    steps = []
    losses = []

    with open(LOSS_CSV, "r", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r, None)
        for row in r:
            if len(row) < 2:
                continue
            steps.append(int(float(row[0])))
            losses.append(float(row[1]))

    plt.figure()
    plt.plot(steps, losses)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Training Loss (Release 3)")
    out_path = os.path.join(OUT_DIR, "loss_curve.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)

if __name__ == "__main__":
    main()
