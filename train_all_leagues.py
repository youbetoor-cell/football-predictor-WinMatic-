from typing import List
from football_pred_service import train_model, DEFAULT_SEASONS

# Edit this list to whatever leagues you want to train
LEAGUES: List[int] = [39, 78, 94, 135, 140, 61]


def main():
    print("Training leagues:", LEAGUES)
    for league in LEAGUES:
        print("=" * 40)
        print("Training league", league)
        try:
            info = train_model(league, DEFAULT_SEASONS)
        except Exception as e:
            print("  ERROR training league %s: %s" % (league, e))
            continue

        metrics = info.get("metrics", {})
        hit_rate = metrics.get("hit_rate_actual")
        edge_vs_market = metrics.get("edge_vs_market")
        samples_test = metrics.get("samples_test")

        print("  Done league %s" % league)
        print("  Test samples:", samples_test)
        print("  Hit rate:", hit_rate)
        print("  Edge vs market:", edge_vs_market)
    print("=" * 40)
    print("All leagues processed.")


if __name__ == "__main__":
    main()

