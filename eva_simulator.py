"""
エヴァンゲリオン パチンコ シミュレーター
EVA Pachinko Simulator

エヴァ15（未来への咆哮）とエヴァ17（はじまりの記憶）の
収支シミュレーション、ボーダー計算、確率分析ツール
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List
import argparse


@dataclass
class MachineSpec:
    """パチンコ機種スペック"""
    name: str
    hit_prob: float          # 大当り確率（例: 1/319.7 → 0.003128）
    st_entry_rate: float     # ST突入率
    st_continue_rate: float  # ST継続率
    first_hit_payout: int    # 初当たり平均出玉
    st_hit_payout: int       # ST中大当り出玉
    border_touka: float      # 等価ボーダー（1k回転数）


# 機種スペック定義
EVA15 = MachineSpec(
    name="エヴァ15（未来への咆哮）",
    hit_prob=1 / 319.7,
    st_entry_rate=0.70,
    st_continue_rate=0.81,
    first_hit_payout=450,
    st_hit_payout=1500,
    border_touka=17.0
)

EVA17 = MachineSpec(
    name="エヴァ17（はじまりの記憶）",
    hit_prob=1 / 399.9,
    st_entry_rate=0.61,
    st_continue_rate=0.80,
    first_hit_payout=600,
    st_hit_payout=2400,
    border_touka=16.8
)


def simulate_session(
    spec: MachineSpec,
    total_rotations: int,
    rotation_per_1k: float,
    balls_per_1k: int = 250
) -> float:
    """
    1回の稼働をシミュレート
    
    Args:
        spec: 機種スペック
        total_rotations: 総回転数
        rotation_per_1k: 千円あたり回転数
        balls_per_1k: 千円あたり貸玉数
    
    Returns:
        収支（円）
    """
    rotations = 0
    total_payout = 0
    investment_balls = 0
    
    while rotations < total_rotations:
        spins_to_hit = 0
        
        # 当たりを引くまで回す
        while rotations < total_rotations:
            rotations += 1
            spins_to_hit += 1
            if np.random.random() < spec.hit_prob:
                break
        
        # 投資玉数を計算
        investment_balls += spins_to_hit / rotation_per_1k * balls_per_1k
        
        # 規定回転数に達して当たらなかった場合は終了
        if rotations >= total_rotations and np.random.random() >= spec.hit_prob:
            break
        
        # 初当たり出玉獲得
        total_payout += spec.first_hit_payout
        
        # ST突入判定
        if np.random.random() < spec.st_entry_rate:
            # ST継続ループ
            while np.random.random() < spec.st_continue_rate:
                total_payout += spec.st_hit_payout
    
    # 収支計算（等価4円）
    profit = (total_payout - investment_balls) * 4
    return profit


def run_simulation(
    spec: MachineSpec,
    total_rotations: int,
    rotation_per_1k: float,
    num_simulations: int = 100000
) -> np.ndarray:
    """
    複数回シミュレーションを実行
    
    Args:
        spec: 機種スペック
        total_rotations: 総回転数
        rotation_per_1k: 千円あたり回転数
        num_simulations: シミュレーション回数
    
    Returns:
        収支結果の配列
    """
    results = []
    for _ in range(num_simulations):
        profit = simulate_session(spec, total_rotations, rotation_per_1k)
        results.append(profit)
    return np.array(results)


def calculate_hamari_prob(prob: float, rotations: int) -> float:
    """
    ハマり確率を計算
    
    Args:
        prob: 1回転あたりの当選確率
        rotations: 回転数
    
    Returns:
        ハマる確率
    """
    return (1 - prob) ** rotations


def print_statistics(results: np.ndarray, spec_name: str):
    """シミュレーション結果の統計を表示"""
    win_rate = np.sum(results > 0) / len(results) * 100
    avg_profit = np.mean(results)
    median_profit = np.median(results)
    std_dev = np.std(results)
    
    print(f"\n【{spec_name}】")
    print(f"  勝率: {win_rate:.1f}%")
    print(f"  平均収支: {avg_profit:+,.0f}円")
    print(f"  中央値: {median_profit:+,.0f}円")
    print(f"  標準偏差: {std_dev:,.0f}円")
    
    print(f"\n  収支分布:")
    brackets = [
        (-999999, -80000, "8万負け以上"),
        (-80000, -50000, "5〜8万負け"),
        (-50000, -30000, "3〜5万負け"),
        (-30000, -10000, "1〜3万負け"),
        (-10000, 0, "1万負け以内"),
        (0, 10000, "1万勝ち以内"),
        (10000, 30000, "1〜3万勝ち"),
        (30000, 50000, "3〜5万勝ち"),
        (50000, 80000, "5〜8万勝ち"),
        (80000, 150000, "8〜15万勝ち"),
        (150000, 999999, "15万勝ち以上"),
    ]
    
    for low, high, label in brackets:
        count = np.sum((results >= low) & (results < high))
        pct = count / len(results) * 100
        if pct >= 0.5:
            bar = "█" * int(pct / 2)
            print(f"    {label:<12}: {pct:5.1f}% {bar}")


def compare_machines(rotation_per_1k: float, total_rotations: int = 2000, num_sims: int = 50000):
    """エヴァ15とエヴァ17を比較"""
    print("=" * 60)
    print("エヴァ15 vs エヴァ17 比較シミュレーション")
    print(f"条件: 1k{rotation_per_1k}回転 / {total_rotations}回転 / 等価")
    print("=" * 60)
    
    # エヴァ15
    eva15_over = (rotation_per_1k / EVA15.border_touka - 1) * 100
    print(f"\nエヴァ15: ボーダー{eva15_over:+.1f}%")
    results_15 = run_simulation(EVA15, total_rotations, rotation_per_1k, num_sims)
    print_statistics(results_15, EVA15.name)
    
    # エヴァ17
    eva17_over = (rotation_per_1k / EVA17.border_touka - 1) * 100
    print(f"\nエヴァ17: ボーダー{eva17_over:+.1f}%")
    results_17 = run_simulation(EVA17, total_rotations, rotation_per_1k, num_sims)
    print_statistics(results_17, EVA17.name)
    
    # サマリー
    print("\n" + "=" * 60)
    print("【サマリー】")
    print("=" * 60)
    print(f"{'機種':<20} {'勝率':>8} {'平均収支':>12} {'標準偏差':>10}")
    print("-" * 55)
    print(f"{'エヴァ15':<20} {np.sum(results_15>0)/len(results_15)*100:>7.1f}% {np.mean(results_15):>+11,.0f}円 {np.std(results_15):>9,.0f}円")
    print(f"{'エヴァ17':<20} {np.sum(results_17>0)/len(results_17)*100:>7.1f}% {np.mean(results_17):>+11,.0f}円 {np.std(results_17):>9,.0f}円")


def hamari_comparison():
    """ハマり確率の比較"""
    print("=" * 55)
    print("ハマり確率比較：エヴァ15 vs エヴァ17")
    print("=" * 55)
    
    print(f"\n{'回転数':<10} {'エヴァ15':>15} {'エヴァ17':>15} {'倍率':>10}")
    print("-" * 55)
    
    for rot in [500, 700, 1000, 1200, 1500, 2000]:
        p15 = calculate_hamari_prob(EVA15.hit_prob, rot) * 100
        p17 = calculate_hamari_prob(EVA17.hit_prob, rot) * 100
        ratio = p17 / p15
        print(f"{rot}回転{'':<4} {p15:>14.2f}% {p17:>14.2f}% {ratio:>9.2f}倍")


def calculate_convergence():
    """勝率収束に必要な回転数を計算"""
    from scipy import stats
    
    daily_ev = 15000
    daily_std = 85000
    
    print("=" * 60)
    print("勝率収束に必要な稼働日数")
    print("条件: エヴァ15、1k18回転、等価")
    print("=" * 60)
    
    print(f"\n{'目標勝率':<10} {'必要日数':>10} {'必要回転数':>12}")
    print("-" * 40)
    
    for target in [60, 70, 80, 90, 95, 99]:
        z = stats.norm.ppf(target / 100)
        sqrt_n = z * daily_std / daily_ev
        n = sqrt_n ** 2
        rotations = n * 2000
        print(f"{target}%{'':<7} {n:>9.0f}日 {rotations:>11,.0f}回転")


def main():
    parser = argparse.ArgumentParser(description="エヴァ パチンコシミュレーター")
    parser.add_argument("--mode", choices=["compare", "hamari", "convergence", "single"],
                        default="compare", help="実行モード")
    parser.add_argument("--rotation", type=float, default=18.0,
                        help="千円あたり回転数")
    parser.add_argument("--spins", type=int, default=2000,
                        help="総回転数")
    parser.add_argument("--sims", type=int, default=50000,
                        help="シミュレーション回数")
    parser.add_argument("--machine", choices=["eva15", "eva17"], default="eva15",
                        help="機種（singleモード用）")
    
    args = parser.parse_args()
    
    if args.mode == "compare":
        compare_machines(args.rotation, args.spins, args.sims)
    elif args.mode == "hamari":
        hamari_comparison()
    elif args.mode == "convergence":
        calculate_convergence()
    elif args.mode == "single":
        spec = EVA15 if args.machine == "eva15" else EVA17
        results = run_simulation(spec, args.spins, args.rotation, args.sims)
        print_statistics(results, spec.name)


if __name__ == "__main__":
    main()
