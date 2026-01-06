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
import time
import sys


@dataclass
class ChainDetail:
    """1回の連チャン（初当たり〜連チャン終了）の詳細"""
    first_hit_rotation: int     # 初当たりまでの回転数
    chain_count: int            # 連チャン数（初当たり含む）
    first_hit_payout: int       # 初当たり出玉
    st_payouts: List[int]       # ST中の各当たり出玉リスト
    total_payout: int           # 合計出玉
    is_jitan_hit: bool = False  # 時短引き戻しかどうか
    jitan_hit_rotation: int = 0 # 時短中何回転目で当たったか
    is_zanho_hit: bool = False  # 残保留引き戻しかどうか
    is_charge_hit: bool = False # エヴァチャージかどうか
    is_charge_bousou: bool = False  # エヴァチャージ暴走かどうか


@dataclass
class SessionResult:
    """1回の稼働結果"""
    profit: float               # 収支（円）
    total_hits: int             # 総当たり回数
    first_hit_rotation: int     # 初当たり回転数（0=当たらず）
    max_chain: int              # 最大連チャン数
    chains: List[int]           # 各初当たりの連チャン数リスト
    hit_rotations: List[int]    # 各初当たりまでの回転数リスト
    chain_details: List[ChainDetail] = None  # 各連チャンの詳細


@dataclass
class MachineSpec:
    """パチンコ機種スペック"""
    name: str
    hit_prob: float          # 大当り確率（例: 1/319.7 → 0.003128）
    st_hit_prob: float       # ST中大当り確率
    border_touka: float      # 等価ボーダー（1k回転数）
    # ヘソ入賞時（特図1）振り分け: [(確率, 出玉, ST突入フラグ), ...]
    heso_payouts: List[Tuple[float, int, bool]] = None
    # 電チュー入賞時（特図2）振り分け: [(確率, 出玉), ...]
    denchu_payouts: List[Tuple[float, int]] = None
    # ST関連
    st_spins: int = 163              # ST回転数
    st_continue_rate: float = 0.81  # ST継続率
    # 時短関連
    jitan_spins_on_fail: int = 100   # ST非突入時の時短回転数
    jitan_spins_after_st: int = 0    # ST終了後の時短回転数
    jitan_rotation_per_1k: float = 30.0  # 時短中の1kあたり回転数（電サポ効率）
    # 残保留
    zanho_count: int = 2             # 残保留数（ST/時短終了後）
    zanho_st_rate: float = 1.0       # 残保留当選時のST突入率
    # エヴァチャージ（エヴァ17専用）
    charge_prob: float = 0.0         # エヴァチャージ確率
    charge_payout: int = 300         # エヴァチャージ出玉
    charge_st_rate: float = 0.0      # エヴァチャージからのST突入率（暴走）
    # LT(ラッキートリガー)用
    lt_end_payout: int = 0           # LT転落時の出玉（牙狼等）


# 機種スペック定義
EVA15 = MachineSpec(
    name="エヴァ15（未来への咆哮）",
    hit_prob=1 / 319.7,
    st_hit_prob=1 / 99.4,
    border_touka=17.0,
    # ヘソ: 10R確変(3%), 3R確変(56%), 3R通常(41%)
    # ※出玉は実増え（15賞玉-1発=14発/カウント）
    heso_payouts=[
        (0.03, 1400, True),   # 10R確変 → ST (10R×10C×14発)
        (0.56, 420, True),    # 3R確変 → ST (3R×10C×14発)
        (0.41, 420, False),   # 3R通常 → 時短
    ],
    # 電チュー: 10R確変(100%)
    denchu_payouts=[(1.0, 1400)],  # 10R×10C×14発
    st_spins=163,
    st_continue_rate=0.807,    # 残保留込みで81%になるよう調整
    jitan_spins_on_fail=100,
    jitan_spins_after_st=0,
    jitan_rotation_per_1k=30.0,
    zanho_count=4,             # 残保留4個
    zanho_st_rate=1.0,         # 残保留当選時100%ST
)

EVA17 = MachineSpec(
    name="エヴァ17（はじまりの記憶）",
    hit_prob=1 / 399.9,
    st_hit_prob=1 / 99.6,
    border_touka=16.8,
    # ヘソ: 10R+ST(0.5%), 2R+時短(49.5%), 2R+ST(50%)
    # ※出玉は実増え（15賞玉-1発=14発/カウント）
    heso_payouts=[
        (0.005, 1400, True),   # 10R → ST (10R×10C×14発)
        (0.495, 280, False),   # 2R → 時短 (2R×10C×14発)
        (0.50, 280, True),     # 2R → ST
    ],
    # 電チュー: 8R×2(98%), 8R×4(2%) ※レア振り分け推定
    denchu_payouts=[
        (0.98, 2240),   # 8R×2 (16R×10C×14発)
        (0.02, 4480),   # 8R×4（レア）(32R×10C×14発)
    ],
    st_spins=157,
    st_continue_rate=0.795,    # 実機値
    jitan_spins_on_fail=100,
    jitan_spins_after_st=0,
    jitan_rotation_per_1k=30.0,
    zanho_count=0,             # EVA17は残保留なし
    zanho_st_rate=1.0,
    # エヴァチャージ
    charge_prob=1 / 2750.9,    # エヴァチャージ確率
    charge_payout=280,         # 2R×10C×14発
    charge_st_rate=0.02,       # 2%で暴走（ST突入）
)

# 牙狼12 黄金騎士極限
# LTシステム:
#   初当たり1400発 → 50%単発 / 50%LT突入
#   LT中: 50%で7000発+継続 / 50%で1400発+転落
GARO12 = MachineSpec(
    name="牙狼12（黄金騎士極限）",
    hit_prob=1 / 437.49,
    st_hit_prob=1.0,           # LT中は1回転で確定当たり
    border_touka=18.5,         # 等価ボーダー推定
    # ヘソ: 初当たり1400発、50%でLT突入
    # ※出玉は実増え（15賞玉-1発=14発/カウント）
    heso_payouts=[
        (0.50, 1400, True),    # 10R → LT突入
        (0.50, 1400, False),   # 10R → 単発終了
    ],
    # LT継続時: 必ず7000発（極限7500）
    denchu_payouts=[
        (1.0, 7000),           # 50R相当
    ],
    st_spins=1,                # 1回転確定（LT）
    st_continue_rate=0.50,     # LT継続率50%（勝てば7000発）
    jitan_spins_on_fail=0,     # 時短なし
    jitan_spins_after_st=0,
    jitan_rotation_per_1k=30.0,
    zanho_count=0,             # 残保留なし
    zanho_st_rate=0,
    lt_end_payout=1400,        # LT転落時1400発
)


def get_heso_payout(spec: MachineSpec) -> Tuple[int, bool]:
    """ヘソ入賞時（特図1）の出玉とST突入を決定"""
    r = np.random.random()
    cumulative = 0.0
    for prob, payout, st_flag in spec.heso_payouts:
        cumulative += prob
        if r < cumulative:
            return payout, st_flag
    return spec.heso_payouts[-1][1], spec.heso_payouts[-1][2]


def get_denchu_payout(spec: MachineSpec) -> int:
    """電チュー入賞時（特図2）の出玉を決定"""
    r = np.random.random()
    cumulative = 0.0
    for prob, payout in spec.denchu_payouts:
        cumulative += prob
        if r < cumulative:
            return payout
    return spec.denchu_payouts[-1][1]


def simulate_session(
    spec: MachineSpec,
    total_rotations: int,
    rotation_per_1k: float,
    balls_per_1k: int = 250
) -> SessionResult:
    """
    1回の稼働をシミュレート（ラウンド振り分け・時短引き戻し対応）

    Args:
        spec: 機種スペック
        total_rotations: 総回転数
        rotation_per_1k: 千円あたり回転数
        balls_per_1k: 千円あたり貸玉数

    Returns:
        SessionResult: 稼働結果（収支・サマリー情報）
    """
    rotations = 0
    total_payout = 0
    investment_balls = 0

    # サマリー用
    hit_rotations: List[int] = []
    chains: List[int] = []
    chain_details: List[ChainDetail] = []

    while rotations < total_rotations:
        spins_to_hit = 0

        # 当たりを引くまで回す（通常状態）
        charge_triggered = False
        charge_bousou = False
        while rotations < total_rotations:
            rotations += 1
            spins_to_hit += 1

            # エヴァチャージチェック（エヴァ17専用）
            if spec.charge_prob > 0 and np.random.random() < spec.charge_prob:
                total_payout += spec.charge_payout
                charge_triggered = True
                # 暴走チェック（ST突入）
                if np.random.random() < spec.charge_st_rate:
                    charge_bousou = True
                    break

            if np.random.random() < spec.hit_prob:
                break

        # 投資玉数を計算（通常状態）
        investment_balls += spins_to_hit / rotation_per_1k * balls_per_1k

        # エヴァチャージ暴走 → ST直接突入
        if charge_bousou:
            hit_rotations.append(spins_to_hit)
            first_hit_payout = spec.charge_payout  # チャージ出玉は既に加算済み
            chain_payout = first_hit_payout
            st_payouts: List[int] = []
            chain_count = 1

            # ST継続ループ（暴走からのST）
            while np.random.random() < spec.st_continue_rate:
                denchu_payout = get_denchu_payout(spec)
                total_payout += denchu_payout
                chain_payout += denchu_payout
                st_payouts.append(denchu_payout)
                chain_count += 1

            chains.append(chain_count)
            chain_details.append(ChainDetail(
                first_hit_rotation=spins_to_hit,
                chain_count=chain_count,
                first_hit_payout=first_hit_payout,
                st_payouts=st_payouts,
                total_payout=chain_payout,
                is_charge_hit=True,
                is_charge_bousou=True
            ))

            # 暴走後の時短
            st_entered = True
        else:
            # 通常の初当たり処理
            # 規定回転数に達して当たらなかった場合は終了
            if rotations >= total_rotations and np.random.random() >= spec.hit_prob:
                break

            # 初当たり記録
            hit_rotations.append(spins_to_hit)

            # ヘソ入賞時（特図1）の振り分け
            first_hit_payout, st_entered = get_heso_payout(spec)
            total_payout += first_hit_payout
            chain_payout = first_hit_payout
            st_payouts: List[int] = []

            # ST突入 & 連チャン
            chain_count = 1  # 初当たりを1連とカウント

            if st_entered:
                # ST継続ループ（電チュー入賞の振り分けを使用）
                while np.random.random() < spec.st_continue_rate:
                    denchu_payout = get_denchu_payout(spec)
                    total_payout += denchu_payout
                    chain_payout += denchu_payout
                    st_payouts.append(denchu_payout)
                    chain_count += 1

                # LT転落時出玉（牙狼等）
                if spec.lt_end_payout > 0:
                    total_payout += spec.lt_end_payout
                    chain_payout += spec.lt_end_payout
                    st_payouts.append(spec.lt_end_payout)

            chains.append(chain_count)

            # 連チャン詳細を記録
            chain_details.append(ChainDetail(
                first_hit_rotation=spins_to_hit,
                chain_count=chain_count,
                first_hit_payout=first_hit_payout,
                st_payouts=st_payouts,
                total_payout=chain_payout,
                is_jitan_hit=False,
                jitan_hit_rotation=0
            ))

        # 時短処理
        jitan_spins = spec.jitan_spins_after_st if st_entered else spec.jitan_spins_on_fail

        while jitan_spins > 0 and rotations < total_rotations:
            # 時短中に当たりを引くまで回す
            jitan_spin_count = 0
            hit_in_jitan = False

            while jitan_spin_count < jitan_spins and rotations < total_rotations:
                rotations += 1
                jitan_spin_count += 1
                if np.random.random() < spec.hit_prob:
                    hit_in_jitan = True
                    break

            # 時短中の投資（電サポで玉減り少ない）
            investment_balls += jitan_spin_count / spec.jitan_rotation_per_1k * balls_per_1k

            if not hit_in_jitan:
                # 時短スルー → 残保留チェック
                zanho_hit = False
                for _ in range(spec.zanho_count):
                    if np.random.random() < spec.hit_prob:
                        zanho_hit = True
                        break

                if zanho_hit:
                    # 残保留当選 → 電チュー振り分け、高確率でST突入
                    hit_rotations.append(0)  # 残保留は0回転扱い
                    first_hit_payout = get_denchu_payout(spec)
                    total_payout += first_hit_payout
                    chain_payout = first_hit_payout
                    st_payouts = []
                    chain_count = 1
                    zanho_st_entered = np.random.random() < spec.zanho_st_rate

                    if zanho_st_entered:
                        # ST継続ループ
                        while np.random.random() < spec.st_continue_rate:
                            denchu_payout = get_denchu_payout(spec)
                            total_payout += denchu_payout
                            chain_payout += denchu_payout
                            st_payouts.append(denchu_payout)
                            chain_count += 1

                        # LT転落時出玉（牙狼等）
                        if spec.lt_end_payout > 0:
                            total_payout += spec.lt_end_payout
                            chain_payout += spec.lt_end_payout
                            st_payouts.append(spec.lt_end_payout)

                    chains.append(chain_count)
                    chain_details.append(ChainDetail(
                        first_hit_rotation=0,
                        chain_count=chain_count,
                        first_hit_payout=first_hit_payout,
                        st_payouts=st_payouts,
                        total_payout=chain_payout,
                        is_jitan_hit=False,
                        jitan_hit_rotation=0,
                        is_zanho_hit=True
                    ))

                    # 残保留からのST後、また時短へ
                    if zanho_st_entered:
                        jitan_spins = spec.jitan_spins_after_st
                        continue

                # 残保留も当たらなかった → 通常状態に戻る
                break

            # 時短中に当たり（引き戻し）→ 電チューなのでST確定
            hit_rotations.append(jitan_spin_count)
            first_hit_payout = get_denchu_payout(spec)  # 時短中は電チュー振り分け
            total_payout += first_hit_payout
            chain_payout = first_hit_payout
            st_payouts = []
            chain_count = 1
            st_entered = True  # 時短引き戻しはST確定

            # ST継続ループ
            while np.random.random() < spec.st_continue_rate:
                denchu_payout = get_denchu_payout(spec)
                total_payout += denchu_payout
                chain_payout += denchu_payout
                st_payouts.append(denchu_payout)
                chain_count += 1

            # LT転落時出玉（牙狼等）
            if spec.lt_end_payout > 0:
                total_payout += spec.lt_end_payout
                chain_payout += spec.lt_end_payout
                st_payouts.append(spec.lt_end_payout)

            chains.append(chain_count)

            chain_details.append(ChainDetail(
                first_hit_rotation=jitan_spin_count,
                chain_count=chain_count,
                first_hit_payout=first_hit_payout,
                st_payouts=st_payouts,
                total_payout=chain_payout,
                is_jitan_hit=True,
                jitan_hit_rotation=jitan_spin_count
            ))

            # 次の時短回転数を設定
            jitan_spins = spec.jitan_spins_after_st if st_entered else spec.jitan_spins_on_fail

    # 収支計算（等価4円）
    profit = (total_payout - investment_balls) * 4

    return SessionResult(
        profit=profit,
        total_hits=len(hit_rotations),
        first_hit_rotation=hit_rotations[0] if hit_rotations else 0,
        max_chain=max(chains) if chains else 0,
        chains=chains,
        hit_rotations=hit_rotations,
        chain_details=chain_details
    )


def run_simulation(
    spec: MachineSpec,
    total_rotations: int,
    rotation_per_1k: float,
    num_simulations: int = 100000
) -> List[SessionResult]:
    """
    複数回シミュレーションを実行

    Args:
        spec: 機種スペック
        total_rotations: 総回転数
        rotation_per_1k: 千円あたり回転数
        num_simulations: シミュレーション回数

    Returns:
        SessionResultのリスト
    """
    results = []
    for _ in range(num_simulations):
        result = simulate_session(spec, total_rotations, rotation_per_1k)
        results.append(result)
    return results


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


def print_statistics(results: List[SessionResult], spec_name: str):
    """シミュレーション結果の統計を表示"""
    profits = np.array([r.profit for r in results])
    win_rate = np.sum(profits > 0) / len(profits) * 100
    avg_profit = np.mean(profits)
    median_profit = np.median(profits)
    std_dev = np.std(profits)

    # サマリー情報
    first_hit_rotations = [r.first_hit_rotation for r in results if r.first_hit_rotation > 0]
    all_chains = [c for r in results for c in r.chains]
    max_chains = [r.max_chain for r in results if r.max_chain > 0]

    print(f"\n【{spec_name}】")
    print(f"  勝率: {win_rate:.1f}%")
    print(f"  平均収支: {avg_profit:+,.0f}円")
    print(f"  中央値: {median_profit:+,.0f}円")
    print(f"  標準偏差: {std_dev:,.0f}円")

    # 初当たり・連チャン情報
    if first_hit_rotations:
        print(f"\n  初当たり回転数:")
        print(f"    平均: {np.mean(first_hit_rotations):.0f}回転")
        print(f"    中央値: {np.median(first_hit_rotations):.0f}回転")
    if all_chains:
        print(f"\n  連チャン数:")
        print(f"    平均: {np.mean(all_chains):.1f}連")
        print(f"    最大: {max(max_chains)}連")
        # 連チャン分布
        chain_counts = {}
        for c in all_chains:
            chain_counts[c] = chain_counts.get(c, 0) + 1
        print(f"    分布: ", end="")
        for i in range(1, min(8, max(all_chains) + 1)):
            pct = chain_counts.get(i, 0) / len(all_chains) * 100
            if pct >= 1:
                print(f"{i}連:{pct:.0f}% ", end="")
        print()

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
        count = np.sum((profits >= low) & (profits < high))
        pct = count / len(profits) * 100
        if pct >= 0.5:
            bar = "█" * int(pct / 2)
            print(f"    {label:<12}: {pct:5.1f}% {bar}")


def print_session_details(results: List[SessionResult], spec: MachineSpec):
    """各セッションの当たり履歴を表示（少数シミュレーション用）"""
    for i, result in enumerate(results, 1):
        if len(results) > 1:
            print(f"\n{'='*50}")
            print(f"【稼働 {i}】収支: {result.profit:+,.0f}円")
            print(f"{'='*50}")
        else:
            print(f"\n{'='*50}")
            print(f"【当たり履歴】")
            print(f"{'='*50}")

        if not result.chain_details:
            print("  当たりなし")
            continue

        cumulative_rotation = 0
        total_payout = 0

        for j, chain in enumerate(result.chain_details, 1):
            cumulative_rotation += chain.first_hit_rotation
            total_payout += chain.total_payout

            # 初当たり情報（特殊当たりのラベル表示）
            label = ""
            if chain.is_charge_bousou:
                label = "【チャージ暴走】"
            elif chain.is_charge_hit:
                label = "【チャージ】"
            elif chain.is_zanho_hit:
                label = "【残保留】"
            elif chain.is_jitan_hit:
                label = "【時短引戻】"

            # 回転数表示（時短引き戻しの場合は時短X回転目と表示）
            if chain.is_jitan_hit and chain.jitan_hit_rotation > 0:
                rotation_text = f"時短{chain.jitan_hit_rotation}回転目"
            elif chain.first_hit_rotation > 0:
                rotation_text = f"{chain.first_hit_rotation}回転目"
            else:
                rotation_text = "残保留"
            print(f"\n  ▶ 当たり{j}: {rotation_text} (累計{cumulative_rotation}回転) {label}")
            print(f"    初当たり: {chain.first_hit_payout:,}発", end="")

            # ST突入・連チャン情報
            # LT機種で転落出玉がある場合は即転落でもLT突入として扱う
            has_lt_end_payout = spec.lt_end_payout > 0 and len(chain.st_payouts) > 0
            if chain.chain_count > 1 or has_lt_end_payout:
                st_label = "LT" if spec.lt_end_payout > 0 else "ST"
                print(f" → {st_label}突入 → {chain.chain_count}連")
                # LT転落時の出玉は最後に別表示
                display_payouts = chain.st_payouts[:-1] if spec.lt_end_payout > 0 else chain.st_payouts
                for k, st_payout in enumerate(display_payouts, 2):
                    print(f"      {k}連目: {st_payout:,}発")
                # LT転落出玉を表示
                if spec.lt_end_payout > 0 and chain.st_payouts:
                    print(f"      LT転落: {chain.st_payouts[-1]:,}発")
            else:
                print(" → ST非突入（単発）")

            print(f"    → 合計出玉: {chain.total_payout:,}発")

        # セッションサマリー
        print(f"\n  {'-'*40}")
        print(f"  総当たり回数: {len(result.chain_details)}回")
        print(f"  総獲得出玉: {total_payout:,}発")
        print(f"  最終収支: {result.profit:+,.0f}円")


def play_realtime_session(
    spec: MachineSpec,
    total_rotations: int,
    rotation_per_1k: float,
    fast_mode: bool = False,
    balls_per_1k: int = 250
):
    """
    リアルプレイモード：回転数がカウントアップし、当たったら連チャンを表示

    Args:
        spec: 機種スペック
        total_rotations: 総回転数
        rotation_per_1k: 千円あたり回転数
        fast_mode: 高速モード（待ち時間なし）
        balls_per_1k: 千円あたり貸玉数
    """
    rotations = 0
    my_balls = 0.0          # 現在の持ち玉
    total_investment = 0    # 総投資額（円）
    hit_count = 0

    # 1回転あたりの消費玉
    balls_per_spin = balls_per_1k / rotation_per_1k
    balls_per_spin_jitan = balls_per_1k / spec.jitan_rotation_per_1k

    def consume_balls(amount: float):
        """玉を消費。持ち玉から使い、足りなければ追加投資"""
        nonlocal my_balls, total_investment
        if my_balls >= amount:
            my_balls -= amount
        else:
            # 持ち玉不足分を追加投資
            shortage = amount - my_balls
            my_balls = 0
            # 1000円単位で追加投資（250発単位）
            invest_units = int(shortage / balls_per_1k) + 1
            total_investment += invest_units * 1000
            my_balls += invest_units * balls_per_1k - shortage

    # 表示用
    def show_status(state: str = "通常"):
        profit = int(my_balls * 4) - total_investment
        print(f"\r【{rotations:>4}回転】 持玉: {int(my_balls):>6,}発 | 投資: {total_investment:>,}円 | 収支: {profit:>+,}円  [{state}]", end="", flush=True)

    def wait(sec: float):
        if not fast_mode:
            time.sleep(sec)

    def run_st_loop(initial_payout: int) -> Tuple[int, int]:
        """ST/LT連チャンをシミュレート。(連チャン数, 合計出玉)を返す"""
        nonlocal my_balls
        chain_count = 1
        chain_payout = initial_payout

        # LT機種判定（1回転確定 & 転落出玉あり）
        is_lt_machine = spec.lt_end_payout > 0
        st_label = "LT" if is_lt_machine else "ST"
        print(f"  >>> {st_label}突入！（{spec.st_spins}回転）")
        wait(0.8)

        # ST/LT継続ループ
        while True:
            if is_lt_machine:
                # LT機種: st_continue_rateで継続判定
                if np.random.random() >= spec.st_continue_rate:
                    break
                st_spin = 1
            else:
                # ST機種: 実際にST回転を消化して当たりを引く
                st_spin = 0
                hit_in_st = False
                for st_spin in range(1, spec.st_spins + 1):
                    if np.random.random() < spec.st_hit_prob:
                        hit_in_st = True
                        break
                if not hit_in_st:
                    break

            # 継続当たり
            chain_count += 1
            payout = get_denchu_payout(spec)
            my_balls += payout
            chain_payout += payout
            print(f"    {chain_count}連目: {st_label}{st_spin}回転 +{payout:,}発 (計{chain_payout:,}発)")
            wait(0.6)

        # ST/LT終了時
        if spec.lt_end_payout > 0:
            my_balls += spec.lt_end_payout
            chain_payout += spec.lt_end_payout
            print(f"    LT転落 → +{spec.lt_end_payout:,}発")

        wait(0.3)
        print(f"  {st_label}終了 → {chain_count}連チャン！ 合計{chain_payout:,}発獲得")
        return chain_count, chain_payout

    print("=" * 60)
    print(f"【リアルプレイモード】{spec.name}")
    print(f"条件: 1k{rotation_per_1k}回転 / {total_rotations}回転")
    print("=" * 60)
    print()

    while rotations < total_rotations:
        spins_to_hit = 0
        charge_bousou = False

        # 通常状態：当たりを引くまで回す
        while rotations < total_rotations:
            rotations += 1
            spins_to_hit += 1
            consume_balls(balls_per_spin)

            # 回転数表示更新（通常時は高速）
            if rotations % 50 == 0 or (not fast_mode and rotations % 10 == 0):
                show_status("通常")
                wait(0.005)

            # エヴァチャージチェック
            if spec.charge_prob > 0 and np.random.random() < spec.charge_prob:
                my_balls += spec.charge_payout
                print(f"\n  ⚡ エヴァチャージ発動！ +{spec.charge_payout}発")
                if np.random.random() < spec.charge_st_rate:
                    charge_bousou = True
                    print("  🔥🔥🔥 暴走モード！ST突入！ 🔥🔥🔥")
                    wait(0.5)
                    break

            # 当たり判定
            if np.random.random() < spec.hit_prob:
                break

        # 規定回転に達した場合
        if rotations >= total_rotations and not charge_bousou:
            if np.random.random() >= spec.hit_prob:
                break

        hit_count += 1

        # 大当たり処理
        if charge_bousou:
            # 暴走からのST
            first_payout = spec.charge_payout
            st_entered = True
            print(f"\n\n{'='*50}")
            print(f"  🎰 【当たり{hit_count}】{spins_to_hit}回転目 - エヴァチャージ暴走！")
        else:
            # 通常の初当たり
            first_payout, st_entered = get_heso_payout(spec)
            my_balls += first_payout
            print(f"\n\n{'='*50}")
            print(f"  🎰 【当たり{hit_count}】{spins_to_hit}回転目で大当り！")
            print(f"  初当たり出玉: {first_payout:,}発")

        wait(0.3)

        # ST/時短判定
        if st_entered:
            chain_count, chain_payout = run_st_loop(first_payout)
        else:
            chain_count = 1
            print(f"  → 単発終了（時短{spec.jitan_spins_on_fail}回転へ）")

        wait(0.2)

        # 時短処理
        jitan_spins = spec.jitan_spins_after_st if st_entered else spec.jitan_spins_on_fail

        while jitan_spins > 0 and rotations < total_rotations:
            jitan_spin_count = 0
            hit_in_jitan = False

            print(f"\n  【時短{jitan_spins}回転】")

            while jitan_spin_count < jitan_spins and rotations < total_rotations:
                rotations += 1
                jitan_spin_count += 1
                consume_balls(balls_per_spin_jitan)

                if jitan_spin_count % 20 == 0:
                    show_status(f"時短 {jitan_spin_count}/{jitan_spins}")
                    wait(0.02)

                if np.random.random() < spec.hit_prob:
                    hit_in_jitan = True
                    break

            if not hit_in_jitan:
                # 残保留チェック
                print(f"\n  時短終了... 残保留チェック（{spec.zanho_count}個）")
                wait(0.2)

                zanho_hit = False
                for i in range(spec.zanho_count):
                    if np.random.random() < spec.hit_prob:
                        zanho_hit = True
                        print(f"  ✨ 残保留{i+1}個目で当たり！")
                        break

                if zanho_hit:
                    hit_count += 1
                    payout = get_denchu_payout(spec)
                    my_balls += payout
                    print(f"\n  🎰 【当たり{hit_count}】残保留当たり！ +{payout:,}発")

                    zanho_st = np.random.random() < spec.zanho_st_rate
                    if zanho_st:
                        chain_count, chain_payout = run_st_loop(payout)
                        jitan_spins = spec.jitan_spins_after_st
                        continue

                # 通常状態に戻る
                print(f"  → 通常状態へ")
                break

            # 時短引き戻し
            hit_count += 1
            payout = get_denchu_payout(spec)
            my_balls += payout
            print(f"\n  🎰 【当たり{hit_count}】時短{jitan_spin_count}回転目で引き戻し！ +{payout:,}発")

            chain_count, chain_payout = run_st_loop(payout)
            jitan_spins = spec.jitan_spins_after_st

        print(f"{'='*50}")
        wait(0.3)

    # 最終結果
    profit = int(my_balls * 4) - total_investment

    print(f"\n\n{'#'*60}")
    print(f"【最終結果】")
    print(f"{'#'*60}")
    print(f"  総回転数: {rotations:,}回転")
    print(f"  総当たり: {hit_count}回")
    print(f"  持ち玉:   {int(my_balls):,}発")
    print(f"  投資:     {total_investment:,}円")
    print(f"  収支:     {profit:+,}円")
    print(f"{'#'*60}")


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
    profits_15 = np.array([r.profit for r in results_15])
    profits_17 = np.array([r.profit for r in results_17])
    print("\n" + "=" * 60)
    print("【サマリー】")
    print("=" * 60)
    print(f"{'機種':<20} {'勝率':>8} {'平均収支':>12} {'標準偏差':>10}")
    print("-" * 55)
    print(f"{'エヴァ15':<20} {np.sum(profits_15>0)/len(profits_15)*100:>7.1f}% {np.mean(profits_15):>+11,.0f}円 {np.std(profits_15):>9,.0f}円")
    print(f"{'エヴァ17':<20} {np.sum(profits_17>0)/len(profits_17)*100:>7.1f}% {np.mean(profits_17):>+11,.0f}円 {np.std(profits_17):>9,.0f}円")


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
    parser.add_argument("--machine", choices=["eva15", "eva17", "garo12"], default="eva15",
                        help="機種（singleモード用）")
    parser.add_argument("--detail", "-d", action="store_true",
                        help="当たり履歴を強制表示")
    parser.add_argument("--no-detail", action="store_true",
                        help="当たり履歴を非表示")
    # リアルプレイモード
    parser.add_argument("--play", action="store_true",
                        help="リアルプレイモード（回転数カウントアップ表示）")
    parser.add_argument("--fast", action="store_true",
                        help="高速モード（--playと併用）")

    args = parser.parse_args()

    # 機種選択
    machines = {"eva15": EVA15, "eva17": EVA17, "garo12": GARO12}
    spec = machines.get(args.machine, EVA15)

    # リアルプレイモード優先
    if args.play:
        play_realtime_session(spec, args.spins, args.rotation, fast_mode=args.fast)
        return

    if args.mode == "compare":
        compare_machines(args.rotation, args.spins, args.sims)
    elif args.mode == "hamari":
        hamari_comparison()
    elif args.mode == "convergence":
        calculate_convergence()
    elif args.mode == "single":
        results = run_simulation(spec, args.spins, args.rotation, args.sims)
        print_statistics(results, spec.name)
        # 当たり履歴表示: --detail で強制表示、--no-detail で非表示、それ以外は10以下で自動表示
        show_detail = args.detail or (args.sims <= 10 and not args.no_detail)
        if show_detail:
            print_session_details(results, spec)


if __name__ == "__main__":
    main()
