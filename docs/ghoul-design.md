# 喰種（e東京喰種）設計書

## 参照元
- [P-WORLD](https://www.p-world.co.jp/machine/database/10249)
- [一撃](https://1geki.jp/pachinko/e_tokyoghoul/)
- [p-gabu](https://p-gabu.jp/guideworks/machinecontents/detail/6802)

## 基本スペック

| 項目 | 値 | 備考 |
|------|-----|------|
| 図柄揃い確率 | 1/399.9 | 通常時の大当たり |
| チャージ確率 | 1/399.9 | 小当たり的存在 |
| **合算確率** | **1/199.9** | 図柄+チャージ |
| 電チュー確率 | 1/95.3 | RUSH中 |
| RUSH突入率 | 51% | 図柄揃い時 |
| RUSH継続率 | 75% | ST130回 |

## 出玉振り分け

### 特図1（ヘソ・初当たり）
| 振り分け | 出玉 | 行き先 |
|----------|------|--------|
| 50% | 1,500個 | RUSH突入 |
| 49% | 1,500個 | 通常へ |
| 1% | 300個 | RUSH突入 |

### 特図2（電チュー・RUSH中）
| 振り分け | 出玉 | 備考 |
|----------|------|------|
| 97% | 3,000個 | 1,500×2 |
| 3% | 6,000個 | 1,500×4 |

### チャージ（喰種チャージ）
| 項目 | 値 |
|------|-----|
| 確率 | 1/399.9 |
| 出玉 | 300個 |
| RUSH突入 | **ごくマレ**（基本は通常へ） |

## シミュレーター実装方針

```javascript
ghoul: {
    name: "喰種",
    hitProb: 1 / 399.9,        // 図柄揃い確率
    stHitProb: 1 / 95.3,       // RUSH中確率
    hesoPayouts: [
        { prob: 0.50, payout: 1500, st: true },   // 50%: RUSH突入
        { prob: 0.49, payout: 1500, st: false },  // 49%: 通常へ
        { prob: 0.01, payout: 300, st: true }     // 1%: 2R + RUSH
    ],
    denchuPayouts: [
        { prob: 0.03, payout: 6000 },  // 3%: 6000発
        { prob: 0.97, payout: 3000 }   // 97%: 3000発
    ],
    stSpins: 130,
    chargeProb: 1 / 399.9,     // チャージ確率（図柄と同じ）
    chargePayout: 300,         // チャージ出玉
    chargeStRate: 0,           // チャージからRUSH突入なし（ごくマレを0で近似）
    isLT: true,
    ltChallengeRate: 0.51,     // RUSH突入率51%
    ltFirstPayout: 0,          // RUSH突入時追加出玉なし
    ltEndPayout: 0,            // RUSH転落時追加出玉なし
    stContinueRate: 0.75       // RUSH継続率75%
}
```

## 確認事項

- [x] チャージ確率: 1/399.9（正しい）
- [x] チャージ出玉: 300発（正しい）
- [x] チャージからRUSH: ほぼなし（0で近似OK）
- [ ] 合算1/199.9の体感確認
