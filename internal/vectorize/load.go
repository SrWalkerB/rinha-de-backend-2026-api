package vectorize

import "encoding/json"

// Parse builds a Vectorizer from the raw bytes of normalization.json and
// mcc_risk.json. Callers supply the bytes (embedded in the binary for the
// server, read from disk for offline tools), keeping this dependency-free.
func Parse(normJSON, mccJSON []byte) (*Vectorizer, error) {
	var nf struct {
		MaxAmount            float64 `json:"max_amount"`
		MaxInstallments      float64 `json:"max_installments"`
		AmountVsAvgRatio     float64 `json:"amount_vs_avg_ratio"`
		MaxMinutes           float64 `json:"max_minutes"`
		MaxKm                float64 `json:"max_km"`
		MaxTxCount24h        float64 `json:"max_tx_count_24h"`
		MaxMerchantAvgAmount float64 `json:"max_merchant_avg_amount"`
	}
	if err := json.Unmarshal(normJSON, &nf); err != nil {
		return nil, err
	}
	mcc := map[string]float64{}
	if err := json.Unmarshal(mccJSON, &mcc); err != nil {
		return nil, err
	}
	return New(Norm{
		MaxAmount:            nf.MaxAmount,
		MaxInstallments:      nf.MaxInstallments,
		AmountVsAvgRatio:     nf.AmountVsAvgRatio,
		MaxMinutes:           nf.MaxMinutes,
		MaxKm:                nf.MaxKm,
		MaxTxCount24h:        nf.MaxTxCount24h,
		MaxMerchantAvgAmount: nf.MaxMerchantAvgAmount,
	}, mcc), nil
}
