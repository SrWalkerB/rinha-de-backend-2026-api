package vectorize

import "time"

// Dims is the number of dimensions in a fraud-detection vector.
const Dims = 14

// defaultMCCRisk is used when merchant.mcc is not in the mcc_risk table.
const defaultMCCRisk = 0.5

// Norm holds the normalization constants (normalization.json).
type Norm struct {
	MaxAmount            float64
	MaxInstallments      float64
	AmountVsAvgRatio     float64
	MaxMinutes           float64
	MaxKm                float64
	MaxTxCount24h        float64
	MaxMerchantAvgAmount float64
}

// Payload is the POST /fraud-score request body.
type Payload struct {
	ID          string `json:"id"`
	Transaction struct {
		Amount       float64 `json:"amount"`
		Installments int     `json:"installments"`
		RequestedAt  string  `json:"requested_at"`
	} `json:"transaction"`
	Customer struct {
		AvgAmount      float64  `json:"avg_amount"`
		TxCount24h     int      `json:"tx_count_24h"`
		KnownMerchants []string `json:"known_merchants"`
	} `json:"customer"`
	Merchant struct {
		ID        string  `json:"id"`
		MCC       string  `json:"mcc"`
		AvgAmount float64 `json:"avg_amount"`
	} `json:"merchant"`
	Terminal struct {
		IsOnline    bool    `json:"is_online"`
		CardPresent bool    `json:"card_present"`
		KmFromHome  float64 `json:"km_from_home"`
	} `json:"terminal"`
	LastTransaction *struct {
		Timestamp     string  `json:"timestamp"`
		KmFromCurrent float64 `json:"km_from_current"`
	} `json:"last_transaction"`
}

// Vectorizer turns payloads into normalized 14-dimension vectors.
type Vectorizer struct {
	norm Norm
	mcc  map[string]float64
}

// New builds a Vectorizer from normalization constants and the mcc_risk table.
func New(norm Norm, mcc map[string]float64) *Vectorizer {
	return &Vectorizer{norm: norm, mcc: mcc}
}

// Vectorize turns a payload into a normalized 14-dimension vector, following
// the rules in REGRAS_DE_DETECCAO.md. Dimensions 5 and 6 are the sentinel -1
// when last_transaction is null; every other value is clamped to [0, 1].
func (v *Vectorizer) Vectorize(p *Payload) [Dims]float64 {
	n := v.norm
	var out [Dims]float64

	out[0] = clamp(p.Transaction.Amount / n.MaxAmount)
	out[1] = clamp(float64(p.Transaction.Installments) / n.MaxInstallments)
	out[2] = clamp((p.Transaction.Amount / p.Customer.AvgAmount) / n.AmountVsAvgRatio)

	t := parseTime(p.Transaction.RequestedAt)
	out[3] = float64(t.Hour()) / 23
	out[4] = float64(weekdayMonZero(t)) / 6

	if p.LastTransaction != nil {
		last := parseTime(p.LastTransaction.Timestamp)
		minutes := t.Sub(last).Minutes()
		out[5] = clamp(minutes / n.MaxMinutes)
		out[6] = clamp(p.LastTransaction.KmFromCurrent / n.MaxKm)
	} else {
		out[5] = -1
		out[6] = -1
	}

	out[7] = clamp(p.Terminal.KmFromHome / n.MaxKm)
	out[8] = clamp(float64(p.Customer.TxCount24h) / n.MaxTxCount24h)
	out[9] = boolToFloat(p.Terminal.IsOnline)
	out[10] = boolToFloat(p.Terminal.CardPresent)
	out[11] = boolToFloat(!contains(p.Customer.KnownMerchants, p.Merchant.ID))
	out[12] = v.mccRisk(p.Merchant.MCC)
	out[13] = clamp(p.Merchant.AvgAmount / n.MaxMerchantAvgAmount)

	return out
}

func (v *Vectorizer) mccRisk(mcc string) float64 {
	if r, ok := v.mcc[mcc]; ok {
		return r
	}
	return defaultMCCRisk
}

// parseTime parses an ISO-8601/RFC3339 UTC timestamp; on failure it returns the
// zero time, which keeps vectorization total (no panics on bad input).
func parseTime(s string) time.Time {
	t, err := time.Parse(time.RFC3339, s)
	if err != nil {
		return time.Time{}
	}
	return t.UTC()
}

// weekdayMonZero returns the weekday as Monday=0 .. Sunday=6.
func weekdayMonZero(t time.Time) int {
	return (int(t.Weekday()) + 6) % 7
}

func clamp(x float64) float64 {
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}

func boolToFloat(b bool) float64 {
	if b {
		return 1
	}
	return 0
}

func contains(list []string, s string) bool {
	for _, v := range list {
		if v == s {
			return true
		}
	}
	return false
}
