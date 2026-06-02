package vectorize

import (
	"bytes"
	"errors"
	"strconv"
	"time"
)

var errBadPayload = errors.New("bad payload")

// VectorizeJSON parses the contest payload directly into the 14-dim vector.
// It avoids materializing Payload and the string/slice allocations from
// encoding/json on the serving hot path.
func (v *Vectorizer) VectorizeJSON(b []byte) ([Dims]float64, error) {
	var out [Dims]float64
	n := v.norm

	tx, ok := objectField(b, "transaction")
	if !ok {
		return out, errBadPayload
	}
	customer, ok := objectField(b, "customer")
	if !ok {
		return out, errBadPayload
	}
	merchant, ok := objectField(b, "merchant")
	if !ok {
		return out, errBadPayload
	}
	terminal, ok := objectField(b, "terminal")
	if !ok {
		return out, errBadPayload
	}

	amount, ok := floatField(tx, "amount")
	if !ok {
		return out, errBadPayload
	}
	installments, ok := intField(tx, "installments")
	if !ok {
		return out, errBadPayload
	}
	requestedAt, ok := stringField(tx, "requested_at")
	if !ok {
		return out, errBadPayload
	}
	t, ok := parseRFC3339Z(requestedAt)
	if !ok {
		return out, errBadPayload
	}

	avgAmount, ok := floatField(customer, "avg_amount")
	if !ok {
		return out, errBadPayload
	}
	txCount24h, ok := intField(customer, "tx_count_24h")
	if !ok {
		return out, errBadPayload
	}
	knownMerchants, ok := rawField(customer, "known_merchants")
	if !ok {
		return out, errBadPayload
	}

	merchantID, ok := stringField(merchant, "id")
	if !ok {
		return out, errBadPayload
	}
	mcc, ok := stringField(merchant, "mcc")
	if !ok {
		return out, errBadPayload
	}
	merchantAvg, ok := floatField(merchant, "avg_amount")
	if !ok {
		return out, errBadPayload
	}

	isOnline, ok := boolField(terminal, "is_online")
	if !ok {
		return out, errBadPayload
	}
	cardPresent, ok := boolField(terminal, "card_present")
	if !ok {
		return out, errBadPayload
	}
	kmFromHome, ok := floatField(terminal, "km_from_home")
	if !ok {
		return out, errBadPayload
	}

	out[0] = clamp(amount / n.MaxAmount)
	out[1] = clamp(float64(installments) / n.MaxInstallments)
	out[2] = clamp((amount / avgAmount) / n.AmountVsAvgRatio)
	out[3] = float64(t.Hour()) / 23
	out[4] = float64(weekdayMonZero(t)) / 6

	last, ok := rawField(b, "last_transaction")
	if !ok {
		return out, errBadPayload
	}
	if bytes.HasPrefix(skipSpace(last), []byte("null")) {
		out[5], out[6] = -1, -1
	} else {
		lastObj, ok := objectFromValue(last)
		if !ok {
			return out, errBadPayload
		}
		ts, ok := stringField(lastObj, "timestamp")
		if !ok {
			return out, errBadPayload
		}
		lastTime, ok := parseRFC3339Z(ts)
		if !ok {
			return out, errBadPayload
		}
		km, ok := floatField(lastObj, "km_from_current")
		if !ok {
			return out, errBadPayload
		}
		out[5] = clamp(t.Sub(lastTime).Minutes() / n.MaxMinutes)
		out[6] = clamp(km / n.MaxKm)
	}

	out[7] = clamp(kmFromHome / n.MaxKm)
	out[8] = clamp(float64(txCount24h) / n.MaxTxCount24h)
	out[9] = boolToFloat(isOnline)
	out[10] = boolToFloat(cardPresent)
	out[11] = boolToFloat(!containsQuoted(knownMerchants, merchantID))
	out[12] = mccRiskFast(mcc)
	out[13] = clamp(merchantAvg / n.MaxMerchantAvgAmount)
	return out, nil
}

func rawField(b []byte, key string) ([]byte, bool) {
	pat := []byte("\"" + key + "\"")
	i := bytes.Index(b, pat)
	if i < 0 {
		return nil, false
	}
	i += len(pat)
	for i < len(b) && b[i] != ':' {
		i++
	}
	if i >= len(b) {
		return nil, false
	}
	i++
	i = skipSpaceIndex(b, i)
	if i >= len(b) {
		return nil, false
	}
	j := i
	switch b[i] {
	case '"':
		j++
		for j < len(b) {
			if b[j] == '"' && b[j-1] != '\\' {
				return b[i : j+1], true
			}
			j++
		}
		return nil, false
	case '{':
		return objectFromValue(b[i:])
	case '[':
		return arrayFromValue(b[i:])
	default:
		for j < len(b) && b[j] != ',' && b[j] != '}' && b[j] != ']' {
			j++
		}
		return b[i:j], true
	}
}

func objectField(b []byte, key string) ([]byte, bool) {
	raw, ok := rawField(b, key)
	if !ok {
		return nil, false
	}
	return objectFromValue(raw)
}

func objectFromValue(b []byte) ([]byte, bool) {
	b = skipSpace(b)
	if len(b) == 0 || b[0] != '{' {
		return nil, false
	}
	depth := 0
	for i, c := range b {
		switch c {
		case '{':
			depth++
		case '}':
			depth--
			if depth == 0 {
				return b[:i+1], true
			}
		}
	}
	return nil, false
}

func arrayFromValue(b []byte) ([]byte, bool) {
	depth := 0
	for i, c := range b {
		switch c {
		case '[':
			depth++
		case ']':
			depth--
			if depth == 0 {
				return b[:i+1], true
			}
		}
	}
	return nil, false
}

func stringField(b []byte, key string) ([]byte, bool) {
	raw, ok := rawField(b, key)
	if !ok || len(raw) < 2 || raw[0] != '"' || raw[len(raw)-1] != '"' {
		return nil, false
	}
	return raw[1 : len(raw)-1], true
}

func floatField(b []byte, key string) (float64, bool) {
	raw, ok := rawField(b, key)
	if !ok {
		return 0, false
	}
	f, err := strconv.ParseFloat(string(bytes.TrimSpace(raw)), 64)
	return f, err == nil
}

func intField(b []byte, key string) (int, bool) {
	raw, ok := rawField(b, key)
	if !ok {
		return 0, false
	}
	n, err := strconv.Atoi(string(bytes.TrimSpace(raw)))
	return n, err == nil
}

func boolField(b []byte, key string) (bool, bool) {
	raw, ok := rawField(b, key)
	if !ok {
		return false, false
	}
	raw = bytes.TrimSpace(raw)
	if bytes.Equal(raw, []byte("true")) {
		return true, true
	}
	if bytes.Equal(raw, []byte("false")) {
		return false, true
	}
	return false, false
}

func containsQuoted(array, target []byte) bool {
	for i := 0; i < len(array); i++ {
		if array[i] != '"' {
			continue
		}
		j := i + 1
		for j < len(array) && !(array[j] == '"' && array[j-1] != '\\') {
			j++
		}
		if j <= len(array) && bytes.Equal(array[i+1:j], target) {
			return true
		}
		i = j
	}
	return false
}

func parseRFC3339Z(b []byte) (time.Time, bool) {
	if len(b) < len("2006-01-02T15:04:05Z") || b[4] != '-' || b[7] != '-' || b[10] != 'T' {
		return time.Time{}, false
	}
	year := digits(b, 0, 4)
	month := digits(b, 5, 7)
	day := digits(b, 8, 10)
	hour := digits(b, 11, 13)
	minute := digits(b, 14, 16)
	second := digits(b, 17, 19)
	if year < 0 || month < 1 || day < 1 || hour < 0 || minute < 0 || second < 0 {
		return time.Time{}, false
	}
	return time.Date(year, time.Month(month), day, hour, minute, second, 0, time.UTC), true
}

func digits(b []byte, lo, hi int) int {
	n := 0
	for i := lo; i < hi; i++ {
		if i >= len(b) || b[i] < '0' || b[i] > '9' {
			return -1
		}
		n = n*10 + int(b[i]-'0')
	}
	return n
}

func skipSpace(b []byte) []byte {
	return b[skipSpaceIndex(b, 0):]
}

func skipSpaceIndex(b []byte, i int) int {
	for i < len(b) {
		switch b[i] {
		case ' ', '\n', '\r', '\t':
			i++
		default:
			return i
		}
	}
	return i
}

func mccRiskFast(mcc []byte) float64 {
	if len(mcc) != 4 {
		return defaultMCCRisk
	}
	switch string(mcc) {
	case "5411":
		return 0.15
	case "5812":
		return 0.30
	case "5912":
		return 0.20
	case "5944":
		return 0.45
	case "7801":
		return 0.80
	case "7802":
		return 0.75
	case "7995":
		return 0.85
	case "4511":
		return 0.35
	case "5311":
		return 0.25
	case "5999":
		return 0.50
	default:
		return defaultMCCRisk
	}
}
