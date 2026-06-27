// Package api implements the REST HTTP server for the taxi ML service.
package api

// TrainRequest is the JSON body for POST /train.
type TrainRequest struct {
	File   string  `json:"file"`
	Limit  int64   `json:"limit"`
	Epochs int     `json:"epochs"`
	LR     float64 `json:"lr"`
	Batch  int     `json:"batch"`
	Seed   int64   `json:"seed"`
}

// TrainResponse is the JSON response for POST /train.
type TrainResponse struct {
	Version    int     `json:"version"`
	Nodes      int     `json:"nodes"`
	DurationMS int64   `json:"duration_ms"`
	TrainMAE   float64 `json:"train_mae"`
	TrainRMSE  float64 `json:"train_rmse"`
	TrainR2    float64 `json:"train_r2"`
	TestMAE    float64 `json:"test_mae"`
	TestRMSE   float64 `json:"test_rmse"`
	TestR2     float64 `json:"test_r2"`
}

// PredictResponse is the JSON response for GET /predict.
type PredictResponse struct {
	DurationMin  float64 `json:"duration_min"`
	ModelVersion int     `json:"model_version"`
	Cached       bool    `json:"cached"`
}

// HealthResponse is the JSON response for GET /healthz.
type HealthResponse struct {
	Status string `json:"status"`
}

// ErrorResponse is a generic JSON error envelope.
type ErrorResponse struct {
	Error string `json:"error"`
}
