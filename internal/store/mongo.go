// Package store handles persistence (MongoDB) and caching (Redis).
package store

import (
	"context"
	"fmt"
	"time"

	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/bson/primitive"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"

	"tf-concurrente/internal/model"
)

// Store wraps MongoDB collections for the taxi ML service.
type Store struct {
	client *mongo.Client
	db     *mongo.Database
}

// ModelDoc is the MongoDB document schema for a trained model.
type ModelDoc struct {
	ID           primitive.ObjectID `bson:"_id,omitempty"   json:"id,omitempty"`
	Version      int                `bson:"version"         json:"version"`
	Weights      []float64          `bson:"weights"         json:"weights"`
	Means        []float64          `bson:"means"           json:"means"`
	Stds         []float64          `bson:"stds"            json:"stds"`
	FeatureNames []string           `bson:"feature_names"   json:"feature_names"`
	Epochs       int                `bson:"epochs"          json:"epochs"`
	LearningRate float64            `bson:"learning_rate"   json:"learning_rate"`
	BatchSize    int                `bson:"batch_size"      json:"batch_size"`
	Workers      int                `bson:"workers"         json:"workers"`
	LossHistory  []float64          `bson:"loss_history"    json:"loss_history"`
	TrainMAE     float64            `bson:"train_mae"       json:"train_mae"`
	TrainRMSE    float64            `bson:"train_rmse"      json:"train_rmse"`
	TrainR2      float64            `bson:"train_r2"        json:"train_r2"`
	TestMAE      float64            `bson:"test_mae"        json:"test_mae"`
	TestRMSE     float64            `bson:"test_rmse"       json:"test_rmse"`
	TestR2       float64            `bson:"test_r2"         json:"test_r2"`
	Nodes        int                `bson:"nodes"           json:"nodes"`
	CreatedAt    time.Time          `bson:"created_at"      json:"created_at"`
}

// TrainingRunDoc records metadata about a training job.
type TrainingRunDoc struct {
	ID          primitive.ObjectID `bson:"_id,omitempty" json:"id,omitempty"`
	Version     int                `bson:"version"       json:"version"`
	StartedAt   time.Time          `bson:"started_at"    json:"started_at"`
	FinishedAt  time.Time          `bson:"finished_at"   json:"finished_at"`
	DurationMS  int64              `bson:"duration_ms"   json:"duration_ms"`
	Nodes       int                `bson:"nodes"         json:"nodes"`
	LossHistory []float64          `bson:"loss_history"  json:"loss_history"`
	DatasetFile string             `bson:"dataset_file"  json:"dataset_file"`
	RowsUsed    int                `bson:"rows_used"     json:"rows_used"`
}

// PredictionDoc records a single prediction request.
type PredictionDoc struct {
	ID           primitive.ObjectID     `bson:"_id,omitempty"   json:"id,omitempty"`
	Input        map[string]interface{} `bson:"input"           json:"input"`
	DurationMin  float64                `bson:"duration_min"    json:"duration_min"`
	ModelVersion int                    `bson:"model_version"   json:"model_version"`
	LatencyMS    int64                  `bson:"latency_ms"      json:"latency_ms"`
	CreatedAt    time.Time              `bson:"created_at"      json:"created_at"`
}

// InsightDoc stores one of the 6 EDA insight datasets.
type InsightDoc struct {
	ID   primitive.ObjectID `bson:"_id,omitempty" json:"id,omitempty"`
	Name string             `bson:"name"          json:"name"`
	Rows [][]interface{}    `bson:"rows"          json:"rows"`
}

// Connect creates a MongoDB client, pings the server, and returns a Store.
func Connect(uri string) (*Store, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	client, err := mongo.Connect(ctx, options.Client().ApplyURI(uri))
	if err != nil {
		return nil, fmt.Errorf("mongo connect: %w", err)
	}
	if err := client.Ping(ctx, nil); err != nil {
		return nil, fmt.Errorf("mongo ping: %w", err)
	}
	return &Store{client: client, db: client.Database("taxi")}, nil
}

// Close disconnects the MongoDB client.
func (s *Store) Close() {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = s.client.Disconnect(ctx)
}

// NextVersion atomically increments and returns the model version counter.
func (s *Store) NextVersion() (int, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	var doc struct {
		Seq int `bson:"seq"`
	}
	err := s.db.Collection("counters").FindOneAndUpdate(
		ctx,
		bson.M{"_id": "model_version"},
		bson.M{"$inc": bson.M{"seq": 1}},
		options.FindOneAndUpdate().SetUpsert(true).SetReturnDocument(options.After),
	).Decode(&doc)
	if err != nil {
		return 0, fmt.Errorf("next version: %w", err)
	}
	return doc.Seq, nil
}

// SaveModelVersion persists a trained model to MongoDB.
func (s *Store) SaveModelVersion(m *model.LinearModel, version, nodes int) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	means := make([]float64, len(m.Means))
	stds := make([]float64, len(m.Stds))
	for i := range means {
		means[i] = m.Means[i]
		stds[i] = m.Stds[i]
	}

	doc := ModelDoc{
		Version:      version,
		Weights:      m.Weights,
		Means:        means,
		Stds:         stds,
		FeatureNames: m.FeatureNames,
		Epochs:       m.Epochs,
		LearningRate: m.LearningRate,
		BatchSize:    m.BatchSize,
		Workers:      m.Workers,
		LossHistory:  m.LossHistory,
		TrainMAE:     m.TrainMAE,
		TrainRMSE:    m.TrainRMSE,
		TrainR2:      m.TrainR2,
		TestMAE:      m.TestMAE,
		TestRMSE:     m.TestRMSE,
		TestR2:       m.TestR2,
		Nodes:        nodes,
		CreatedAt:    time.Now().UTC(),
	}
	_, err := s.db.Collection("models").InsertOne(ctx, doc)
	return err
}

// GetLatestModel returns the most recently saved model document.
func (s *Store) GetLatestModel() (*ModelDoc, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	var doc ModelDoc
	err := s.db.Collection("models").FindOne(
		ctx,
		bson.M{},
		options.FindOne().SetSort(bson.D{{Key: "version", Value: -1}}),
	).Decode(&doc)
	if err != nil {
		return nil, err
	}
	return &doc, nil
}

// ModelDocToLinearModel reconstructs a LinearModel from a stored document.
func ModelDocToLinearModel(d *ModelDoc) *model.LinearModel {
	m := &model.LinearModel{
		Weights:      d.Weights,
		FeatureNames: d.FeatureNames,
		Epochs:       d.Epochs,
		LearningRate: d.LearningRate,
		BatchSize:    d.BatchSize,
		Workers:      d.Workers,
		LossHistory:  d.LossHistory,
		TrainMAE:     d.TrainMAE,
		TrainRMSE:    d.TrainRMSE,
		TrainR2:      d.TrainR2,
		TestMAE:      d.TestMAE,
		TestRMSE:     d.TestRMSE,
		TestR2:       d.TestR2,
	}
	for i := range m.Means {
		if i < len(d.Means) {
			m.Means[i] = d.Means[i]
		}
		if i < len(d.Stds) {
			m.Stds[i] = d.Stds[i]
		}
	}
	return m
}

// SaveTrainingRun records metadata about a completed training job.
func (s *Store) SaveTrainingRun(run TrainingRunDoc) error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_, err := s.db.Collection("training_runs").InsertOne(ctx, run)
	return err
}

// SavePrediction records an individual prediction (called asynchronously).
func (s *Store) SavePrediction(pred PredictionDoc) error {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	_, err := s.db.Collection("predictions").InsertOne(ctx, pred)
	return err
}

// UpsertInsight stores or replaces an EDA insight dataset by name.
func (s *Store) UpsertInsight(name string, rows [][]interface{}) error {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	_, err := s.db.Collection("insights").UpdateOne(
		ctx,
		bson.M{"name": name},
		bson.M{"$set": bson.M{"name": name, "rows": rows}},
		options.Update().SetUpsert(true),
	)
	return err
}

// GetInsight retrieves a named EDA insight dataset.
func (s *Store) GetInsight(name string) (*InsightDoc, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	var doc InsightDoc
	if err := s.db.Collection("insights").FindOne(ctx, bson.M{"name": name}).Decode(&doc); err != nil {
		return nil, err
	}
	return &doc, nil
}
