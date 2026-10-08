package main

import (
	"encoding/json"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"
)

type Model struct {
	ID   string
	Type string
	URL  string
}

type RequestProfile struct {
	ModelID     string
	EnableThink bool
	MaxTokens   int
}

var models = map[string]Model{
	"qwen3-4b": {
		ID:   "qwen3-4b",
		Type: "text",
		URL:  "http://localhost:8000",
	},
	"qwen3-vl-2b": {
		ID:   "qwen3-vl-2b",
		Type: "vision",
		URL:  "http://localhost:8001",
	},
}

var (
	modelMu     sync.Mutex
	activeModel string
)

const composeDir = "/home/abidnafi/ai-server"

func main() {
	http.HandleFunc("/v1/chat/completions", chatCompletions)
	http.HandleFunc("/v1/models", listModels)
	http.HandleFunc("/health", health)

	log.Println("AI Gateway listening on :8080")

	if err := http.ListenAndServe(":8080", nil); err != nil {
		log.Fatal(err)
	}
}

func listModels(w http.ResponseWriter, r *http.Request) {
	type ModelResponse struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		OwnedBy string `json:"owned_by"`
	}

	result := []ModelResponse{
		{ID: "auto", Object: "model", OwnedBy: "local"},
		{ID: "qwen3-4b-fast", Object: "model", OwnedBy: "local"},
		{ID: "qwen3-4b-thinking", Object: "model", OwnedBy: "local"},
		{ID: "qwen3-4b-deep", Object: "model", OwnedBy: "local"},
		{ID: "qwen3-vl-2b", Object: "model", OwnedBy: "local"},
	}

	writeJSON(w, map[string]interface{}{
		"object": "list",
		"data":   result,
	})
}

func health(w http.ResponseWriter, r *http.Request) {
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("ok"))
}

func chatCompletions(w http.ResponseWriter, r *http.Request) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	var request map[string]interface{}

	if err := json.Unmarshal(body, &request); err != nil {
		http.Error(w, "invalid JSON", http.StatusBadRequest)
		return
	}

	requestedModel, _ := request["model"].(string)
	vision := containsImage(request)

	profile := resolveProfile(requestedModel, vision)
	target := models[profile.ModelID]

	if err := ensureModel(target); err != nil {
		http.Error(
			w,
			"failed to start model: "+err.Error(),
			http.StatusBadGateway,
		)
		return
	}

	request["model"] = target.ID

	request["chat_template_kwargs"] = map[string]interface{}{
		"enable_thinking": profile.EnableThink,
	}

	if profile.MaxTokens > 0 {
		request["max_tokens"] = profile.MaxTokens
	}

	updatedBody, err := json.Marshal(request)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	log.Printf(
		"model=%s thinking=%v max_tokens=%d",
		profile.ModelID,
		profile.EnableThink,
		profile.MaxTokens,
	)

	forwardRequest(
		w,
		r,
		target.URL+"/v1/chat/completions",
		updatedBody,
	)
}

func resolveProfile(requestedModel string, vision bool) RequestProfile {
	if vision {
		return RequestProfile{
			ModelID:     "qwen3-vl-2b",
			EnableThink: false,
			MaxTokens:   1024,
		}
	}

	switch requestedModel {
	case "qwen3-4b-fast":
		return RequestProfile{
			ModelID:     "qwen3-4b",
			EnableThink: false,
			MaxTokens:   1024,
		}

	case "qwen3-4b-thinking":
		return RequestProfile{
			ModelID:     "qwen3-4b",
			EnableThink: true,
			MaxTokens:   2048,
		}

	case "qwen3-4b-deep":
		return RequestProfile{
			ModelID:     "qwen3-4b",
			EnableThink: true,
			MaxTokens:   4096,
		}

	default:
		return RequestProfile{
			ModelID:     "qwen3-4b",
			EnableThink: true,
			MaxTokens:   2048,
		}
	}
}

func containsImage(request map[string]interface{}) bool {
	messages, ok := request["messages"].([]interface{})
	if !ok {
		return false
	}

	for _, rawMessage := range messages {
		message, ok := rawMessage.(map[string]interface{})
		if !ok {
			continue
		}

		content, ok := message["content"].([]interface{})
		if !ok {
			continue
		}

		for _, rawContent := range content {
			item, ok := rawContent.(map[string]interface{})
			if !ok {
				continue
			}

			contentType, _ := item["type"].(string)

			if contentType == "image_url" || contentType == "image" {
				return true
			}
		}
	}

	return false
}

func ensureModel(target Model) error {
	modelMu.Lock()
	defer modelMu.Unlock()

	if isModelReady(target) {
		activeModel = target.ID
		log.Printf("Model %s is already running", target.ID)
		return nil
	}

	oppositeProfile := "vision"

	if target.Type == "vision" {
		oppositeProfile = "text"
	}

	log.Printf(
		"Model %s is not running. Stopping %s model...",
		target.ID,
		oppositeProfile,
	)

	if err := compose(oppositeProfile, "down"); err != nil {
		log.Printf(
			"Warning stopping %s model: %v",
			oppositeProfile,
			err,
		)
	}

	profile := "text"

	if target.Type == "vision" {
		profile = "vision"
	}

	log.Printf("Starting %s...", target.ID)

	if err := compose(profile, "up", "-d"); err != nil {
		return err
	}

	log.Printf("Waiting for %s to become ready...", target.ID)

	if err := waitForModel(target); err != nil {
		return err
	}

	activeModel = target.ID

	log.Printf("Model %s is ready", target.ID)

	return nil
}

func compose(profile string, args ...string) error {
	cmdArgs := []string{
		"compose",
		"--profile",
		profile,
	}

	cmdArgs = append(cmdArgs, args...)

	cmd := exec.Command("docker", cmdArgs...)
	cmd.Dir = composeDir

	output, err := cmd.CombinedOutput()

	if err != nil {
		log.Printf(
			"docker compose error: %s",
			strings.TrimSpace(string(output)),
		)
	}

	return err
}

func waitForModel(model Model) error {
	client := &http.Client{
		Timeout: 2 * time.Second,
	}

	timeout := time.After(5 * time.Minute)
	ticker := time.NewTicker(2 * time.Second)

	defer ticker.Stop()

	for {
		select {
		case <-timeout:
			return os.ErrDeadlineExceeded

		case <-ticker.C:
			resp, err := client.Get(model.URL + "/v1/models")

			if err != nil {
				continue
			}

			resp.Body.Close()

			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}
	}
}

func isModelReady(model Model) bool {
	client := &http.Client{
		Timeout: 2 * time.Second,
	}

	resp, err := client.Get(model.URL + "/v1/models")

	if err != nil {
		return false
	}

	defer resp.Body.Close()

	return resp.StatusCode == http.StatusOK
}

func forwardRequest(
	w http.ResponseWriter,
	r *http.Request,
	targetURL string,
	body []byte,
) {
	req, err := http.NewRequest(
		r.Method,
		targetURL,
		strings.NewReader(string(body)),
	)

	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	req.Header.Set("Content-Type", "application/json")

	if auth := r.Header.Get("Authorization"); auth != "" {
		req.Header.Set("Authorization", auth)
	}

	client := &http.Client{}

	resp, err := client.Do(req)

	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}

	defer resp.Body.Close()

	for key, values := range resp.Header {
		for _, value := range values {
			w.Header().Add(key, value)
		}
	}

	w.WriteHeader(resp.StatusCode)

	io.Copy(w, resp.Body)
}

func writeJSON(w http.ResponseWriter, value interface{}) {
	w.Header().Set("Content-Type", "application/json")

	if err := json.NewEncoder(w).Encode(value); err != nil {
		log.Println("JSON encoding error:", err)
	}
}
