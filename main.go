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
		{
			ID:      "auto",
			Object:  "model",
			OwnedBy: "local",
		},
		{
			ID:      "qwen3-4b",
			Object:  "model",
			OwnedBy: "local",
		},
		{
			ID:      "qwen3-vl-2b",
			Object:  "model",
			OwnedBy: "local",
		},
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

	vision := containsImage(request)

	var target Model

	if vision {
		target = models["qwen3-vl-2b"]
		log.Printf("Request requires vision model")
	} else {
		target = models["qwen3-4b"]
		log.Printf("Request requires text model")
	}

	// Make sure the correct model is running.
	if err := ensureModel(target); err != nil {
		http.Error(w, "failed to start model: "+err.Error(), http.StatusBadGateway)
		return
	}

	// vLLM expects its actual served model name.
	request["model"] = target.ID

	updatedBody, err := json.Marshal(request)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	forwardRequest(
		w,
		r,
		target.URL+"/v1/chat/completions",
		updatedBody,
	)
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

			if contentType == "image_url" ||
				contentType == "image" {
				return true
			}
		}
	}

	return false
}

func ensureModel(target Model) error {
	modelMu.Lock()
	defer modelMu.Unlock()

	if activeModel == target.ID {
		if isModelReady(target) {
			return nil
		}

		log.Printf("Model %s is no longer ready", target.ID)
		activeModel = ""
	}

	// Stop both profiles to guarantee only one model is running.
	log.Printf("Stopping existing models...")
	if err := compose("text", "down"); err != nil {
		log.Printf("text model stop warning: %v", err)
	}

	if err := compose("vision", "down"); err != nil {
		log.Printf("vision model stop warning: %v", err)
	}

	profile := "text"

	if target.Type == "vision" {
		profile = "vision"
	}

	log.Printf("Starting %s model...", target.ID)

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

			log.Printf(
				"Model not ready yet: HTTP %d",
				resp.StatusCode,
			)
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
	json.NewEncoder(w).Encode(value)
}
