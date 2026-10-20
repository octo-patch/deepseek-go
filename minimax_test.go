package deepseek_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"

	deepseek "github.com/cohesion-org/deepseek-go"
	"github.com/stretchr/testify/require"
)

func miniMaxResponse(req *http.Request, body string) *http.Response {
	return &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body)), Request: req}
}

func TestMiniMaxRequestControls(t *testing.T) {
	cases := []struct{ model, thinking, effort string }{
		{deepseek.MiniMaxM3_1Flash, "", ""},
		{deepseek.MiniMaxM3_1Flash, "adaptive", "low"},
		{deepseek.MiniMaxM3_1Flash, "adaptive", "medium"},
		{deepseek.MiniMaxM3_1Flash, "adaptive", "high"},
		{deepseek.MiniMaxM3_1Flash, "adaptive", "xhigh"},
		{deepseek.MiniMaxM3_1Flash, "adaptive", "max"},
		{deepseek.MiniMaxM3, "", ""},
		{deepseek.MiniMaxM3, "adaptive", ""},
		{deepseek.MiniMaxM3, "disabled", ""},
		{deepseek.MiniMaxM2_7, "", ""},
		{"saved-custom-model", "disabled", ""},
	}
	for _, region := range []struct{ chat, anthropic string }{
		{deepseek.MiniMaxBaseURL, deepseek.MiniMaxAnthropicAPIBaseURL},
		{deepseek.MiniMaxCNBaseURL, deepseek.MiniMaxCNAnthropicAPIBaseURL},
		{"https://custom.example/v1/", "https://custom.example/anthropic"},
	} {
		for _, anthropic := range []bool{false, true} {
			for _, streaming := range []bool{false, true} {
				for _, tc := range cases {
					name := fmt.Sprintf("%s/anthropic=%t/stream=%t/%s/%s/%s", region.chat, anthropic, streaming, tc.model, tc.thinking, tc.effort)
					t.Run(name, func(t *testing.T) {
						called := false
						capture := requestCaptureFunc(func(req *http.Request) (*http.Response, error) {
							called = true
							require.Equal(t, http.MethodPost, req.Method)
							var payload map[string]interface{}
							require.NoError(t, json.NewDecoder(req.Body).Decode(&payload))
							require.Equal(t, tc.model, payload["model"])
							if tc.thinking == "" {
								require.NotContains(t, payload, "thinking")
							} else {
								require.Equal(t, map[string]interface{}{"type": tc.thinking}, payload["thinking"])
							}
							require.NotContains(t, payload, "service_tier")
							require.NotContains(t, payload, "reasoning_split")
							require.NotContains(t, payload, "enable_thinking")
							if streaming {
								require.Equal(t, true, payload["stream"])
							}
							if anthropic {
								require.Equal(t, region.anthropic+"/v1/messages", req.URL.String())
								require.Equal(t, "fixture-key", req.Header.Get("x-api-key"))
								require.Equal(t, float64(4096), payload["max_tokens"])
								require.NotContains(t, payload, "reasoning_effort")
								if tc.effort == "" {
									require.NotContains(t, payload, "output_config")
								} else {
									require.Equal(t, map[string]interface{}{"effort": tc.effort}, payload["output_config"])
								}
								return miniMaxResponse(req, `{"id":"msg-test","role":"assistant","content":[]}`), nil
							}
							require.Equal(t, region.chat+"chat/completions", req.URL.String())
							require.Equal(t, "Bearer fixture-key", req.Header.Get("Authorization"))
							require.Equal(t, float64(4096), payload["max_completion_tokens"])
							require.NotContains(t, payload, "extra_fields")
							require.NotContains(t, payload, "output_config")
							if tc.effort == "" {
								require.NotContains(t, payload, "reasoning_effort")
							} else {
								require.Equal(t, tc.effort, payload["reasoning_effort"])
							}
							return miniMaxResponse(req, `{"id":"chatcmpl-test","choices":[{"message":{"role":"assistant","content":"Hello"}}]}`), nil
						})
						if anthropic {
							client := deepseek.NewAnthropicClient("fixture-key")
							client.BaseURL, client.Path, client.HTTPClient = region.anthropic, "/v1/messages", capture
							request := &deepseek.AnthropicRequest{Model: tc.model, MaxTokens: 4096,
								Messages: []deepseek.AnthropicMessage{{Role: "user", Content: []interface{}{deepseek.AnthropicTextBlock{Type: "text", Text: "Hello"}}}}}
							if tc.thinking != "" {
								request.Thinking = &deepseek.AnthropicThinkingConfig{Type: tc.thinking}
							}
							if tc.effort != "" {
								request.OutputConfig = &deepseek.AnthropicOutputConfig{Effort: tc.effort}
							}
							if streaming {
								stream, err := client.CreateAnthropicMessageStream(context.Background(), request)
								require.NoError(t, err)
								require.NoError(t, stream.Close())
							} else {
								_, err := client.CreateAnthropicMessage(context.Background(), request)
								require.NoError(t, err)
							}
						} else {
							client := deepseek.NewClient("fixture-key", region.chat)
							client.HTTPClient = capture
							messages := []deepseek.ChatCompletionMessage{{Role: "user", Content: "Hello"}}
							var thinking *deepseek.ThinkingConfig
							if tc.thinking != "" {
								thinking = &deepseek.ThinkingConfig{Type: tc.thinking}
							}
							extra := map[string]interface{}{"max_completion_tokens": 4096}
							if streaming {
								stream, err := client.CreateChatCompletionStream(context.Background(), &deepseek.StreamChatCompletionRequest{
									Model: tc.model, Messages: messages, Thinking: thinking, ReasoningEffort: tc.effort, ExtraFields: extra})
								require.NoError(t, err)
								require.NoError(t, stream.Close())
							} else {
								_, err := client.CreateChatCompletion(context.Background(), &deepseek.ChatCompletionRequest{
									Model: tc.model, Messages: messages, Thinking: thinking, ReasoningEffort: tc.effort, ExtraFields: extra})
								require.NoError(t, err)
							}
						}
						require.True(t, called)
					})
				}
			}
		}
	}
}

func TestMiniMaxToolHistoryAndImages(t *testing.T) {
	for _, model := range []string{deepseek.MiniMaxM3_1Flash, deepseek.MiniMaxM3} {
		for _, baseURL := range []string{deepseek.MiniMaxBaseURL, deepseek.MiniMaxCNBaseURL} {
			t.Run(baseURL+model, func(t *testing.T) {
				calls := 0
				client := deepseek.NewClient("fixture-key", baseURL)
				client.HTTPClient = requestCaptureFunc(func(req *http.Request) (*http.Response, error) {
					calls++
					var payload map[string]interface{}
					require.NoError(t, json.NewDecoder(req.Body).Decode(&payload))
					require.Equal(t, baseURL+"chat/completions", req.URL.String())
					require.Equal(t, model, payload["model"])
					require.Len(t, payload["tools"], 1)
					if calls > 1 {
						messages := payload["messages"].([]interface{})
						assistant := messages[1].(map[string]interface{})
						require.Equal(t, "<think>original reasoning</think>", assistant["content"])
						require.Equal(t, "separate reasoning", assistant["reasoning_content"])
						require.Len(t, assistant["tool_calls"], 1)
						require.Equal(t, "call-1", messages[2].(map[string]interface{})["tool_call_id"])
						content := messages[3].(map[string]interface{})["content"].([]interface{})
						require.Equal(t, "data:image/png;base64,aW1hZ2U=", content[1].(map[string]interface{})["image_url"].(map[string]interface{})["url"])
					}
					return miniMaxResponse(req, `{"id":"chatcmpl-test","choices":[{"message":{"role":"assistant","content":"<think>original reasoning</think>","reasoning_content":"separate reasoning","tool_calls":[{"id":"call-1","type":"function","function":{"name":"weather","arguments":"{}"}}]}}]}`), nil
				})
				tools := []deepseek.Tool{{Type: "function", Function: deepseek.Function{Name: "weather", Parameters: &deepseek.FunctionParameters{Type: "object"}}}}
				response, err := client.CreateChatCompletion(context.Background(), &deepseek.ChatCompletionRequest{
					Model: model, Tools: tools, Messages: []deepseek.ChatCompletionMessage{{Role: "user", Content: "Weather?"}},
				})
				require.NoError(t, err)
				message := response.Choices[0].Message
				messages := []deepseek.ChatCompletionMessageWithImage{
					{Role: "user", Content: "Weather?"},
					{Role: message.Role, Content: message.Content, ReasoningContent: message.ReasoningContent, ToolCalls: message.ToolCalls},
					{Role: "tool", Content: "Sunny", ToolCallID: "call-1"},
					deepseek.NewImageMessage("user", "What about here?", "data:image/png;base64,aW1hZ2U="),
				}
				_, err = client.CreateChatCompletionWithImage(context.Background(), &deepseek.ChatCompletionRequestWithImage{Model: model, Messages: messages, Tools: tools})
				require.NoError(t, err)
				stream, err := client.CreateChatCompletionStreamWithImage(context.Background(), &deepseek.StreamChatCompletionRequestWithImage{Model: model, Messages: messages, Tools: tools})
				require.NoError(t, err)
				require.NoError(t, stream.Close())
				require.Equal(t, 3, calls)
			})
		}
	}
}

func TestMiniMaxAnthropicToolHistoryAndImages(t *testing.T) {
	for _, model := range []string{deepseek.MiniMaxM3_1Flash, deepseek.MiniMaxM3} {
		for _, baseURL := range []string{deepseek.MiniMaxAnthropicAPIBaseURL, deepseek.MiniMaxCNAnthropicAPIBaseURL} {
			t.Run(baseURL+model, func(t *testing.T) {
				calls := 0
				client := deepseek.NewAnthropicClient("fixture-key")
				client.BaseURL, client.Path = baseURL, "/v1/messages"
				const blocks = `[{"type":"thinking","thinking":"original reasoning","signature":"original-signature"},{"type":"text","text":"Checking"},{"type":"tool_use","id":"call-1","name":"weather","input":{}}]`
				client.HTTPClient = requestCaptureFunc(func(req *http.Request) (*http.Response, error) {
					calls++
					var payload map[string]json.RawMessage
					require.NoError(t, json.NewDecoder(req.Body).Decode(&payload))
					require.Equal(t, baseURL+"/v1/messages", req.URL.String())
					require.JSONEq(t, `"`+model+`"`, string(payload["model"]))
					require.Contains(t, payload, "tools")
					if calls > 1 {
						var messages []map[string]json.RawMessage
						require.NoError(t, json.Unmarshal(payload["messages"], &messages))
						require.JSONEq(t, blocks, string(messages[1]["content"]))
						require.JSONEq(t, `[{"type":"tool_result","tool_use_id":"call-1","content":"Sunny"},{"type":"image","source":{"type":"base64","media_type":"image/png","data":"aW1hZ2U="}}]`, string(messages[2]["content"]))
					}
					return miniMaxResponse(req, `{"id":"msg-test","role":"assistant","content":`+blocks+`}`), nil
				})
				request := &deepseek.AnthropicRequest{Model: model, MaxTokens: 4096,
					Tools:    []deepseek.AnthropicTool{{Name: "weather", InputSchema: map[string]string{"type": "object"}}},
					Messages: []deepseek.AnthropicMessage{{Role: "user", Content: []interface{}{deepseek.AnthropicTextBlock{Type: "text", Text: "Weather?"}}}}}
				response, err := client.CreateAnthropicMessage(context.Background(), request)
				require.NoError(t, err)
				request.Messages = append(request.Messages,
					deepseek.AnthropicMessage{Role: response.Role, Content: response.Content},
					deepseek.AnthropicMessage{Role: "user", Content: []interface{}{
						deepseek.AnthropicToolResultBlock{Type: "tool_result", ToolUseID: "call-1", Content: "Sunny"},
						map[string]interface{}{"type": "image", "source": map[string]string{"type": "base64", "media_type": "image/png", "data": "aW1hZ2U="}},
					}})
				_, err = client.CreateAnthropicMessage(context.Background(), request)
				require.NoError(t, err)
				stream, err := client.CreateAnthropicMessageStream(context.Background(), request)
				require.NoError(t, err)
				require.NoError(t, stream.Close())
				require.Equal(t, 3, calls)
			})
		}
	}
}
