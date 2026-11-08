package aiwire

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/shared"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type fakeCompletion struct {
	Completion
	mu    sync.Mutex
	steps [][]openai.ChatCompletionMessageToolCallUnion
	calls int
}

func (f *fakeCompletion) next(ctx context.Context) (openai.ChatCompletionMessage, error) {
	if err := ctx.Err(); err != nil {
		return openai.ChatCompletionMessage{}, err
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	step := f.calls
	f.calls++
	if step < len(f.steps) {
		return openai.ChatCompletionMessage{ToolCalls: f.steps[step]}, nil
	}
	return openai.ChatCompletionMessage{Content: "done"}, nil
}

func (f *fakeCompletion) Completions(ctx context.Context, _ []openai.ChatCompletionMessageParamUnion, _ []openai.ChatCompletionToolUnionParam, _ CompletionOption) (CompletionResponse, error) {
	msg, err := f.next(ctx)
	if err != nil {
		return CompletionResponse{}, err
	}
	return CompletionResponse{Message: msg, Provider: "fake"}, nil
}

func (f *fakeCompletion) CompletionsStream(ctx context.Context, _ []openai.ChatCompletionMessageParamUnion, _ []openai.ChatCompletionToolUnionParam, _ CompletionOption, cb StreamCallback) error {
	msg, err := f.next(ctx)
	if err != nil {
		return err
	}
	if msg.Content != "" {
		if err := cb(StreamChunk{Content: msg.Content, Provider: "fake"}); err != nil {
			return err
		}
	}
	if len(msg.ToolCalls) > 0 {
		if err := cb(StreamChunk{ToolCalls: msg.ToolCalls, Provider: "fake"}); err != nil {
			return err
		}
	}
	return cb(StreamChunk{Done: true, Provider: "fake", Usage: &Usage{}})
}

type barrier struct {
	mu    sync.Mutex
	n     int
	count int
	all   chan struct{}
}

func newBarrier(n int) *barrier {
	return &barrier{n: n, all: make(chan struct{})}
}

func (b *barrier) arrive() bool {
	b.mu.Lock()
	b.count++
	if b.count == b.n {
		close(b.all)
	}
	b.mu.Unlock()
	select {
	case <-b.all:
		return true
	case <-time.After(time.Second):
		return false
	}
}

type eventLog struct {
	mu     sync.Mutex
	events []string
}

func (l *eventLog) add(e string) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.events = append(l.events, e)
}

func (l *eventLog) list() []string {
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]string(nil), l.events...)
}

type fakeTool struct {
	name       string
	concurrent *bool
	run        func(ctx context.Context, inputs map[string]any) (ToolResult, error)
}

func (t *fakeTool) Definition() openai.ChatCompletionToolUnionParam {
	return openai.ChatCompletionFunctionTool(shared.FunctionDefinitionParam{Name: t.name})
}

func (t *fakeTool) Execute(ctx context.Context, inputs map[string]any) (ToolResult, error) {
	return t.run(ctx, inputs)
}

type concurrentFakeTool struct{ *fakeTool }

func (t concurrentFakeTool) Concurrent() bool { return *t.concurrent }

func newTool(name string, concurrent bool, run func(ctx context.Context, inputs map[string]any) (ToolResult, error)) Tool {
	t := &fakeTool{name: name, concurrent: &concurrent, run: run}
	if concurrent {
		return concurrentFakeTool{t}
	}
	return t
}

func call(id, name, args string) openai.ChatCompletionMessageToolCallUnion {
	return openai.ChatCompletionMessageToolCallUnion{
		ID:       id,
		Type:     "function",
		Function: openai.ChatCompletionMessageFunctionToolCallFunction{Name: name, Arguments: args},
	}
}

func threeCalls() []openai.ChatCompletionMessageToolCallUnion {
	return []openai.ChatCompletionMessageToolCallUnion{
		call("c1", "t1", `{"n":1}`),
		call("c2", "t2", `{"n":2}`),
		call("c3", "t3", `{"n":3}`),
	}
}

type recorder struct {
	log      *eventLog
	inCb     atomic.Int32
	overlaps atomic.Int32
	pre      []PreTool
	post     []PostTool
}

func (r *recorder) enter() {
	if r.inCb.Add(1) != 1 {
		r.overlaps.Add(1)
	}
	time.Sleep(time.Millisecond)
}

func (r *recorder) preCb(p PreTool) {
	r.enter()
	defer r.inCb.Add(-1)
	r.pre = append(r.pre, p)
	r.log.add("pre:" + p.ID)
}

func (r *recorder) postCb(p PostTool) {
	r.enter()
	defer r.inCb.Add(-1)
	r.post = append(r.post, p)
	r.log.add("post:" + p.ID)
}

func barrierTools(log *eventLog, concurrent [3]bool) ([]Tool, *barrier) {
	b := newBarrier(3)
	tools := make([]Tool, 3)
	for i := range tools {
		name := fmt.Sprintf("t%d", i+1)
		tools[i] = newTool(name, concurrent[i], func(ctx context.Context, inputs map[string]any) (ToolResult, error) {
			log.add("run:" + name)
			if !b.arrive() {
				return &SimpleToolResult{ToolContent: name + " timeout"}, nil
			}
			return &SimpleToolResult{ToolContent: fmt.Sprintf("%s ok %v", name, inputs["n"])}, nil
		})
	}
	return tools, b
}

func TestAgent_ParallelStep(t *testing.T) {
	log := &eventLog{}
	tools, _ := barrierTools(log, [3]bool{true, true, true})
	svc := &fakeCompletion{steps: [][]openai.ChatCompletionMessageToolCallUnion{threeCalls()}}
	rec := &recorder{log: log}

	result, err := NewAgent(svc, 5).Execute(context.Background(), nil, tools, CompletionOption{}, rec.preCb, rec.postCb)
	require.NoError(t, err)

	require.Len(t, result.ToolResults, 3)
	for i, r := range result.ToolResults {
		assert.Equal(t, fmt.Sprintf("t%d ok %d", i+1, i+1), r.Content())
		assert.Equal(t, fmt.Sprintf("c%d", i+1), result.ToolCalls[i].ID)
	}

	events := log.list()
	assert.Equal(t, []string{"pre:c1", "pre:c2", "pre:c3"}, events[:3])
	assert.Equal(t, []string{"post:c1", "post:c2", "post:c3"}, events[6:])
	assert.Zero(t, rec.overlaps.Load())

	require.Len(t, rec.post, 3)
	for i, p := range rec.post {
		assert.Equal(t, rec.pre[i].ID, p.ID)
		assert.Equal(t, fmt.Sprintf("t%d", i+1), p.Name)
		assert.Equal(t, fmt.Sprintf(`{"n":%d}`, i+1), p.Args)
		assert.Positive(t, p.Duration)
	}
}

func TestAgent_ParallelStep_StreamOrder(t *testing.T) {
	done := []chan struct{}{make(chan struct{}), make(chan struct{}), make(chan struct{})}
	tools := make([]Tool, 3)
	for i := range tools {
		name := fmt.Sprintf("t%d", i+1)
		tools[i] = newTool(name, true, func(ctx context.Context, _ map[string]any) (ToolResult, error) {
			defer close(done[i])
			if i < 2 {
				select {
				case <-done[i+1]:
				case <-time.After(time.Second):
					return &SimpleToolResult{ToolContent: name + " timeout"}, nil
				}
			}
			return &SimpleToolResult{ToolContent: name}, nil
		})
	}
	svc := &fakeCompletion{steps: [][]openai.ChatCompletionMessageToolCallUnion{threeCalls()}}

	var streamed []string
	result, err := NewAgent(svc, 5).ExecuteStream(context.Background(), nil, tools, CompletionOption{},
		func(c StreamChunk) error {
			for _, r := range c.ToolResults {
				streamed = append(streamed, r.Content)
			}
			return nil
		}, nil, nil)
	require.NoError(t, err)

	assert.Equal(t, []string{"t1", "t2", "t3"}, streamed)
	require.Len(t, result.ToolResults, 3)
	for i, r := range result.ToolResults {
		assert.Equal(t, fmt.Sprintf("t%d", i+1), r.Content())
	}
}

func assertSerial(t *testing.T, log *eventLog, rec *recorder, result *AgentResult) {
	t.Helper()
	assert.Equal(t, []string{
		"pre:c1", "run:t1", "post:c1",
		"pre:c2", "run:t2", "post:c2",
		"pre:c3", "run:t3", "post:c3",
	}, log.list())
	assert.Zero(t, rec.overlaps.Load())
	require.Len(t, rec.post, 3)
	for i, p := range rec.post {
		assert.Equal(t, fmt.Sprintf("c%d", i+1), p.ID)
		assert.Equal(t, rec.pre[i].ID, p.ID)
		assert.Equal(t, fmt.Sprintf(`{"n":%d}`, i+1), p.Args)
	}
	require.Len(t, result.ToolResults, 3)
}

func serialTools(log *eventLog, concurrent [3]bool) []Tool {
	tools := make([]Tool, 3)
	for i := range tools {
		name := fmt.Sprintf("t%d", i+1)
		tools[i] = newTool(name, concurrent[i], func(context.Context, map[string]any) (ToolResult, error) {
			log.add("run:" + name)
			return &SimpleToolResult{ToolContent: name}, nil
		})
	}
	return tools
}

func TestAgent_MixedStepRunsSerial(t *testing.T) {
	log := &eventLog{}
	svc := &fakeCompletion{steps: [][]openai.ChatCompletionMessageToolCallUnion{threeCalls()}}
	rec := &recorder{log: log}

	result, err := NewAgent(svc, 5).Execute(context.Background(), nil, serialTools(log, [3]bool{true, false, true}), CompletionOption{}, rec.preCb, rec.postCb)
	require.NoError(t, err)
	assertSerial(t, log, rec, result)
}

func TestAgent_SwitchOffRunsSerial(t *testing.T) {
	log := &eventLog{}
	svc := &fakeCompletion{steps: [][]openai.ChatCompletionMessageToolCallUnion{threeCalls()}}
	rec := &recorder{log: log}

	result, err := NewAgent(svc, 5, WithParallelTools(false)).Execute(context.Background(), nil, serialTools(log, [3]bool{true, true, true}), CompletionOption{}, rec.preCb, rec.postCb)
	require.NoError(t, err)
	assertSerial(t, log, rec, result)
}

func TestAgent_UnknownAndBadArgsCountAsSafe(t *testing.T) {
	b := newBarrier(2)
	tool := func(name string) Tool {
		return newTool(name, true, func(context.Context, map[string]any) (ToolResult, error) {
			if !b.arrive() {
				return &SimpleToolResult{ToolContent: "timeout"}, nil
			}
			return &SimpleToolResult{ToolContent: "ok"}, nil
		})
	}
	unmarked := newTool("unmarked", false, func(context.Context, map[string]any) (ToolResult, error) {
		return &SimpleToolResult{ToolContent: "should not run"}, nil
	})
	svc := &fakeCompletion{steps: [][]openai.ChatCompletionMessageToolCallUnion{{
		call("c1", "t1", `{}`),
		call("c2", "missing", `{}`),
		call("c3", "unmarked", `not json`),
		call("c4", "t2", `{}`),
	}}}

	result, err := NewAgent(svc, 5).Execute(context.Background(), nil, []Tool{tool("t1"), tool("t2"), unmarked}, CompletionOption{}, nil, nil)
	require.NoError(t, err)
	require.Len(t, result.ToolResults, 4)
	assert.Equal(t, "ok", result.ToolResults[0].Content())
	assert.Contains(t, result.ToolResults[1].Content(), "not found")
	assert.Contains(t, result.ToolResults[2].Content(), "Failed to unmarshal")
	assert.Equal(t, "ok", result.ToolResults[3].Content())
}

func TestAgent_ParallelCancelReachesTools(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	b := newBarrier(3)
	go func() {
		<-b.all
		cancel()
	}()
	tools := make([]Tool, 3)
	for i := range tools {
		tools[i] = newTool(fmt.Sprintf("t%d", i+1), true, func(ctx context.Context, _ map[string]any) (ToolResult, error) {
			b.arrive()
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(time.Second):
				return &SimpleToolResult{ToolContent: "not cancelled"}, nil
			}
		})
	}
	svc := &fakeCompletion{steps: [][]openai.ChatCompletionMessageToolCallUnion{threeCalls()}}
	var post []PostTool

	_, err := NewAgent(svc, 5).Execute(ctx, nil, tools, CompletionOption{}, nil, func(p PostTool) { post = append(post, p) })
	assert.ErrorIs(t, err, context.Canceled)
	require.Len(t, post, 3)
	for _, p := range post {
		assert.ErrorIs(t, p.Result.Error(), context.Canceled)
	}
}

func TestAgent_ToolPanicBecomesError(t *testing.T) {
	for _, tc := range []struct {
		name       string
		concurrent bool
	}{{"serial", false}, {"parallel", true}} {
		t.Run(tc.name, func(t *testing.T) {
			tools := []Tool{
				newTool("t1", tc.concurrent, func(context.Context, map[string]any) (ToolResult, error) {
					panic("boom")
				}),
				newTool("t2", tc.concurrent, func(context.Context, map[string]any) (ToolResult, error) {
					return &SimpleToolResult{ToolContent: "fine"}, nil
				}),
			}
			svc := &fakeCompletion{steps: [][]openai.ChatCompletionMessageToolCallUnion{{
				call("c1", "t1", `{}`),
				call("c2", "t2", `{}`),
			}}}
			var post []PostTool

			result, err := NewAgent(svc, 5).Execute(context.Background(), nil, tools, CompletionOption{}, nil, func(p PostTool) { post = append(post, p) })
			require.NoError(t, err)
			assert.Equal(t, "done", result.Content)
			require.Len(t, result.ToolResults, 2)

			crashed := result.ToolResults[0]
			assert.True(t, errors.Is(crashed.Error(), ErrToolPanic))
			assert.Contains(t, crashed.Error().Error(), "boom")
			assert.Contains(t, crashed.Error().Error(), "goroutine")
			assert.Equal(t, "Tool t1 crashed", crashed.Content())
			assert.False(t, strings.Contains(crashed.Content(), "goroutine"))
			assert.Equal(t, "fine", result.ToolResults[1].Content())
			require.Len(t, post, 2)
			assert.ErrorIs(t, post[0].Result.Error(), ErrToolPanic)
		})
	}
}
