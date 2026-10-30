package ducktape

import (
	"context"
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"sync"

	"github.com/tetratelabs/wazero"
	"github.com/tetratelabs/wazero/api"
)

//go:embed core.wasm
var coreWasm []byte

// Core is a compiled core.wasm; every operation runs in its own instance of it.
type Core struct {
	runtime  wazero.Runtime
	compiled wazero.CompiledModule
}

// NewCore compiles the given wasm bytes, or the embedded core when nil.
func NewCore(ctx context.Context, wasm []byte) (*Core, error) {
	if wasm == nil {
		wasm = coreWasm
	}
	runtime := wazero.NewRuntime(ctx)
	compiled, err := runtime.CompileModule(ctx, wasm)
	if err != nil {
		_ = runtime.Close(ctx)
		return nil, fmt.Errorf("ducktape: compile core.wasm: %w", err)
	}
	return &Core{runtime: runtime, compiled: compiled}, nil
}

// Close releases the compiled module and every instance made from it.
func (c *Core) Close(ctx context.Context) error {
	return c.runtime.Close(ctx)
}

var (
	defaultCoreOnce sync.Once
	defaultCoreVal  *Core
	defaultCoreErr  error
)

// DefaultCore is the embedded core compiled once per process.
func DefaultCore(ctx context.Context) (*Core, error) {
	defaultCoreOnce.Do(func() {
		defaultCoreVal, defaultCoreErr = NewCore(context.WithoutCancel(ctx), nil)
	})
	return defaultCoreVal, defaultCoreErr
}

type instance struct {
	mod       api.Module
	alloc     api.Function
	free      api.Function
	opStart   api.Function
	opResume  api.Function
	opFeed    api.Function
	opSecret  api.Function
	opFree    api.Function
	scratch   uint32
	operation uint64
}

func (c *Core) instantiate(ctx context.Context) (*instance, error) {
	mod, err := c.runtime.InstantiateModule(ctx, c.compiled, wazero.NewModuleConfig().WithName(""))
	if err != nil {
		return nil, fmt.Errorf("ducktape: instantiate core.wasm: %w", err)
	}
	version, err := mod.ExportedFunction("ducktape_abi_version").Call(ctx)
	if err != nil || len(version) != 1 || version[0] != 1 {
		_ = mod.Close(ctx)
		return nil, errors.New("ducktape: core.wasm has an unknown ABI version")
	}
	in := &instance{
		mod:      mod,
		alloc:    mod.ExportedFunction("ducktape_alloc"),
		free:     mod.ExportedFunction("ducktape_free"),
		opStart:  mod.ExportedFunction("ducktape_op_start"),
		opResume: mod.ExportedFunction("ducktape_op_resume"),
		opFeed:   mod.ExportedFunction("ducktape_op_feed"),
		opSecret: mod.ExportedFunction("ducktape_op_set_secret"),
		opFree:   mod.ExportedFunction("ducktape_op_free"),
	}
	scratch, err := in.put(ctx, make([]byte, 8))
	if err != nil {
		_ = mod.Close(ctx)
		return nil, err
	}
	in.scratch = scratch
	return in, nil
}

func (in *instance) put(ctx context.Context, b []byte) (uint32, error) {
	if len(b) == 0 {
		return 0, nil
	}
	res, err := in.alloc.Call(ctx, uint64(len(b)))
	if err != nil {
		return 0, fmt.Errorf("ducktape: core alloc: %w", err)
	}
	ptr := uint32(res[0])
	if !in.mod.Memory().Write(ptr, b) {
		return 0, errors.New("ducktape: core memory write out of range")
	}
	return ptr, nil
}

func (in *instance) release(ctx context.Context, ptr uint32, n int) {
	if ptr != 0 && n != 0 {
		_, _ = in.free.Call(ctx, uint64(ptr), uint64(n))
	}
}

func (in *instance) take(ctx context.Context) (step, error) {
	mem := in.mod.Memory()
	ptr, ok1 := mem.ReadUint32Le(in.scratch)
	n, ok2 := mem.ReadUint32Le(in.scratch + 4)
	if !ok1 || !ok2 {
		return step{}, errors.New("ducktape: core scratch read out of range")
	}
	view, ok := mem.Read(ptr, n)
	if !ok {
		return step{}, errors.New("ducktape: core step read out of range")
	}
	raw := make([]byte, n)
	copy(raw, view)
	in.release(ctx, ptr, int(n))
	var s step
	if err := json.Unmarshal(raw, &s); err != nil {
		return step{}, fmt.Errorf("ducktape: core step: %w", err)
	}
	s.raw = raw
	return s, nil
}

func (in *instance) start(ctx context.Context, doc any) (step, error) {
	b, err := json.Marshal(doc)
	if err != nil {
		return step{}, fmt.Errorf("ducktape: encode start: %w", err)
	}
	ptr, err := in.put(ctx, b)
	if err != nil {
		return step{}, err
	}
	defer in.release(ctx, ptr, len(b))
	res, err := in.opStart.Call(ctx, uint64(ptr), uint64(len(b)), uint64(in.scratch), uint64(in.scratch+4))
	if err != nil {
		return step{}, fmt.Errorf("ducktape: core start: %w", err)
	}
	in.operation = res[0]
	return in.take(ctx)
}

func (in *instance) resume(ctx context.Context, doc any) (step, error) {
	b, err := json.Marshal(doc)
	if err != nil {
		return step{}, fmt.Errorf("ducktape: encode resume: %w", err)
	}
	ptr, err := in.put(ctx, b)
	if err != nil {
		return step{}, err
	}
	defer in.release(ctx, ptr, len(b))
	if _, err := in.opResume.Call(ctx, in.operation, uint64(ptr), uint64(len(b)), uint64(in.scratch), uint64(in.scratch+4)); err != nil {
		return step{}, fmt.Errorf("ducktape: core resume: %w", err)
	}
	return in.take(ctx)
}

func (in *instance) feed(ctx context.Context, body []byte, clock clockDoc) (step, error) {
	cb, err := json.Marshal(clock)
	if err != nil {
		return step{}, err
	}
	cptr, err := in.put(ctx, cb)
	if err != nil {
		return step{}, err
	}
	defer in.release(ctx, cptr, len(cb))
	bptr, err := in.put(ctx, body)
	if err != nil {
		return step{}, err
	}
	defer in.release(ctx, bptr, len(body))
	if _, err := in.opFeed.Call(ctx, in.operation, uint64(bptr), uint64(len(body)), uint64(cptr), uint64(len(cb)), uint64(in.scratch), uint64(in.scratch+4)); err != nil {
		return step{}, fmt.Errorf("ducktape: core feed: %w", err)
	}
	return in.take(ctx)
}

func (in *instance) setSecret(ctx context.Context, ref string, secret []byte) error {
	rb := []byte(ref)
	rptr, err := in.put(ctx, rb)
	if err != nil {
		return err
	}
	defer in.release(ctx, rptr, len(rb))
	sptr, err := in.put(ctx, secret)
	if err != nil {
		return err
	}
	defer in.release(ctx, sptr, len(secret))
	_, err = in.opSecret.Call(ctx, in.operation, uint64(rptr), uint64(len(rb)), uint64(sptr), uint64(len(secret)))
	return err
}

func (in *instance) close(ctx context.Context) {
	if in.operation != 0 {
		_, _ = in.opFree.Call(ctx, in.operation)
		in.operation = 0
	}
	_ = in.mod.Close(ctx)
}
