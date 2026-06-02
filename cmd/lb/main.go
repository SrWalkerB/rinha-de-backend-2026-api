// Command lb is a minimal L4 (TCP) load balancer for the rinha-fraud stack.
//
// Why not nginx. nginx is an L7 proxy: it parses HTTP, manages an upstream
// keepalive pool, runs its worker/event machinery — all of which costs CPU. On
// this contest the whole budget is 1 CPU shared with the (CPU-bound) APIs, so the
// ~0.30 CPU nginx needs to not throttle itself is 0.30 NOT spent on fraud scoring.
//
// This LB does the minimum a load balancer must: accept a client connection,
// round-robin it to one of the backends, and splice bytes both ways for the
// connection's lifetime. It never parses HTTP. On Linux, io.Copy between two
// TCPConns uses the splice(2) syscall (kernel zero-copy), so per-byte CPU is
// negligible — the LB should hold its SLA on ~0.05-0.10 CPU, freeing the rest for
// the APIs. It satisfies the rule "≥1 load balancer + ≥2 API instances": it IS a
// load balancer (just not nginx) in front of the two API instances.
//
// Pure Go stdlib, no deps; static binary (CGO disabled).
package main

import (
	"io"
	"log"
	"net"
	"os"
	"strings"
	"sync/atomic"
	"time"
)

func getenv(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func main() {
	addr := getenv("LB_ADDR", ":9999")
	backends := strings.Split(getenv("LB_BACKENDS", "api1:8080,api2:8080"), ",")
	for i := range backends {
		backends[i] = strings.TrimSpace(backends[i])
	}
	if len(backends) == 0 || backends[0] == "" {
		log.Fatal("lb: no backends (set LB_BACKENDS=host:port,host:port)")
	}

	ln, err := net.Listen("tcp", addr)
	if err != nil {
		log.Fatalf("lb: listen %s: %v", addr, err)
	}
	log.Printf("lb: L4 proxy on %s -> %v", addr, backends)

	var rr uint64
	for {
		c, err := ln.Accept()
		if err != nil {
			// transient accept errors (e.g. fd pressure): brief backoff, keep serving.
			log.Printf("lb: accept: %v", err)
			time.Sleep(5 * time.Millisecond)
			continue
		}
		n := atomic.AddUint64(&rr, 1)
		go proxy(c, backends, n)
	}
}

// proxy connects the client to a backend (round-robin, with a one-shot failover to
// the next backend) and splices bytes both ways until either side closes.
func proxy(client net.Conn, backends []string, n uint64) {
	defer client.Close()
	tuneTCP(client)

	idx := int(n % uint64(len(backends)))
	backend, err := net.DialTimeout("tcp", backends[idx], time.Second)
	if err != nil {
		// failover: try the next backend once.
		backend, err = net.DialTimeout("tcp", backends[(idx+1)%len(backends)], time.Second)
		if err != nil {
			return
		}
	}
	defer backend.Close()
	tuneTCP(backend)

	// Splice both directions with half-close propagation. On Linux
	// io.Copy(TCPConn,TCPConn) uses splice(2). When one direction's copy ends
	// (a peer half-closed its write side / sent FIN), signal EOF to the OTHER
	// peer with CloseWrite — NOT a hard Close — and wait for BOTH directions to
	// drain before the defers tear the sockets down. Tearing down on the first
	// EOF (the naive `<-done` once) would truncate an in-flight response on the
	// other direction under HTTP/1.1 keep-alive: the client's request stream can
	// end (FIN) while the backend is still streaming the response, and a hard
	// close there RSTs the connection → the request is scored as an error. The
	// CloseWrite lets the backend see end-of-request, finish the response, then
	// close — at which point the second copy returns cleanly.
	done := make(chan struct{}, 2)
	go func() {
		_, _ = io.Copy(backend, client)
		if t, ok := backend.(*net.TCPConn); ok {
			_ = t.CloseWrite() // tell the backend the request stream ended
		}
		done <- struct{}{}
	}()
	go func() {
		_, _ = io.Copy(client, backend)
		if t, ok := client.(*net.TCPConn); ok {
			_ = t.CloseWrite() // tell the client the response stream ended
		}
		done <- struct{}{}
	}()
	<-done
	<-done
}

func tuneTCP(c net.Conn) {
	if t, ok := c.(*net.TCPConn); ok {
		_ = t.SetNoDelay(true) // latency: don't Nagle-buffer the tiny JSON
	}
}
