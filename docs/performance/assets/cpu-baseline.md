# Artefato — CPU profile do baseline (brute-force)

Capturado no passo [01](../01-perfil-do-gargalo.md). Ambiente: `local-pprof` (`go run .`,
sem cap de CPU), sob carga do `hammer.ps1`. Prova que o gargalo é a busca, não o JSON/rede.

## `top` — ranking por custo de CPU

```
Duration: 20s, Total samples = 149.23s (746.13%)
      flat  flat%   sum%        cum   cum%
   146.96s 98.48% 98.48%    147.38s 98.76%  rinha-fraud/internal/knn.(*Index).Score
     0.83s  0.56% 99.04%      0.83s  0.56%  runtime.cgocall
         0     0% 99.04%    147.49s 98.83%  main.(*server).handleScore
         0     0% 99.04%    147.49s 98.83%  net/http...ServeHTTP
```

`knn.(*Index).Score` = **98,48% do CPU**. `encoding/json` nem aparece (custo desprezível
neste regime) — o gargalo é 100% algorítmico.

## `list Score` — custo linha-a-linha (loop interno de distância)

```
       88s     88.34s (flat, cum) 98.94% of Total   knn.(*Index).Score
     1.54s      1.54s     95:	for idx := 0; idx < ix.n; idx++ {      // varre os 3M
    34.01s     34.09s     98:		for d := 0; d < vectorize.Dims; d++ {  // x 14 dimensões
    20.34s     20.41s     99:			diff := int32(q[d]) - int32(data[off+d])
    30.22s     30.40s    100:			dist += uint32(diff * diff)         // o cálculo da distância
```

As linhas 98-100 (o `diff*diff` por dimensão) somam ~84s de 88s. Cada request recalcula isso
3.000.000 × 14 vezes. **É isso que o passo 03 (busca rápida) elimina** — em vez de visitar os
3M, visitar só uns milhares.
