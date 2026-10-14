# minidecode

A small transformer inference engine in numpy. It trains its own tiny model in
seconds on a CPU, then serves it with the tricks real inference servers use:
a KV cache, int8 and int4 weights, batching and speculative decoding. Each one
is measured against the plain version.

Work in progress.
