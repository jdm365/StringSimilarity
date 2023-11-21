##Rust project with pyo3
## build with maturin

install:
	RUSTFLAGS="-C target-cpu=native -C target-feature=+avx2,+fma -C target-feature=+avx,+sse2,+sse4.1,+sse4.2" maturin develop --release

test:
	cargo test --release

clean:
	cargo clean
