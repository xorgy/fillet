<div align=center>

# Fillet

**A thin pointer based contiguous collection.**

[![ISC or Apache 2.0 or MIT license.](https://img.shields.io/badge/license-ISC_OR_Apache--2.0_OR_MIT-blue.svg)](#license)
[![Build status](https://github.com/xorgy/fillet/workflows/CI/badge.svg)](https://github.com/xorgy/fillet/actions)
[![Crates.io](https://img.shields.io/crates/v/fillet.svg)](<https://crates.io/crates/fillet>)
[![Docs](https://docs.rs/fillet/badge.svg)](<https://docs.rs/fillet>)

</div>

Fillet is null/zero when empty, making it ideal for scenarios where most collections are empty and you're storing many references to them.
It handles zero-sized types (ZSTs) without heap allocations, using  `usize` for length.

Fillet is always pointer-sized and zero when empty.

Fillet does not reserve capacity, so repeated `push` operations can be slow as they always invoke the allocator and compute layouts.
However, `Extend` and `FromIterator` use amortized growth and perform similarly to `Vec`.

This crate is `no_std`.

## Features

- Same stack size and alignment as a thin pointer.
- Very efficient for empty collections.
- Fast `from_iter`/`collect` and `extend`, without reserved capacity.
- Performance similar to `Vec` in most cases.
- No dependencies outside `core` and `alloc`.

## Examples

### Basic Usage

```rust
use fillet::Fillet;

let mut f: Fillet<i32> = Fillet::from([1, 2, 3]);
assert_eq!(f.len(), 3);
assert_eq!(*f, [1, 2, 3]);

f.push(4);
assert_eq!(*f, [1, 2, 3, 4]);

f.truncate(2);
assert_eq!(*f, [1, 2]);
```

### Zero-Sized Types

```rust
use fillet::Fillet;
use core::iter::repeat_n;
use core::mem::size_of;

let mut f: Fillet<()> = repeat_n((), 5).collect();
assert_eq!(f.len(), 5);

// No heap allocation
f.push(());
assert_eq!(f.len(), 6);
assert_eq!(size_of::<Fillet<()>>(), size_of::<usize>());
```

### Extending from Within

```rust
use fillet::Fillet;

let mut f = Fillet::from([1, 2]);
f.extend_from_within(..);
assert_eq!(*f, [1, 2, 1, 2]);
```

## Minimum Supported Rust Version (MSRV)

Fillet has been verified to compile with **Rust 1.85** and later.

Future versions might increase the Rust version requirement.
It will not be treated as a breaking change, and as such can even happen with small patch releases.


## License

Triple licensed, at your option:

- ISC license
   ([LICENSE-ISC](LICENSE-ISC))
- Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license
   ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)


## Contribution

Contributions are welcome by pull request or email.
Please feel free to add your name to the [AUTHORS] file in any substantive pull request.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be licensed as above, without any additional terms or conditions.

[AUTHORS]: ./AUTHORS
