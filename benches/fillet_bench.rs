use core::hint::black_box;
use core::iter::FromIterator;
use core::num::NonZeroU128;
use core::time::Duration;
use criterion::{BatchSize, Criterion, SamplingMode, criterion_group, criterion_main};
use rand::{Rng, rng};

use fillet::Fillet; // Replace with your crate path if needed

use shuffling_allocator::ShufflingAllocator;
use std::alloc::System;
#[global_allocator]
static ALLOC: ShufflingAllocator<System> = shuffling_allocator::wrap!(&System);

fn bench_from_iter_large_unknown(c: &mut Criterion) {
    let mut group = c.benchmark_group("FromIterator Large Unknown Size");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(2000);
    let size = 1_000_000;
    let gen_bench = || (0..size).filter(|&x| x % 2 == 0).map(|x| x * 2);

    group.bench_function("Fillet", |b| {
        b.iter_batched(
            &gen_bench,
            |iter| {
                black_box(Fillet::from_iter(iter));
            },
            BatchSize::SmallInput,
        )
    });

    group.bench_function("Vec", |b| {
        b.iter_batched(
            &gen_bench,
            |iter| {
                black_box(Vec::from_iter(iter));
            },
            BatchSize::SmallInput,
        )
    });

    group.finish();
}

fn bench_from_slice(c: &mut Criterion) {
    for (size, time, samples) in [
        (100, Duration::from_secs(3), 5000),
        (10_000, Duration::from_secs(10), 2500),
        (1_000_000, Duration::from_secs(20), 1000),
    ] {
        let mut group = c.benchmark_group(format!("From Slice {size}"));
        group.measurement_time(time);
        group.sample_size(samples);

        let slice: Vec<i32> = (0..size).collect();

        group.bench_with_input("Fillet", &slice, |b, s| {
            b.iter(|| black_box(Fillet::from(&**s)))
        });

        group.bench_with_input("Vec", &slice, |b, s| b.iter(|| black_box(Vec::from(&**s))));

        group.finish();
    }
}

fn bench_from_array(c: &mut Criterion) {
    {
        let array = [1i32; 1000];
        let mut group = c.benchmark_group(format!("From Array {}", array.len()));

        group.measurement_time(Duration::from_secs(10));
        group.sample_size(5000);

        group.bench_function("Fillet", |b| b.iter(|| black_box(Fillet::from(array))));

        group.bench_function("Vec", |b| b.iter(|| black_box(Vec::from(array))));

        group.finish();
    }
    {
        let array = [1i32; 100_000];
        let mut group = c.benchmark_group(format!("From Array {}", array.len()));

        group.measurement_time(Duration::from_secs(10));
        group.sample_size(1000);

        group.bench_function("Fillet", |b| b.iter(|| black_box(Fillet::from(array))));

        group.bench_function("Vec", |b| b.iter(|| black_box(Vec::from(array))));

        group.finish();
    }
}

fn bench_from_iter_random(c: &mut Criterion) {
    let mut rng = rng();
    for n in [32, 8192] {
        let mut gendex = || (rng.random_range(0..n) as usize).saturating_sub(n / 2);
        let mut group = c.benchmark_group(format!(
            "From random range {:?} items long, 50% empty",
            0..(n / 2)
        ));
        group.measurement_time(Duration::from_secs(10));
        group.sample_size(2000);

        group.bench_function("Fillet", |b| {
            b.iter_batched(
                || 0..gendex(),
                |range| {
                    black_box(Fillet::from_iter(range));
                },
                BatchSize::SmallInput,
            )
        });

        group.bench_function("Vec", |b| {
            b.iter_batched(
                || 0..gendex(),
                |range| {
                    black_box(Vec::from_iter(range));
                },
                BatchSize::SmallInput,
            )
        });

        group.finish();
    }
}

fn bench_one_or_none(c: &mut Criterion) {
    let mut rng = rng();
    for n in [2, 10, 100] {
        let mut group = c.benchmark_group(format!("One or None 1:{n} chance of single item"));
        group.measurement_time(Duration::from_secs(3));
        group.sample_size(10000);
        let mut test_iter = || {
            rng.random_ratio(1, n)
                .then_some(unsafe { NonZeroU128::new_unchecked(420) })
        };

        group.bench_function("Fillet::from_iter", |b| {
            b.iter_batched(
                &mut test_iter,
                |data| {
                    black_box(Fillet::from_iter(data));
                },
                BatchSize::SmallInput,
            )
        });

        group.bench_function("Fillet::from", |b| {
            b.iter_batched(
                &mut test_iter,
                |data| {
                    black_box(Fillet::from(data));
                },
                BatchSize::SmallInput,
            )
        });

        group.bench_function("Vec::from_iter", |b| {
            b.iter_batched(
                &mut test_iter,
                |data| {
                    black_box(Vec::from_iter(data));
                },
                BatchSize::SmallInput,
            )
        });

        group.finish();
    }
}

fn bench_recursive_reverse(c: &mut Criterion) {
    let mut group = c.benchmark_group("Recursive build, reverse, and count trie depth 6");
    group.sampling_mode(SamplingMode::Flat);
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(75);

    // Define type aliases for Fillet-based tries at each level
    type FTrie0 = Fillet<()>;
    type FTrie1 = Fillet<FTrie0>;
    type FTrie2 = Fillet<FTrie1>;
    type FTrie3 = Fillet<FTrie2>;
    type FTrie4 = Fillet<FTrie3>;
    type FTrie5 = Fillet<FTrie4>;
    type FTrie6 = Fillet<FTrie5>;

    // Define type aliases for Vec-based tries at each level
    type VTrie0 = Vec<()>;
    type VTrie1 = Vec<VTrie0>;
    type VTrie2 = Vec<VTrie1>;
    type VTrie3 = Vec<VTrie2>;
    type VTrie4 = Vec<VTrie3>;
    type VTrie5 = Vec<VTrie4>;
    type VTrie6 = Vec<VTrie5>;

    const INIT_FANOUT: usize = 15;

    fn rndlrng() -> impl Iterator<Item = ()> {
        let mut irng = rng();
        core::iter::repeat(()).take_while(move |_| !irng.random_ratio(1, 15))
    }

    // Build functions for Fillet tries
    fn build_ftrie1() -> FTrie1 {
        rndlrng().map(|_| Fillet::EMPTY).collect()
    }

    fn build_ftrie2() -> FTrie2 {
        rndlrng().map(|_| build_ftrie1()).collect()
    }

    fn build_ftrie3() -> FTrie3 {
        rndlrng().map(|_| build_ftrie2()).collect()
    }

    fn build_ftrie4() -> FTrie4 {
        rndlrng().map(|_| build_ftrie3()).collect()
    }

    fn build_ftrie5() -> FTrie5 {
        rndlrng().map(|_| build_ftrie4()).collect()
    }

    fn build_ftrie6() -> FTrie6 {
        core::iter::repeat_n((), INIT_FANOUT)
            .map(|_| build_ftrie5())
            .collect()
    }

    // Build functions for Vec tries
    fn build_vtrie1() -> VTrie1 {
        rndlrng().map(|_| vec![]).collect()
    }

    fn build_vtrie2() -> VTrie2 {
        rndlrng().map(|_| build_vtrie1()).collect()
    }

    fn build_vtrie3() -> VTrie3 {
        rndlrng().map(|_| build_vtrie2()).collect()
    }

    fn build_vtrie4() -> VTrie4 {
        rndlrng().map(|_| build_vtrie3()).collect()
    }

    fn build_vtrie5() -> VTrie5 {
        rndlrng().map(|_| build_vtrie4()).collect()
    }

    fn build_vtrie6() -> VTrie6 {
        core::iter::repeat_n((), INIT_FANOUT)
            .map(|_| build_vtrie5())
            .collect()
    }

    // Reverse functions for Fillet tries (recursive reversal)
    fn reverse_ftrie0(trie: FTrie0) -> FTrie0 {
        trie.into_iter().rev().collect()
    }

    fn reverse_ftrie1(trie: FTrie1) -> FTrie1 {
        trie.into_iter().rev().map(reverse_ftrie0).collect()
    }

    fn reverse_ftrie2(trie: FTrie2) -> FTrie2 {
        trie.into_iter().rev().map(reverse_ftrie1).collect()
    }

    fn reverse_ftrie3(trie: FTrie3) -> FTrie3 {
        trie.into_iter().rev().map(reverse_ftrie2).collect()
    }

    fn reverse_ftrie4(trie: FTrie4) -> FTrie4 {
        trie.into_iter().rev().map(reverse_ftrie3).collect()
    }

    fn reverse_ftrie5(trie: FTrie5) -> FTrie5 {
        trie.into_iter().rev().map(reverse_ftrie4).collect()
    }

    fn reverse_ftrie6(trie: FTrie6) -> FTrie6 {
        trie.into_iter().rev().map(reverse_ftrie5).collect()
    }

    // Reverse functions for Vec tries (recursive reversal)
    fn reverse_vtrie0(trie: VTrie0) -> VTrie0 {
        trie.into_iter().rev().collect()
    }

    fn reverse_vtrie1(trie: VTrie1) -> VTrie1 {
        trie.into_iter().rev().map(reverse_vtrie0).collect()
    }

    fn reverse_vtrie2(trie: VTrie2) -> VTrie2 {
        trie.into_iter().rev().map(reverse_vtrie1).collect()
    }

    fn reverse_vtrie3(trie: VTrie3) -> VTrie3 {
        trie.into_iter().rev().map(reverse_vtrie2).collect()
    }

    fn reverse_vtrie4(trie: VTrie4) -> VTrie4 {
        trie.into_iter().rev().map(reverse_vtrie3).collect()
    }

    fn reverse_vtrie5(trie: VTrie5) -> VTrie5 {
        trie.into_iter().rev().map(reverse_vtrie4).collect()
    }

    fn reverse_vtrie6(trie: VTrie6) -> VTrie6 {
        trie.into_iter().rev().map(reverse_vtrie5).collect()
    }

    group.bench_function("Fillet", |b| {
        b.iter_batched(
            || (),
            |_| {
                black_box(
                    reverse_ftrie6(build_ftrie6())
                        .into_iter()
                        .flatten()
                        .flatten()
                        .flatten()
                        .flatten()
                        .flatten()
                        .flatten()
                        .count(),
                )
            },
            BatchSize::LargeInput,
        )
    });

    group.bench_function("Vec", |b| {
        b.iter_batched(
            || (),
            |_| {
                black_box(
                    reverse_vtrie6(build_vtrie6())
                        .into_iter()
                        .flatten()
                        .flatten()
                        .flatten()
                        .flatten()
                        .flatten()
                        .flatten()
                        .count(),
                )
            },
            BatchSize::LargeInput,
        )
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_from_iter_large_unknown,
    bench_from_slice,
    bench_from_array,
    bench_from_iter_random,
    bench_one_or_none,
    bench_recursive_reverse,
);
criterion_main!(benches);
