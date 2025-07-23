#![no_std]

//! Fillet is a thin pointer based contiguous collection.
//!
//! [`Fillet`] is null/zero when empty, and is especially well suited to scenarios where most
//! collections are empty, and you are storing references to many of them.
//!
//! `Fillet` has specialized handling of Zero-Sized Types (ZSTs).
//! When `T` is zero-sized, `Fillet<T>` is a `usize` length, and [`Deref`] and [`Drop`] behaviors are
//! handled with dangling pointers.
//!
//! `Fillet` is guaranteed to be pointer-sized, and is always zero (either [`None`] of [`NonNull`] or
//! `0usize`) when empty.
//!
//! `Fillet` does not reserve capacity, and this means that repeated [`push`] can be slow since it
//! always calls the allocator, and always computes a [`Layout`].
//! `Fillet`'s [`Extend`] and [`FromIterator`] use amortized growth and track capacity internally, so
//! are roughly equivalent in performance to the same implementations on [`Vec`].
//!
//! ## Examples
//! ### Basic Usage
//!
//! ```
//! use fillet::Fillet;
//!
//! let mut f: Fillet<_> = (1..=3).collect();
//! assert_eq!(f.len(), 3);
//! assert_eq!(*f, [1, 2, 3]);
//!
//! f.push(4);
//! assert_eq!(*f, [1, 2, 3, 4]);
//!
//! f.truncate(2);
//! assert_eq!(*f, [1, 2]);
//! ```
//!
//! ### Zero-Sized Types
//!
//! ```
//! use fillet::Fillet;
//! use core::iter::repeat_n;
//! use core::mem::size_of;
//!
//! let mut f: Fillet<()> = repeat_n((), 5).collect();
//! assert_eq!(f.len(), 5);
//!
//! // No heap allocation
//! f.push(());
//! assert_eq!(f.len(), 6);
//! assert_eq!(size_of::<Fillet<()>>(), size_of::<usize>());
//! ```
//!
//! ### Extending from Within
//!
//! ```
//! use fillet::Fillet;
//!
//! let mut f = Fillet::from([1, 2]);
//! f.extend_from_within(..);
//! assert_eq!(*f, [1, 2, 1, 2]);
//! ```
//!
//!
//! [`Extend`]: Fillet::extend
//! [`FromIterator`]: Fillet::from_iter
//! [`push`]: Fillet::push
//! [`Vec`]: alloc::vec::Vec

extern crate alloc;
use alloc::alloc::{alloc, dealloc, handle_alloc_error, realloc};
use alloc::boxed::Box;
use core::alloc::Layout;
use core::borrow::{Borrow, BorrowMut};
use core::clone::Clone;
use core::fmt::{self, Debug, Display, Formatter};
use core::hash::{Hash, Hasher};
use core::hint::unreachable_unchecked;
use core::iter::{
    self, DoubleEndedIterator, ExactSizeIterator, FromIterator, IntoIterator, Iterator,
};
use core::marker::PhantomData;
use core::mem::{ManuallyDrop, MaybeUninit, align_of, size_of};
use core::num::NonZeroUsize;
use core::ops::{Bound, Deref, DerefMut, RangeBounds};
use core::ptr::{self, NonNull};
use core::slice;

/// A thin pointer based contiguous collection.
///
/// When `T` is zero-sized, there is no heap allocation made, and [`Drop`] is handled by length.
///
/// For other types `T` when not empty, a heap allocation is made which starts with the length,
/// followed by an aligned array `[T; len]`.
///
/// Because `usize` and `Option<NonNull<u8>>` have the same size and alignment, `Fillet` is
/// always pointer-sized.
pub union Fillet<T> {
    /// For non ZSTs, this is either `None` when empty or a [`NonNull`] pointing to
    /// a heap allocation `(usize, [T; len])`.
    ptr: Option<NonNull<u8>>,
    /// For ZSTs, this is the length of the `Fillet`.
    len: usize,
    _marker: PhantomData<T>,
}

impl<T> Fillet<T> {
    /// Regardless of `T`, all empty [`Fillet`]s have this value, which is a pointer-sized zero.
    // The value is technically the same for ZSTs and non-ZSTs, but written this way for strictness.
    pub const EMPTY: Self = if size_of::<T>() == 0 {
        Self { len: 0 }
    } else {
        Self { ptr: None }
    };

    /// Offset of the first `T` in the heap allocation.
    ///
    /// Alignment of `[T]` is the same as alignment of `T`, and alignment of `usize` is its size.
    ///
    /// This is irrelevant for ZSTs so can be incorrect.
    const DATA_OFFSET: usize = if size_of::<usize>() > align_of::<T>() {
        size_of::<usize>()
    } else {
        align_of::<T>()
    };

    /// The maximum number of elements this `Fillet` can hold without causing a layout overflow.
    ///
    /// rustc itself is more restrictive than this when it comes to arrays, because LLVM expresses
    /// size in bits rather than bytes, so only supports objects (1 << 61) large.
    ///
    /// Realistically, you won't be able to initialize anything this large on a real computer anyway.
    pub const MAX_LEN: usize = if size_of::<T>() == 0 {
        // Zero-Sized Types are not stored, and their offsets are all 0.
        usize::MAX
    } else {
        (isize::MAX as usize).saturating_sub(Self::DATA_OFFSET) / size_of::<T>()
    };
}

impl<T> Default for Fillet<T> {
    fn default() -> Self {
        Self::EMPTY
    }
}

impl<T> Fillet<T> {
    /// Return the number of elements in the `Fillet`.
    #[inline(always)]
    pub const fn len(&self) -> usize {
        // ZSTs do not have an allocation.
        if size_of::<T>() == 0 {
            return unsafe { self.len };
        }

        match unsafe { self.ptr } {
            None => 0,
            Some(ptr) => unsafe { ptr.cast::<usize>().read() },
        }
    }

    /// Return `true` if the `Fillet` contains no elements.
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        if size_of::<T>() == 0 {
            return self.len() == 0;
        }

        unsafe { self.ptr }.is_none()
    }
}

impl<T> Fillet<T> {
    /// Computes the layout for the heap allocation `(usize, [T; len])` without checking len.
    ///
    /// # Safety
    /// Caller is responsible for validating that `len` <= [`Self::MAX_LEN`], otherwise this can
    /// cause undefined behavior.
    #[inline(always)]
    const unsafe fn compute_layout_unchecked(len: usize) -> Layout {
        if size_of::<T>() == 0 {
            unreachable!()
        }

        let array_size = size_of::<T>() * len;
        let overall_size = Self::DATA_OFFSET + array_size;
        // Because this is the greater of `size_of::<usize>()` and `align_of::<T>()`.
        let overall_align = Self::DATA_OFFSET;

        // SAFETY: `len` should be validated against [`Self::MAX_LEN`] by now.
        unsafe { Layout::from_size_align_unchecked(overall_size, overall_align) }
    }

    /// Computes the layout for heap allocation: `(usize, [T; M])`.
    ///
    /// # Panics
    /// This will panic when `len` exceeds [`Self::MAX_LEN`].
    #[inline(always)]
    fn compute_layout(len: usize) -> Layout {
        if len <= Self::MAX_LEN {
            unsafe { Self::compute_layout_unchecked(len) }
        } else {
            // MAX_LEN is defined such that a layout can be computed even for objects which rustc
            // would refuse to express due to LLVM limitations.
            panic!(
                "{len} elements > MAX_LEN = {}.\nRequested Fillet larger than isize::MAX bytes.",
                Self::MAX_LEN
            );
        }
    }
}

impl<T> Fillet<T> {
    /// Allocate an uninitialized [`Fillet`] of a given length.
    ///
    /// # Safety
    /// Caller is responsible for initializing `[T; len]` at the returned pointer, and for discarding
    /// the pointer, since it is only valid for writing the uninitialized array, and can become
    /// invalid if the heap array is reallocated.
    unsafe fn alloc_nonempty(len: NonZeroUsize) -> (Fillet<T>, NonNull<MaybeUninit<T>>) {
        // ZSTs do not have an allocation.
        if size_of::<T>() == 0 {
            return (
                Fillet { len: len.get() },
                NonNull::<MaybeUninit<T>>::dangling(),
            );
        }

        let len = len.get();
        let layout = Self::compute_layout(len);
        unsafe {
            let nn = NonNull::new(alloc(layout));
            if let Some(nn) = nn {
                nn.cast::<usize>().write(len);
                (
                    Fillet { ptr: Some(nn) },
                    nn.byte_add(Self::DATA_OFFSET).cast::<MaybeUninit<T>>(),
                )
            } else {
                handle_alloc_error(layout);
            }
        }
    }

    /// Deallocate and empty without dropping.
    ///
    /// # Safety
    /// Caller is responsible for the drop obligations of `T`, if any.
    unsafe fn dealloc_nonempty(&mut self) {
        let len = self.len();
        if size_of::<T>() != 0 {
            unsafe {
                // SAFETY: Length was checked when Fillet was allocated.
                let old_layout = Self::compute_layout_unchecked(len);
                dealloc(self.ptr.unwrap_unchecked().as_ptr(), old_layout);
            }
            self.ptr = None;
        } else {
            self.len = 0;
        }
    }

    /// Grow a known non-empty `Fillet` to `new_len`, returning a mutable slice aliasing
    /// the uninitialized elements.
    ///
    /// Assumes the `Fillet` is valid and non-empty and that `new_len >= self.len()`.
    ///
    /// # Panics
    /// Panics if `new_len` > [`Self::MAX_LEN`] or if allocation fails.
    ///
    /// # Safety
    /// It is the caller's job to fill the returned alias slice with initialized `T`.
    #[must_use]
    unsafe fn grow_nonempty(&mut self, new_len: usize) -> &mut [MaybeUninit<T>] {
        // ZSTs do not have an allocation.
        if size_of::<T>() == 0 {
            let old_len = self.len();
            self.len = new_len;
            // SAFETY: Behavior of this slice is standard for ZSTs.
            return unsafe { slice::from_raw_parts_mut(ptr::dangling_mut(), new_len - old_len) };
        }

        unsafe {
            let old_ptr = self.ptr.unwrap_unchecked().as_ptr();
            let old_len = old_ptr.cast::<usize>().read();

            // SAFETY: Layout was computable when the `Fillet` was created.
            let old_layout = Self::compute_layout_unchecked(old_len);

            let new_layout = Self::compute_layout(new_len);
            // SAFETY: Old layout should exist and match.
            self.ptr = NonNull::new(realloc(old_ptr, old_layout, new_layout.size()));

            if let Some(nn) = self.ptr {
                (nn.as_ptr() as *mut usize).write(new_len);
                let data_ptr = nn
                    .as_ptr()
                    .byte_add(Self::DATA_OFFSET)
                    .cast::<MaybeUninit<T>>();
                slice::from_raw_parts_mut(data_ptr.add(old_len), new_len - old_len)
            } else {
                handle_alloc_error(new_layout);
            }
        }
    }

    /// Grow to `new_len`, returning a mutable slice aliasing the uninitialized elements.
    ///
    /// Assumes `new_len > self.len()`.
    ///
    /// # Panics
    /// Panics if `new_len` > [`Self::MAX_LEN`] or if allocation fails.
    ///
    /// # Safety
    /// It is the caller's job to fill the returned alias slice with initialized `T`.
    #[must_use]
    #[inline(always)]
    unsafe fn grow(&mut self, new_len: NonZeroUsize) -> &mut [MaybeUninit<T>] {
        // ZSTs do not have an allocation.
        if size_of::<T>() == 0 {
            let old_len = self.len();
            self.len = new_len.get();
            // SAFETY: Behavior of this slice is standard for ZSTs.
            return unsafe {
                slice::from_raw_parts_mut(
                    ptr::dangling_mut::<MaybeUninit<T>>().add(old_len),
                    new_len.get().saturating_sub(old_len),
                )
            };
        }

        unsafe {
            if !self.is_empty() {
                self.grow_nonempty(new_len.get())
            } else {
                let (f, uninit) = Self::alloc_nonempty(new_len);
                let f = ManuallyDrop::new(f);
                self.ptr = f.ptr;
                slice::from_raw_parts_mut(uninit.as_ptr(), new_len.get())
            }
        }
    }

    /// Shrink a known non-empty `Fillet` to `new_len`, dropping excess elements.
    ///
    /// Assumes the `Fillet` is valid and non-empty, and that `new_len < self.len()`.
    ///
    /// # Panics
    /// Panics if allocation fails.
    /// The new heap layout is assumed to be computable since it is smaller.
    ///
    /// # Safety
    /// Assumes the new heap is smaller, and that the excess elements can be dropped.
    unsafe fn shrink_nonempty(&mut self, new_len: usize) {
        // ZSTs do not have an allocation.
        if size_of::<T>() == 0 {
            let old_len = self.len();
            self.len = new_len;
            // SAFETY: drop_in_place is safe on a dangling pointer for ZSTs
            unsafe {
                ptr::drop_in_place(slice::from_raw_parts_mut(
                    ptr::dangling_mut::<T>().add(new_len),
                    old_len - new_len,
                ));
            }
            return;
        }

        unsafe {
            let old_ptr = self.ptr.unwrap_unchecked().as_ptr();
            let old_len = old_ptr.cast::<usize>().read();

            // SAFETY: Layout was computable when the `Fillet` was created.
            let old_layout = Self::compute_layout_unchecked(old_len);

            // SAFETY: Caller responsible for ensuring `new_len < self.len()`.
            ptr::drop_in_place(slice::from_raw_parts_mut(
                old_ptr.byte_add(Self::DATA_OFFSET).cast::<T>().add(new_len),
                old_len - new_len,
            ));

            if new_len != 0 {
                // SAFETY: If the old array layout was computable, then a shorter one is too.
                let new_layout = Self::compute_layout_unchecked(new_len);

                self.ptr = NonNull::new(realloc(old_ptr, old_layout, new_layout.size()));
                if let Some(nn) = self.ptr {
                    (nn.as_ptr() as *mut usize).write(new_len);
                } else {
                    handle_alloc_error(new_layout);
                }
            } else {
                dealloc(old_ptr, old_layout);
                self.ptr = None;
            }
        }
    }

    /// Shrinks a known non-empty `Fillet` to `new_len`, without dropping excess elements.
    ///
    /// Assumes the `Fillet` is valid and non-empty, that excess elements are uninitialized,
    /// and that `new_len < self.len()`.
    ///
    /// # Panics
    /// Panics if allocation fails.
    /// The new heap layout is assumed to be computable since it is smaller.
    ///
    /// # Safety
    /// Assumes the new heap is smaller, and that the excess elements don't need to be dropped.
    unsafe fn shrink_uninit_nonempty(&mut self, new_len: NonZeroUsize) {
        // ZSTs do not have an allocation.
        if size_of::<T>() == 0 {
            self.len = new_len.get();
            return;
        }

        unsafe {
            let old_ptr = self.ptr.unwrap_unchecked().as_ptr();
            let old_len = old_ptr.cast::<usize>().read();

            // SAFETY: Layout was computable when the `Fillet` was created.
            let old_layout = Self::compute_layout_unchecked(old_len);
            // SAFETY: If the old array layout was computable, then a shorter one is too.
            let new_layout = Self::compute_layout_unchecked(new_len.get());

            self.ptr = NonNull::new(realloc(old_ptr, old_layout, new_layout.size()));
            if let Some(nn) = self.ptr {
                nn.cast::<usize>().write(new_len.get());
            } else {
                handle_alloc_error(new_layout);
            }
        }
    }
}

impl<T> Drop for Fillet<T> {
    fn drop(&mut self) {
        // ZSTs do not have an allocation.
        if size_of::<T>() == 0 {
            let len = self.len();
            if len != 0 {
                // SAFETY: drop_in_place is safe on a dangling pointer for ZSTs
                unsafe {
                    ptr::drop_in_place(slice::from_raw_parts_mut(ptr::dangling_mut::<T>(), len));
                }
            }
            return;
        }

        let len = self.len();
        if len != 0 {
            unsafe {
                let ptr = self.ptr.unwrap_unchecked().as_ptr();
                // SAFETY: Layout was computable when the `Fillet` was created.
                let layout = Fillet::<T>::compute_layout_unchecked(len);
                ptr::drop_in_place(ptr::slice_from_raw_parts_mut(
                    ptr.byte_add(Self::DATA_OFFSET).cast::<T>(),
                    len,
                ));
                dealloc(ptr, layout);
            }
        }
    }
}

// SAFETY: Fillet owns [T], so is Send/Sync as long as T is.
unsafe impl<T: Send> Send for Fillet<T> {}
unsafe impl<T: Sync> Sync for Fillet<T> {}

impl<T> Fillet<T> {
    /// Extract a slice over the `Fillet` heap array, or a dangling slice when empty.
    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        // ZSTs do not have an allocation.
        if size_of::<T>() == 0 {
            return unsafe { slice::from_raw_parts(ptr::dangling::<T>(), self.len()) };
        }

        unsafe {
            match self.ptr {
                None => slice::from_raw_parts(ptr::dangling::<T>(), 0),
                Some(ptr) => slice::from_raw_parts(
                    ptr.byte_add(Self::DATA_OFFSET).cast::<T>().as_ptr(),
                    ptr.cast::<usize>().read(),
                ),
            }
        }
    }

    /// Extract a mutable slice over the `Fillet` heap array, or a dangling slice when empty.
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // ZSTs do not have an allocation.
        if size_of::<T>() == 0 {
            return unsafe { slice::from_raw_parts_mut(ptr::dangling_mut(), self.len()) };
        }

        unsafe {
            match self.ptr {
                None => slice::from_raw_parts_mut(ptr::dangling_mut(), 0),
                Some(ptr) => slice::from_raw_parts_mut(
                    ptr.byte_add(Self::DATA_OFFSET).cast::<T>().as_ptr(),
                    ptr.cast::<usize>().read(),
                ),
            }
        }
    }
}

impl<T> Deref for Fillet<T> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> DerefMut for Fillet<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T> AsRef<[T]> for Fillet<T> {
    #[inline(always)]
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> AsMut<[T]> for Fillet<T> {
    #[inline(always)]
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T> Borrow<[T]> for Fillet<T> {
    fn borrow(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T> BorrowMut<[T]> for Fillet<T> {
    fn borrow_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T: PartialEq> PartialEq for Fillet<T> {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        if size_of::<T>() == 0 {
            self.len() == other.len() || **self == **other
        } else {
            unsafe { self.ptr == other.ptr || **self == **other }
        }
    }
}

impl<T: Eq> Eq for Fillet<T> {}

impl<T: Hash> Hash for Fillet<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.deref().hash(state)
    }
}

impl<T> Fillet<T> {
    /// Push one element to the end of the `Fillet` — not recommended.
    ///
    /// If you could conceivably push more than once, use [`extend`] instead.
    ///
    /// [`extend`]: Fillet::extend
    #[inline(always)]
    pub fn push(&mut self, v: T) {
        self.extend(iter::once(v));
    }

    /// Remove and return the last element, or [`None`] if it is empty.
    pub fn pop(&mut self) -> Option<T> {
        // ZSTs do not have an allocation.
        if size_of::<T>() == 0 {
            let old_len = self.len();
            return if old_len > 0 {
                unsafe {
                    self.len -= 1;
                    Some(ptr::read(ptr::dangling::<T>().add(old_len - 1)))
                }
            } else {
                None
            };
        }

        let old_ptr = unsafe { self.ptr }?.as_ptr();

        let len = unsafe { old_ptr.cast::<usize>().read() };
        let new_len = len - 1;

        // SAFETY: Layout was computable when the `Fillet` was created.
        let old_layout = unsafe { Self::compute_layout_unchecked(len) };
        // Read the last element before shrinking
        let item = unsafe {
            let data_ptr = old_ptr.byte_add(Self::DATA_OFFSET).cast::<T>();
            ptr::read(data_ptr.add(new_len))
        };

        self.ptr = unsafe {
            if new_len != 0 {
                // SAFETY: If longer array Layout was computable, then a shorter layout is too.
                let new_layout = Self::compute_layout_unchecked(new_len);

                // SAFETY: Old layout should exist and match.
                let nn = NonNull::new(realloc(old_ptr, old_layout, new_layout.size()));
                if let Some(nn) = nn {
                    nn.cast::<usize>().write(new_len);
                } else {
                    handle_alloc_error(new_layout);
                }

                nn
            } else {
                dealloc(old_ptr, old_layout);
                None
            }
        };

        Some(item)
    }

    /// Shorten the `Fillet`, keeping the first `len` elements and dropping the rest.
    ///
    /// If `len` is greater than or equal to the fillet's current length, this has no effect.
    ///
    /// If there is a heap allocation for this `Fillet`, it will be reallocated to fit the new
    /// length, or deallocated if the new length is `0`.
    #[inline(always)]
    pub fn truncate(&mut self, len: usize) {
        if self.len() > len {
            unsafe {
                self.shrink_nonempty(len);
            }
        }
    }

    /// Clear the `Fillet`, dropping all values.
    ///
    /// If there is a heap allocation for this `Fillet`, it will be deallocated.
    #[inline(always)]
    pub fn clear(&mut self) {
        self.truncate(0);
    }

    /// Retain only the elements for which predicate `f` is true.
    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&T) -> bool,
    {
        let old_len = self.len();
        if old_len == 0 {
            return;
        }
        let s = self.as_mut_ptr();
        let mut dst = 0;
        for src in 0..old_len {
            unsafe {
                let r = f(&*s.add(src));
                if r {
                    if src != dst {
                        s.add(dst).write(ptr::read(s.add(src)));
                    }
                    dst += 1;
                } else {
                    drop(s.add(src).read());
                }
            }
        }

        if dst == old_len {
            return;
        }

        unsafe {
            if let Some(len) = NonZeroUsize::new(dst) {
                self.shrink_uninit_nonempty(len);
            } else {
                self.dealloc_nonempty();
            }
        }
    }
}

impl<T: Clone> Fillet<T> {
    /// Given a range `src`, clones elements in that range and appends them to the end.
    ///
    /// `src` is bounded within the length of the `Fillet`, and nothing happens if it is invalid.
    pub fn extend_from_within<R>(&mut self, src: R)
    where
        R: RangeBounds<usize>,
    {
        let old_len = self.len();
        let end = match src.end_bound() {
            Bound::Included(end) => end.saturating_add(1).min(old_len),
            Bound::Excluded(&end) => end.min(old_len),
            Bound::Unbounded => old_len,
        };
        let start = match src.start_bound() {
            Bound::Included(&start) => start.min(end),
            Bound::Excluded(start) => start.saturating_add(1).min(end),
            Bound::Unbounded => 0,
        };
        let range = start..end;

        if !range.is_empty() {
            if size_of::<T>() == 0 {
                self.len = old_len + range.len();
                let p = self.as_mut_ptr();
                for (i, isrc) in range.enumerate() {
                    unsafe { p.add(old_len + i).write(ptr::read(p.add(isrc))) };
                }
                return;
            }

            // SAFETY: Non-ZST non-empty, will have allocation after `grow`.
            unsafe {
                let new_len = old_len + range.len();
                // SAFETY: Since by definition the range can't be longer than the `Fillet`,
                //         if the range is not empty then the `Fillet` is also not empty.
                let uninit = self.grow_nonempty(new_len);
                // Need to consume uninit into a pointer so that it isn't mutably borrowed
                // when we go to read the source range from the initialized portion.
                let dst = uninit.as_mut_ptr();
                // SAFETY: The range is already bounds checked in the heap array.
                let src = self
                    .ptr
                    .unwrap_unchecked()
                    .byte_add(Self::DATA_OFFSET)
                    .cast::<T>();
                for i in range {
                    dst.add(i)
                        .write(MaybeUninit::new(src.add(i).read().clone()));
                }
            }
        }
    }
}

impl<T> Extend<T> for Fillet<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        // ZSTs do not have an allocation.
        if size_of::<T>() == 0 {
            let added = iter.map(ManuallyDrop::new).count();
            unsafe {
                self.len += added;
            }
            return;
        }

        let initial_len = self.len();

        let (lower, upper_opt) = iter.size_hint();

        // Exact size iterator.
        if upper_opt.is_some_and(|u| u == lower) {
            if lower != 0 {
                let uninit = unsafe { self.grow(NonZeroUsize::new_unchecked(initial_len + lower)) };
                for (i, item) in iter.enumerate() {
                    uninit[i].write(item);
                }
            }
            return;
        }

        let mut capacity = initial_len;
        let mut growth = upper_opt.unwrap_or(lower.max(4));
        capacity += growth;
        let mut remaining_uninit = unsafe { self.grow(NonZeroUsize::new_unchecked(capacity)) };
        let mut len = initial_len;
        for item in iter {
            if remaining_uninit.is_empty() {
                growth *= 2;
                capacity += growth;
                remaining_uninit = unsafe { self.grow_nonempty(capacity) };
            }
            remaining_uninit[0].write(item);
            remaining_uninit = &mut remaining_uninit[1..];
            len += 1;
        }

        if !remaining_uninit.is_empty() {
            unsafe { self.shrink_uninit_nonempty(NonZeroUsize::new_unchecked(len)) };
        }
    }
}

impl<T> FromIterator<T> for Fillet<T> {
    #[inline(always)]
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut f = Self::EMPTY;
        f.extend(iter);
        f
    }
}

impl<T, const M: usize> From<[T; M]> for Fillet<T> {
    #[inline]
    fn from(array: [T; M]) -> Self {
        // SAFETY: Arrays in rustc are never within an order of magnitude of MAX_LEN because LLVM
        //         expresses object size as a 64-bit integer number of bits, even if T is 1 byte.
        if M > Self::MAX_LEN {
            unsafe {
                unreachable_unchecked();
            }
        }

        Self::from_iter(array)
    }
}

impl<T: Clone> From<&[T]> for Fillet<T> {
    #[inline(always)]
    fn from(slice: &[T]) -> Self {
        slice.iter().cloned().collect()
    }
}

impl<T> From<Box<[T]>> for Fillet<T> {
    #[inline(always)]
    fn from(boxed: Box<[T]>) -> Self {
        Self::from_iter(boxed)
    }
}

impl<T> From<Option<T>> for Fillet<T> {
    #[inline(always)]
    fn from(o: Option<T>) -> Self {
        o.map_or(Self::EMPTY, Self::from_one)
    }
}

impl<T> Fillet<T> {
    #[inline(always)]
    pub fn from_one(v: T) -> Self {
        // SAFETY: `v` is moved, drop is handled by Fillet.
        unsafe {
            let v = ManuallyDrop::new(v);
            let (f, uninit) = Self::alloc_nonempty(NonZeroUsize::new_unchecked(1));
            uninit.write(MaybeUninit::new(ptr::read(&*v)));
            f
        }
    }
}

impl<T: Debug> Debug for Fillet<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(self.deref(), f)
    }
}

impl<T: Display> Display for Fillet<T>
where
    [T]: Display,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(self.deref(), f)
    }
}

impl<T: Clone> Clone for Fillet<T> {
    fn clone(&self) -> Self {
        Self::from(self.deref())
    }
}

/// Inner data for [`FilletIntoIter`].
///
/// For ZSTs this is a `usize` length. For non-ZSTs, it is a pointer to an allocation of T.
union FilletIntoIterInner<T> {
    ptr: Option<NonNull<T>>,
    len: usize,
}

/// A double-ended iterator over [`Fillet<T>`] that consumes and drops the collection.
pub struct FilletIntoIter<T> {
    inner: FilletIntoIterInner<T>,
    start: usize,
    end: usize,
    _marker: PhantomData<T>,
}

impl<T> ExactSizeIterator for FilletIntoIter<T> {
    fn len(&self) -> usize {
        self.end - self.start
    }
}

impl<T> Iterator for FilletIntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start >= self.end {
            return None;
        }

        // ZSTs do not have an allocation.
        if size_of::<T>() == 0 {
            let start = self.start;
            self.start += 1;
            return unsafe { Some(ptr::dangling::<T>().add(start).read()) };
        }

        // SAFETY:
        // - self.start < self.end <= len, so index in bounds.
        // - pointer known non-null since this is not empty.
        // - Data initialized, read transfers ownership.
        unsafe {
            let data_ptr = self.inner.ptr.unwrap_unchecked().as_ptr();
            let item = data_ptr.add(self.start).read();
            self.start += 1;
            Some(item)
        }
    }
}

impl<T> DoubleEndedIterator for FilletIntoIter<T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start >= self.end {
            return None;
        }
        self.end -= 1;

        // ZSTs do not have an allocation.
        if size_of::<T>() == 0 {
            return unsafe { Some(ptr::dangling::<T>().add(self.end).read()) };
        }

        // SAFETY: Similar to next, but from end.
        unsafe {
            // SAFETY: Layout was computable when the `Fillet` was created.
            let data_ptr = self.inner.ptr.unwrap_unchecked().as_ptr();
            let item = data_ptr.add(self.end).read();
            Some(item)
        }
    }
}

impl<T> Drop for FilletIntoIter<T> {
    fn drop(&mut self) {
        // ZSTs do not have an allocation.
        if size_of::<T>() == 0 {
            let len = self.len();
            if len != 0 {
                // SAFETY: drop_in_place is safe on a dangling pointer for ZSTs
                unsafe {
                    ptr::drop_in_place(slice::from_raw_parts_mut(
                        ptr::dangling_mut::<T>().add(self.start),
                        len,
                    ));
                }
            }
            return;
        }

        unsafe {
            if let Some(data_ptr) = self.inner.ptr.map(NonNull::as_ptr) {
                let ptr = (data_ptr as *mut u8).byte_sub(Fillet::<T>::DATA_OFFSET);
                let len = ptr.cast::<usize>().read();
                // SAFETY: Layout was computable when the `Fillet` was created.
                let layout = Fillet::<T>::compute_layout_unchecked(len);
                ptr::drop_in_place(ptr::slice_from_raw_parts_mut(
                    data_ptr.add(self.start),
                    self.end - self.start,
                ));
                dealloc(ptr, layout);
            }
        }
    }
}

impl<T> IntoIterator for Fillet<T> {
    type Item = T;
    type IntoIter = FilletIntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        let f = ManuallyDrop::new(self);

        // ZSTs do not have an allocation.
        if size_of::<T>() == 0 {
            let len = f.len();
            return FilletIntoIter {
                inner: FilletIntoIterInner { len },
                start: 0,
                end: len,
                _marker: PhantomData,
            };
        }

        unsafe {
            match f.ptr {
                None => FilletIntoIter {
                    inner: FilletIntoIterInner { ptr: None },
                    start: 0,
                    end: 0,
                    _marker: PhantomData,
                },
                Some(p) => {
                    let len = p.cast::<usize>().read();
                    let data_ptr = p.byte_add(Fillet::<T>::DATA_OFFSET).cast::<T>();
                    FilletIntoIter {
                        inner: FilletIntoIterInner {
                            ptr: Some(data_ptr),
                        },
                        start: 0,
                        end: len,
                        _marker: PhantomData,
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use core::iter::repeat_n;
    use core::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Clone)]
    struct Dropper;
    static DROPS: AtomicUsize = AtomicUsize::new(0);
    impl Drop for Dropper {
        fn drop(&mut self) {
            DROPS.fetch_add(1, Ordering::SeqCst);
        }
    }

    struct CloneDropper(i32);
    impl Clone for CloneDropper {
        fn clone(&self) -> Self {
            Self(self.0)
        }
    }
    impl Drop for CloneDropper {
        fn drop(&mut self) {
            DROPS.fetch_add(1, Ordering::SeqCst);
        }
    }

    /// Construct an empty `Fillet`.
    #[test]
    fn construction_empty() {
        let f: Fillet<u8> = Fillet::EMPTY;
        assert_eq!(f.len(), 0);
        assert!(f.is_empty());
        assert_eq!(&*f, &[]);
    }

    /// Construct from an array.
    #[test]
    fn construction_from_array() {
        let f: Fillet<i32> = [1, 2, 3].into();
        assert_eq!(f.len(), 3);
        assert!(!f.is_empty());
        assert_eq!(*f, [1, 2, 3]);
    }

    /// Construct from an empty array.
    #[test]
    fn construction_from_array_empty() {
        let f: Fillet<i32> = [].into();
        assert_eq!(f.len(), 0);
        assert!(f.is_empty());
        assert_eq!(*f, []);
    }

    /// Construct from an array of ZST.
    #[test]
    fn construction_from_array_zst() {
        let f: Fillet<()> = [(); 5].into();
        assert_eq!(f.len(), 5);
        assert!(!f.is_empty());
    }

    /// Construct from a slice.
    #[test]
    fn construction_from_slice() {
        let slice: &[i32] = &[1, 2, 3];
        let f: Fillet<i32> = slice.into();
        assert_eq!(f.len(), 3);
        assert_eq!(*f, [1, 2, 3]);
    }

    /// Construct from an empty slice.
    #[test]
    fn construction_from_slice_empty() {
        let slice: &[i32] = &[];
        let f: Fillet<i32> = slice.into();
        assert_eq!(f.len(), 0);
        assert_eq!(*f, []);
    }

    /// Construct from a boxed slice.
    #[test]
    fn construction_from_box_slice() {
        let boxed: Box<[i32]> = vec![1, 2, 3].into_boxed_slice();
        let f: Fillet<i32> = boxed.into();
        assert_eq!(f.len(), 3);
        assert_eq!(*f, [1, 2, 3]);
    }

    /// Construct from an empty boxed slice.
    #[test]
    fn construction_from_box_slice_empty() {
        let boxed: Box<[i32]> = vec![].into_boxed_slice();
        assert_eq!(boxed.len(), 0);
        let f: Fillet<i32> = boxed.into();
        assert_eq!(f.len(), 0);
        assert_eq!(*f, []);
    }

    /// Construct from a boxed slice of ZST.
    #[test]
    fn construction_from_box_slice_zst() {
        let boxed: Box<[()]> = vec![(); 5].into_boxed_slice();
        let f: Fillet<()> = boxed.into();
        assert_eq!(f.len(), 5);
    }

    /// Construct from a `Some`.
    #[test]
    fn construction_from_option_some() {
        let o: Option<i32> = Some(42);
        let f: Fillet<i32> = o.into();
        assert_eq!(f.len(), 1);
        assert_eq!(f[0], 42);
    }

    /// Construct from a `None`.
    #[test]
    fn construction_from_option_none() {
        let o: Option<i32> = None;
        let f: Fillet<i32> = o.into();
        assert_eq!(f.len(), 0);
    }

    /// Construct from a single value.
    #[test]
    fn construction_from_one() {
        let f = Fillet::from_one(42i32);
        assert_eq!(f.len(), 1);
        assert_eq!(f[0], 42);
    }

    /// Construct from a simple iterator.
    #[test]
    fn construction_from_iterator() {
        let f: Fillet<i32> = (0..3).collect();
        assert_eq!(f.len(), 3);
        assert_eq!(*f, [0, 1, 2]);
    }

    /// Construct from an empty iterator.
    #[test]
    fn construction_from_iterator_empty() {
        let f: Fillet<i32> = core::iter::empty().collect();
        assert_eq!(f.len(), 0);
        assert_eq!(*f, []);
    }

    /// Construct from an iterator of ZST.
    #[test]
    fn construction_from_iterator_zst() {
        let f: Fillet<()> = repeat_n((), 5).collect();
        assert_eq!(f.len(), 5);
        let slice: &[_] = &f;
        assert_eq!(slice.len(), 5);
        assert_eq!(*f, [(), (), (), (), ()]);
    }

    /// Test 64-byte aligned type.
    #[test]
    fn alignment_large() {
        #[repr(align(64))]
        struct Aligned(u8);
        let f: Fillet<Aligned> = [Aligned(1), Aligned(2)].into();
        assert_eq!(f.len(), 2);
        assert_eq!(f[0].0, 1);
        assert_eq!(f[1].0, 2);
    }

    /// When layout is not computable because of the size of the heap array, panic.
    #[test]
    #[should_panic]
    fn overflow_layout_panic() {
        let _f: Fillet<u128> = repeat_n(0u128, usize::MAX).collect();
    }

    /// Test [`len`] and [`is_empty`].
    ///
    /// [`len`]: Fillet::len
    /// [`is_empty`]: Fillet::is_empty
    #[test]
    fn methods_len_is_empty() {
        let f: Fillet<i32> = [1, 2].into();
        assert_eq!(f.len(), 2);
        assert!(!f.is_empty());
        let e: Fillet<i32> = Fillet::EMPTY;
        assert_eq!(e.len(), 0);
        assert!(e.is_empty());
        let d: Fillet<()> = repeat_n((), 42).collect();
        assert_eq!(d.len(), 42);
        assert!(!d.is_empty());
    }

    /// [`as_ptr`] behaviors.
    ///
    /// [`as_ptr`]: Fillet::as_ptr
    #[test]
    fn methods_as_ptr() {
        let f: Fillet<i32> = [1, 2].into();
        let p = f.as_ptr();
        unsafe {
            assert_eq!(*p, 1);
            assert_eq!(*p.add(1), 2);
        }
        let e: Fillet<i32> = Fillet::EMPTY;
        assert_eq!(e.as_ptr(), ptr::dangling::<i32>());
        let d: Fillet<()> = repeat_n((), 42).collect();
        assert_eq!(d.as_ptr(), ptr::dangling::<()>());
    }

    /// [`as_mut_ptr`] behaviors.
    ///
    /// [`as_mut_ptr`]: Fillet::as_mut_ptr
    #[test]
    fn methods_as_mut_ptr() {
        let mut f: Fillet<i32> = [1, 2].into();
        let p = f.as_mut_ptr();
        unsafe {
            assert_eq!(*p, 1);
            *p = 3;
        }
        assert_eq!(f[0], 3);
    }

    /// [`deref`] and [`deref_mut`] behaviors.
    ///
    /// [`deref`]: Fillet::deref
    /// [`deref_mut`]: Fillet::deref_mut
    #[test]
    fn deref_and_deref_mut() {
        let mut f: Fillet<i32> = [1, 2, 3].into();
        assert_eq!(*f, [1, 2, 3]);
        f[0] = 4;
        assert_eq!(*f, [4, 2, 3]);
        let slice: &mut [i32] = &mut f;
        slice[1] = 5;
        assert_eq!(*f, [4, 5, 3]);
    }

    /// `Clone` behavior.
    #[test]
    fn clone_behavior() {
        let f: Fillet<i32> = [1, 2].into();
        let c = f.clone();
        assert_eq!(*c, [1, 2]);
        assert_eq!(f, c);
    }

    /// `Eq` and `Hash` behavior.
    #[test]
    fn eq_and_hash() {
        extern crate std;
        use std::collections::hash_map::DefaultHasher;
        let f1: Fillet<i32> = [1, 2].into();
        let f2: Fillet<i32> = [1, 2].into();
        let f3: Fillet<i32> = [3].into();
        assert_eq!(f1, f2);
        assert_ne!(f1, f3);
        let mut h1 = DefaultHasher::new();
        f1.hash(&mut h1);
        let mut h2 = DefaultHasher::new();
        f2.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    /// `Debug` formatting.
    #[test]
    fn debug_fmt() {
        use alloc::format;
        let f: Fillet<i32> = [1, 2].into();
        assert_eq!(format!("{f:?}"), "[1, 2]");
    }

    /// Simple [`drop`] behavior.
    ///
    /// [`drop`]: Fillet::drop
    #[test]
    fn drop_counts_basic() {
        DROPS.store(0, Ordering::SeqCst);
        let f: Fillet<Dropper> = (0..3).map(|_| Dropper).collect();
        assert_eq!(DROPS.load(Ordering::SeqCst), 0);
        drop(f);
        assert_eq!(DROPS.load(Ordering::SeqCst), 3);
    }

    /// [`drop`] behavior during construction from an array.
    ///
    /// [`drop`]: Fillet::drop
    #[test]
    fn drop_counts_from_array() {
        DROPS.store(0, Ordering::SeqCst);
        let arr = [Dropper, Dropper];
        let f: Fillet<Dropper> = arr.into();
        assert_eq!(DROPS.load(Ordering::SeqCst), 0);
        drop(f);
        assert_eq!(DROPS.load(Ordering::SeqCst), 2);
    }

    /// [`drop`] behavior during construction from a shared slice of an array.
    ///
    /// [`drop`]: Fillet::drop
    #[test]
    fn drop_counts_from_slice_clone() {
        DROPS.store(0, Ordering::SeqCst);
        let array = [CloneDropper(0), CloneDropper(1)];
        let f: Fillet<CloneDropper> = array.as_ref().into();
        assert_eq!(DROPS.load(Ordering::SeqCst), 0);
        drop(f);
        assert_eq!(DROPS.load(Ordering::SeqCst), 2);
        drop(array);
        assert_eq!(DROPS.load(Ordering::SeqCst), 4);
    }

    /// No double [`drop`] when constructing from an owned `Box<[T]>`.
    ///
    /// [`drop`]: Fillet::drop
    #[test]
    fn drop_counts_from_box_no_double() {
        DROPS.store(0, Ordering::SeqCst);
        let boxed: Box<[Dropper]> = vec![Dropper, Dropper].into_boxed_slice();
        let f: Fillet<Dropper> = boxed.into();
        assert_eq!(DROPS.load(Ordering::SeqCst), 0);
        drop(f);
        assert_eq!(DROPS.load(Ordering::SeqCst), 2);
    }

    /// [`drop`] behavior during construction from an owned [`Option`].
    ///
    /// [`drop`]: Fillet::drop
    #[test]
    fn drop_counts_from_option() {
        DROPS.store(0, Ordering::SeqCst);
        let o: Option<Dropper> = Some(Dropper);
        let f: Fillet<Dropper> = o.into();
        assert_eq!(DROPS.load(Ordering::SeqCst), 0);
        drop(f);
        assert_eq!(DROPS.load(Ordering::SeqCst), 1);
    }

    /// [`FilletIntoIter`] should consume and iterate over the whole [`Fillet`].
    #[test]
    fn into_iter_full_consumption() {
        let f: Fillet<i32> = [1, 2, 3].into();
        let mut iter = f.into_iter();
        assert_eq!(iter.len(), 3);
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.len(), 0);
    }

    /// [`FilletIntoIter`] should cause dropping during iteration.
    #[test]
    fn into_iter_partial_drop() {
        DROPS.store(0, Ordering::SeqCst);
        let f: Fillet<Dropper> = [Dropper, Dropper, Dropper].into();
        let mut iter = f.into_iter();
        assert_eq!(DROPS.load(Ordering::SeqCst), 0);
        assert_eq!(iter.len(), 3);
        let _ = iter.next();
        assert_eq!(DROPS.load(Ordering::SeqCst), 1);
        drop(iter);
        assert_eq!(DROPS.load(Ordering::SeqCst), 3);
    }

    /// [`FilletIntoIter`] should function correctly as a [`DoubleEndedIterator`].
    #[test]
    fn into_iter_double_ended() {
        let f: Fillet<i32> = [1, 2, 3].into();
        let mut iter = f.into_iter();
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next_back(), Some(3));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next_back(), None);
        assert_eq!(iter.len(), 0);
    }

    /// [`FilletIntoIter`] should work with ZSTs.
    #[test]
    fn into_iter_zst() {
        let f: Fillet<()> = [(); 3].into();
        let mut iter = f.into_iter();
        assert_eq!(iter.len(), 3);
        assert_eq!(iter.next(), Some(()));
        assert_eq!(iter.next_back(), Some(()));
        assert_eq!(iter.next(), Some(()));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
        assert_eq!(iter.len(), 0);
    }

    /// Construction from `Box<[String]>`.
    #[test]
    fn special_from_box_string() {
        use alloc::string::String;
        let boxed: Box<[String]> = ["a".into(), "b".into()].into();
        let f: Fillet<String> = boxed.into();
        assert_eq!(f.len(), 2);
        assert_eq!(f[0], "a");
        assert_eq!(f[1], "b");
    }

    /// Construction from `Box<[Arc<u32>]>`.
    #[test]
    fn special_from_box_arc() {
        use alloc::sync::Arc;
        let boxed: Box<[Arc<u32>]> = [Arc::new(42), Arc::new(43)].into();
        let f: Fillet<Arc<u32>> = boxed.into();
        assert_eq!(f.len(), 2);
        assert_eq!(*f[0], 42);
        assert_eq!(*f[1], 43);
    }

    /// Construction from `Box<[MutexGuard<T>]>`.
    #[test]
    fn special_from_box_mutex_guard() {
        extern crate std;
        use std::sync::{Mutex, MutexGuard};
        let mutex = Mutex::new(42u32);
        let guard = mutex.lock().unwrap();
        let boxed: Box<[MutexGuard<u32>]> = vec![guard].into_boxed_slice();
        let f: Fillet<MutexGuard<u32>> = boxed.into();
        assert_eq!(f.len(), 1);
        assert_eq!(*f[0], 42);
    }

    /// [`shrink_uninit_nonempty`] should function after grow.
    ///
    /// [`shrink_uninit_nonempty`]: Fillet::shrink_uninit_nonempty
    #[test]
    fn shrink_uninit_after_grow() {
        let mut f: Fillet<i32> = [1, 2].into();
        unsafe {
            let uninit = f.grow_nonempty(5);
            assert_eq!(uninit.len(), 3);
            // Partial init: only first of uninit.
            uninit[0].write(3);
            // Leave [3..5] uninit.
            f.shrink_uninit_nonempty(NonZeroUsize::new_unchecked(3));
        }
        assert_eq!(*f, [1, 2, 3]);
    }

    /// Basic [`truncate`] behavior.
    ///
    /// [`truncate`]: Fillet::truncate
    #[test]
    fn truncate_basic() {
        let mut f: Fillet<i32> = [1, 2, 3].into();
        f.truncate(2);
        assert_eq!(*f, [1, 2]);
        f.truncate(3); // no-op
        assert_eq!(*f, [1, 2]);
    }

    /// [`truncate`] to empty.
    ///
    /// [`truncate`]: Fillet::truncate
    #[test]
    fn truncate_to_empty() {
        let mut f: Fillet<i32> = [1, 2].into();
        f.truncate(0);
        assert_eq!(f.len(), 0);
        assert!(f.is_empty());
    }

    /// [`truncate`] should cause appropriate drops.
    ///
    /// [`truncate`]: Fillet::truncate
    #[test]
    fn truncate_drop() {
        DROPS.store(0, Ordering::SeqCst);
        let mut f: Fillet<Dropper> = [Dropper, Dropper, Dropper].into();
        assert_eq!(DROPS.load(Ordering::SeqCst), 0);
        f.truncate(1);
        assert_eq!(DROPS.load(Ordering::SeqCst), 2);
        drop(f);
        assert_eq!(DROPS.load(Ordering::SeqCst), 3);
    }

    /// Basic [`clear`] behavior.
    ///
    /// [`clear`]: Fillet::clear
    #[test]
    fn clear_basic() {
        let mut f: Fillet<i32> = [1, 2].into();
        f.clear();
        assert_eq!(f.len(), 0);
        assert!(f.is_empty());
        f.clear(); // no-op on empty
        assert_eq!(f.len(), 0);
    }

    /// [`clear`] should cause appropriate drops, and should not set up double drops.
    ///
    /// [`clear`]: Fillet::clear
    #[test]
    fn clear_drop() {
        DROPS.store(0, Ordering::SeqCst);
        let mut f: Fillet<Dropper> = [Dropper, Dropper].into();
        assert_eq!(DROPS.load(Ordering::SeqCst), 0);
        f.clear();
        assert_eq!(DROPS.load(Ordering::SeqCst), 2);
        drop(f);
        assert_eq!(DROPS.load(Ordering::SeqCst), 2);
    }

    /// Basic [`as_slice`] behavior.
    ///
    /// [`as_slice`]: Fillet::as_slice
    #[test]
    fn as_slice_basic() {
        let f: Fillet<i32> = [1, 2].into();
        assert_eq!(f.as_slice(), &[1, 2]);
        let e: Fillet<i32> = Fillet::EMPTY;
        assert_eq!(e.as_slice(), &[]);
    }

    /// Basic [`as_mut_slice`] behavior.
    ///
    /// [`as_mut_slice`]: Fillet::as_mut_slice
    #[test]
    fn as_mut_slice_basic() {
        let mut f: Fillet<i32> = [1, 2].into();
        let s = f.as_mut_slice();
        assert_eq!(s, &[1, 2]);
        s[0] = 3;
        assert_eq!(*f, [3, 2]);
        assert_eq!(f.as_mut_slice(), &[3, 2]);
        let mut e: Fillet<i32> = Fillet::EMPTY;
        assert_eq!(e.as_mut_slice(), &mut []);
    }

    /// Basic [`from_one`] behavior.
    ///
    /// [`from_one`]: Fillet::from_one
    #[test]
    fn from_one_empty_after_move() {
        let mut f: Fillet<i32> = Fillet::from_one(42);
        assert_eq!(*f, [42]);
        f.clear();
        assert_eq!(f.len(), 0);
    }

    /// [`from_one`] should work with ZSTs.
    ///
    /// [`from_one`]: Fillet::from_one
    #[test]
    fn from_one_zst() {
        let f: Fillet<()> = Fillet::from_one(());
        assert_eq!(f.len(), 1);
    }

    /// [`extend`] should work with exact size iterators.
    ///
    /// [`extend`]: Fillet::extend
    #[test]
    fn extend_exact_size() {
        let mut f: Fillet<i32> = [1, 2].into();
        f.extend(3..=5);
        assert_eq!(*f, [1, 2, 3, 4, 5]);
    }

    /// [`extend`] should work with non-exact size iterators.
    ///
    /// [`extend`]: Fillet::extend
    #[test]
    fn extend_unknown_size() {
        let mut f: Fillet<i32> = [1, 2].into();
        f.extend((3..10).filter(|&x| x % 2 == 0));
        assert_eq!(*f, [1, 2, 4, 6, 8]);
    }

    /// [`extend`] should work with empty iterators.
    ///
    /// [`extend`]: Fillet::extend
    #[test]
    fn extend_empty() {
        let mut f: Fillet<i32> = [1].into();
        f.extend(core::iter::empty::<i32>());
        assert_eq!(*f, [1]);
    }

    /// Basic [`extend_from_within`] behavior.
    ///
    /// [`extend_from_within`]: Fillet::extend_from_within
    #[test]
    fn extend_from_within() {
        let mut f: Fillet<i32> = [1, 2].into();
        f.extend_from_within(..);
        assert_eq!(*f, [1, 2, 1, 2]);
        f.extend_from_within(..2);
        assert_eq!(*f, [1, 2, 1, 2, 1, 2]);
        f.extend_from_within(..0);
        assert_eq!(*f, [1, 2, 1, 2, 1, 2]);
    }

    /// [`extend_from_within`] should work with ZSTs.
    ///
    /// [`extend_from_within`]: Fillet::extend_from_within
    #[test]
    fn extend_from_within_zst() {
        DROPS.store(0, Ordering::SeqCst);
        let mut f: Fillet<Dropper> = repeat_n(Dropper, 2).collect();
        f.extend_from_within(0..1);
        assert_eq!(f.len(), 3);
        f.extend_from_within(..);
        assert_eq!(f.len(), 6);
        assert_eq!(DROPS.load(Ordering::SeqCst), 0);
        drop(f);
        assert_eq!(DROPS.load(Ordering::SeqCst), 6);
    }

    /// [`extend`] should not drop items.
    ///
    /// [`extend`]: Fillet::extend
    #[test]
    fn extend_drop() {
        DROPS.store(0, Ordering::SeqCst);
        let mut f: Fillet<Dropper> = [Dropper].into();
        assert_eq!(DROPS.load(Ordering::SeqCst), 0);
        f.extend((0..2).map(|_| Dropper));
        assert_eq!(DROPS.load(Ordering::SeqCst), 0);
        assert_eq!(f.len(), 3);
        drop(f);
        assert_eq!(DROPS.load(Ordering::SeqCst), 3);
    }

    /// [`extend`] should work with ZSTs.
    ///
    /// [`extend`]: Fillet::extend
    #[test]
    fn extend_zst() {
        let mut f: Fillet<()> = [(); 2].into();
        f.extend(repeat_n((), 3));
        assert_eq!(f.len(), 5);
    }

    /// [`from_iter`] should work with exact-size iterators.
    ///
    /// [`from_iter`]: Fillet::from_iter
    #[test]
    fn from_iter_exact_size() {
        let f: Fillet<i32> = (0..3).collect();
        assert_eq!(*f, [0, 1, 2]);
    }

    /// [`from_iter`] should work with non exact-size iterators.
    ///
    /// [`from_iter`]: Fillet::from_iter
    #[test]
    fn from_iter_unknown_size() {
        let f: Fillet<i32> = (0..10).filter(|&x| x % 2 == 0).collect();
        assert_eq!(*f, [0, 2, 4, 6, 8]);
    }

    /// [`from_iter`] should work with `IntoIter = &[T]`.
    ///
    /// [`from_iter`]: Fillet::from_iter
    #[test]
    fn from_slice_via_iter() {
        let slice: &[i32] = &[1, 2, 3];
        let f: Fillet<i32> = Fillet::from(slice);
        assert_eq!(*f, [1, 2, 3]);
    }

    /// [`drop`] behavior for `Fillet<T: Clone>` from shared slice.
    ///
    /// [`drop`]: Fillet::drop
    #[test]
    fn from_slice_clone_drop() {
        DROPS.store(0, Ordering::SeqCst);
        let array = [CloneDropper(0), CloneDropper(1)];
        let f: Fillet<CloneDropper> = Fillet::from(array.as_ref());
        assert_eq!(DROPS.load(Ordering::SeqCst), 0);
        drop(f);
        assert_eq!(DROPS.load(Ordering::SeqCst), 2);
        drop(array);
        assert_eq!(DROPS.load(Ordering::SeqCst), 4);
    }

    /// Should not [`drop`] during `from_iter`, even for ZSTs.
    ///
    /// [`drop`]: Fillet::drop
    #[test]
    fn from_iter_drop() {
        DROPS.store(0, Ordering::SeqCst);
        let f: Fillet<Dropper> = (0..3).map(|_| Dropper).collect();
        assert_eq!(DROPS.load(Ordering::SeqCst), 0);
        drop(f);
        assert_eq!(DROPS.load(Ordering::SeqCst), 3);
    }

    /// Basic [`basic`] behavior.
    ///
    /// [`basic`]: Fillet::truncate
    #[test]
    fn pop_correct() {
        let mut f: Fillet<i32> = [1, 2, 3].into();
        assert_eq!(f.pop(), Some(3));
        assert_eq!(*f, [1, 2]);
        assert_eq!(f.pop(), Some(2));
        assert_eq!(*f, [1]);
        assert_eq!(f.pop(), Some(1));
        assert_eq!(*f, []);
        assert_eq!(f.pop(), None);

        DROPS.store(0, Ordering::SeqCst);
        let mut f: Fillet<Dropper> = [Dropper, Dropper].into();
        assert_eq!(DROPS.load(Ordering::SeqCst), 0);
        let item = f.pop();
        assert_eq!(DROPS.load(Ordering::SeqCst), 0); // Popped item not dropped yet
        assert_eq!(f.len(), 1);
        drop(item);
        assert_eq!(DROPS.load(Ordering::SeqCst), 1); // Popped item dropped
        drop(f);
        assert_eq!(DROPS.load(Ordering::SeqCst), 2); // Remaining item dropped
    }

    /// Basic [`push`] behavior.
    ///
    /// [`push`]: Fillet::push
    #[test]
    fn push_basic() {
        let mut f: Fillet<i32> = Fillet::EMPTY;
        f.push(1);
        assert_eq!(*f, [1]);
        f.push(2);
        assert_eq!(*f, [1, 2]);
    }

    /// [`push`] should work with non-empty `Fillet`.
    ///
    /// [`push`]: Fillet::push
    #[test]
    fn push_to_non_empty() {
        let mut f: Fillet<i32> = [1].into();
        f.push(2);
        assert_eq!(*f, [1, 2]);
        f.push(3);
        assert_eq!(*f, [1, 2, 3]);
    }

    /// [`push`] should work with ZSTs.
    ///
    /// [`push`]: Fillet::push
    #[test]
    fn push_zst() {
        let mut f: Fillet<()> = Fillet::EMPTY;
        f.push(());
        assert_eq!(f.len(), 1);
        f.push(());
        assert_eq!(f.len(), 2);
    }

    /// [`push`] should not drop the pushed value.
    ///
    /// [`push`]: Fillet::push
    #[test]
    fn push_drop() {
        DROPS.store(0, Ordering::SeqCst);
        let mut f: Fillet<Dropper> = Fillet::EMPTY;
        f.push(Dropper);
        assert_eq!(DROPS.load(Ordering::SeqCst), 0);
        f.push(Dropper);
        assert_eq!(DROPS.load(Ordering::SeqCst), 0);
        drop(f);
        assert_eq!(DROPS.load(Ordering::SeqCst), 2);
    }

    /// Basic [`retain`] behavior.
    ///
    /// [`retain`]: Fillet::retain
    #[test]
    fn retain_basic() {
        let mut f: Fillet<i32> = [1, 2, 3, 4, 5].into();
        f.retain(|&x| x % 2 == 0);
        assert_eq!(*f, [2, 4]);
    }

    /// [`retain`] should trigger the correct number of drops at the correct time.
    ///
    /// [`retain`]: Fillet::retain
    #[test]
    fn retain_drop_counts() {
        DROPS.store(0, Ordering::SeqCst);
        let mut f: Fillet<Dropper> = repeat_n(Dropper, 5).collect();
        assert_eq!(DROPS.load(Ordering::SeqCst), 0);
        f.retain(|_| false); // Drop all
        assert_eq!(f.len(), 0);
        assert_eq!(DROPS.load(Ordering::SeqCst), 5);
    }

    /// [`retain`] should work with ZSTs.
    ///
    /// [`retain`]: Fillet::retain
    #[test]
    fn retain_zst() {
        let mut f: Fillet<()> = [(); 5].into();
        f.retain(|_| false);
        assert_eq!(f.len(), 0);
    }

    /// [`retain`] should not trigger UB when giving the predicate a reference to a ZST.
    ///
    /// [`retain`]: Fillet::retain
    #[test]
    fn retain_zst_predicate_ub() {
        let mut f: Fillet<()> = [(); 3].into();
        f.retain(|_| true);
        assert_eq!(f.len(), 3);
    }

    /// [`retain`] should drop non-retained elements.
    ///
    /// [`retain`]: Fillet::retain
    #[test]
    fn retain_dropper_side_effect() {
        DROPS.store(0, Ordering::SeqCst);
        let mut f: Fillet<Dropper> = repeat_n(Dropper, 4).collect();
        f.retain(|_| false);
        assert_eq!(f.len(), 0);
        assert_eq!(DROPS.load(Ordering::SeqCst), 4);
    }

    /// [`retain`] should be able to panic in the predicate without triggering UB.
    ///
    /// [`retain`]: Fillet::retain
    #[test]
    #[cfg(miri)]
    #[should_panic]
    fn retain_panic_mid_loop() {
        // Tests panic safety: predicate panics after processing some elements.
        // If code is incorrect, during unwind, Fillet::drop will attempt drop_in_place on the full old_len,
        // but [dst..old_len] contains uninitialized slots (after read/drop or read/write), leading to UB
        // (Miri detects: drop of uninitialized memory).
        // Use Dropper to force Drop glue on uninit slots.
        let mut f: Fillet<Dropper> = repeat_n(Dropper, 5).collect();
        let mut count = 0;
        f.retain(|_| {
            count += 1;
            if count == 3 {
                panic!("Simulated panic in predicate");
            }
            count % 2 == 1 // Keep first, drop second, panic on third.
        });
        // If no panic, test fails; on panic, Miri should error if UB in drop.
    }

    /// [`as_ref`] behavior for non-empty [`Fillet`].
    ///
    /// [`as_ref`]: Fillet::as_ref
    #[test]
    fn as_ref_non_empty() {
        let f: Fillet<i32> = [1, 2, 3].into();
        let slice: &[i32] = f.as_ref();
        assert_eq!(slice, &[1, 2, 3]);

        // Generic usage
        fn check_slice<R: AsRef<[i32]>>(r: R) {
            assert_eq!(r.as_ref(), &[1, 2, 3]);
        }
        check_slice(f);
    }

    /// [`as_ref`] behavior for empty [`Fillet`].
    ///
    /// [`as_ref`]: Fillet::as_ref
    #[test]
    fn as_ref_empty() {
        let f: Fillet<i32> = Fillet::EMPTY;
        let slice: &[i32] = f.as_ref();
        assert_eq!(slice, &[]);

        // Generic usage
        fn check_slice<R: AsRef<[i32]>>(r: R) {
            assert!(r.as_ref().is_empty());
        }
        check_slice(f);
    }

    /// [`as_mut`] behavior for non-empty [`Fillet`].
    ///
    /// [`as_mut`]: Fillet::as_mut
    #[test]
    fn as_mut_non_empty() {
        let mut f: Fillet<i32> = [1, 2, 3].into();
        let slice: &[i32] = f.as_mut();
        assert_eq!(slice, &mut [1, 2, 3]);

        // Generic usage
        fn mut_slice<R: AsMut<[i32]>>(mut r: R) {
            let s = r.as_mut();
            for item in s {
                *item += 3;
            }
            assert_eq!(r.as_mut().len(), 3);
            assert_eq!(r.as_mut().first(), Some(&4i32));
        }
        mut_slice(f);
    }

    /// [`as_mut`] behavior for empty [`Fillet`].
    ///
    /// [`as_mut`]: Fillet::as_mut
    #[test]
    fn as_mut_empty() {
        let mut f: Fillet<i32> = Fillet::EMPTY;
        let slice: &[i32] = f.as_mut();
        assert_eq!(slice, &mut []);

        // Generic usage
        fn mut_slice<R: AsMut<[i32]>>(mut r: R) {
            assert!(r.as_mut().is_empty());
        }
        mut_slice(f);
    }

    /// [`borrow`] behavior for non-empty [`Fillet`].
    ///
    /// [`borrow`]: Fillet::borrow
    #[test]
    fn borrow_non_empty() {
        let f: Fillet<i32> = [1, 2, 3].into();
        let slice: &[i32] = f.borrow();
        assert_eq!(slice, &[1, 2, 3]);

        // Generic usage (e.g., for HashMap keys)
        fn check_borrow<B: Borrow<[i32]>>(b: B) {
            assert_eq!(b.borrow(), &[1, 2, 3]);
        }
        check_borrow(f);
    }

    /// [`borrow`] behavior for empty [`Fillet`].
    ///
    /// [`borrow`]: Fillet::borrow
    #[test]
    fn borrow_empty() {
        let f: Fillet<i32> = Fillet::EMPTY;
        let slice: &[i32] = f.borrow();
        assert_eq!(slice, &[]);

        // Generic usage
        fn check_borrow<B: Borrow<[i32]>>(b: B) {
            assert!(b.borrow().is_empty());
        }
        check_borrow(f);
    }

    /// [`borrow_mut`] behavior for non-empty [`Fillet`].
    ///
    /// [`borrow_mut`]: Fillet::borrow_mut
    #[test]
    fn borrow_mut_non_empty() {
        let mut f: Fillet<i32> = [1, 2, 3].into();
        let slice: &mut [i32] = f.borrow_mut();
        assert_eq!(slice, &mut [1, 2, 3]);

        // Generic usage with mutation
        fn mut_borrow<BM: BorrowMut<[i32]>>(mut bm: BM) {
            let s = bm.borrow_mut();
            for item in s {
                *item += 3;
            }
            assert_eq!(bm.borrow_mut().len(), 3);
            assert_eq!(bm.borrow_mut().first(), Some(&4i32));
        }
        mut_borrow(f);
    }

    /// [`borrow_mut`] behavior for empty [`Fillet`].
    ///
    /// [`borrow_mut`]: Fillet::borrow_mut
    #[test]
    fn borrow_mut_empty() {
        let mut f: Fillet<i32> = Fillet::EMPTY;
        let slice: &mut [i32] = f.borrow_mut();
        assert_eq!(slice, &mut []);

        // Generic usage
        fn mut_borrow<BM: BorrowMut<[i32]>>(mut bm: BM) {
            assert!(bm.borrow_mut().is_empty());
        }
        mut_borrow(f);
    }

    /// Layout for `T` having large alignment.
    #[test]
    fn large_alignment_layout() {
        #[repr(align(512))]
        #[derive(Clone, Copy, Debug, PartialEq)]
        struct BigAlign(u64);

        let mut f: Fillet<BigAlign> = Fillet::EMPTY;
        f.push(BigAlign(1));
        f.push(BigAlign(2));
        assert_eq!(f.len(), 2);
        assert_eq!(f[0].0, 1);
        assert_eq!(f[1].0, 2);

        // Trigger realloc and access post-grow.
        f.extend([BigAlign(3), BigAlign(4)]);
        assert_eq!(f[3].0, 4);

        // Pop and check alignment after shrink.
        assert_eq!(f.pop().unwrap().0, 4);
        assert_eq!(f[2].0, 3);
    }

    /// ZST with drop during retain.
    #[test]
    fn zst_drop_retain() {
        DROPS.store(0, Ordering::SeqCst);
        let mut f: Fillet<Dropper> = repeat_n(Dropper, 10).collect();
        f.retain(|_| false); // Drop all via read/drop.
        assert_eq!(f.len(), 0);
        assert_eq!(DROPS.load(Ordering::SeqCst), 10);

        let mut f: Fillet<Dropper> = repeat_n(Dropper, 10).collect();
        f.retain(|_| true); // Keep all via read/write if moved.
        assert_eq!(f.len(), 10);
        assert_eq!(DROPS.load(Ordering::SeqCst), 10);
        drop(f);
        assert_eq!(DROPS.load(Ordering::SeqCst), 20);
    }

    /// MAX_LEN overflow panic.
    #[test]
    #[should_panic(expected = "9007199254740992 elements > MAX_LEN = 9007199254740991.
Requested Fillet larger than isize::MAX bytes.")]
    fn max_len_overflow_panic() {
        type Large = [u8; 1024];
        let max = Fillet::<Large>::MAX_LEN;
        let _f: Fillet<Large> = repeat_n([0u8; 1024], max + 1).collect();
    }

    /// Layout for `T` having small alignment.
    #[test]
    fn small_align_layout() {
        #[repr(align(1))]
        #[derive(Clone, Copy, Debug, PartialEq)]
        struct SmallAlign(u8);

        let mut f: Fillet<SmallAlign> = Fillet::EMPTY;
        for i in 0..10 {
            f.push(SmallAlign(i as u8));
        }
        assert_eq!(f.len(), 10);
        for i in 0..10 {
            assert_eq!(f[i].0, i as u8);
        }
        // Grow and shrink to trigger realloc.
        f.truncate(5);
        assert_eq!(f[4].0, 4);
    }

    /// ZST with drop during extend_from_within.
    #[test]
    fn zst_drop_extend_from_within() {
        DROPS.store(0, Ordering::SeqCst);
        let mut f: Fillet<Dropper> = repeat_n(Dropper, 5).collect();
        f.extend_from_within(1..4);
        assert_eq!(f.len(), 8);
        assert_eq!(DROPS.load(Ordering::SeqCst), 0);
        drop(f);
        assert_eq!(DROPS.load(Ordering::SeqCst), 8);
    }

    /// Verify repr(packed) alignment not bungled.
    #[test]
    fn repr_packed_misalign() {
        #[repr(Rust, packed)]
        #[derive(Clone, Copy, Debug, PartialEq)]
        struct Packed {
            a: u8,
            b: u64,
        }

        impl Packed {
            fn read_b_safe(&self) -> u64 {
                unsafe { ptr::read_unaligned(&raw const self.b) }
            }
        }

        let mut f: Fillet<Packed> = Fillet::EMPTY;
        f.push(Packed { a: 1, b: 2 });
        f.push(Packed { a: 3, b: 4 });
        assert_eq!(f.len(), 2);
        assert_eq!(f[1].read_b_safe(), 4);
        f.truncate(1);
        assert_eq!(f[0].read_b_safe(), 2);
    }

    /// repr(transparent) with inner field + ZST.
    #[test]
    fn repr_transparent() {
        use core::marker::PhantomData;
        #[repr(transparent)]
        #[derive(Clone, Copy, Debug, PartialEq)]
        struct Trans(u32, PhantomData<i64>); // Transparent over u32.

        assert_eq!(size_of::<Trans>(), size_of::<u32>());
        assert_eq!(align_of::<Trans>(), align_of::<u32>());

        let mut f: Fillet<Trans> = [Trans(1, PhantomData), Trans(2, PhantomData)].into();
        assert_eq!(f.len(), 2);
        assert_eq!(f[0].0, 1);
        f.pop();
        assert_eq!(f[0].0, 1);
    }

    /// repr(C) enum/tagged union.
    #[test]
    fn repr_c_enum() {
        #[repr(C)]
        #[derive(Clone, Copy, Debug, PartialEq)]
        enum Tagged {
            A { x: u32 },
            B { y: u64 },
        }

        let mut f: Fillet<Tagged> = Fillet::EMPTY;
        f.push(Tagged::A { x: 42 });
        f.push(Tagged::B { y: 99 });
        assert_eq!(f.len(), 2);
        if let Tagged::A { x } = f[0] {
            assert_eq!(x, 42);
        }
    }
}
