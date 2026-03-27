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
use core::fmt::{self, Debug, Formatter};
use core::hash::{Hash, Hasher};
use core::hint::unreachable_unchecked;
use core::iter::{
    self, DoubleEndedIterator, ExactSizeIterator, FromIterator, FusedIterator, IntoIterator,
    Iterator,
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
                ptr::drop_in_place(ptr::slice_from_raw_parts_mut(
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

            // Write new_len BEFORE drop_in_place so that if a destructor panics,
            // Fillet::drop during unwind sees only the surviving elements.
            (old_ptr as *mut usize).write(new_len);

            // Realloc (or dealloc) the allocation to match new_len.
            // Runs after drop_in_place on both the normal and panic paths,
            // so the allocation size matches the header when Fillet::drop runs.
            struct ReallocGuard<T> {
                fillet: *mut Fillet<T>,
                old_layout: Layout,
                new_len: usize,
            }
            impl<T> Drop for ReallocGuard<T> {
                fn drop(&mut self) {
                    unsafe {
                        let fillet = &mut *self.fillet;
                        if self.new_len != 0 {
                            let new_layout = Fillet::<T>::compute_layout_unchecked(self.new_len);
                            let old_ptr = fillet.ptr.unwrap_unchecked().as_ptr();
                            fillet.ptr =
                                NonNull::new(realloc(old_ptr, self.old_layout, new_layout.size()));
                            if let Some(nn) = fillet.ptr {
                                nn.cast::<usize>().write(self.new_len);
                            } else {
                                handle_alloc_error(new_layout);
                            }
                        } else {
                            dealloc(fillet.ptr.unwrap_unchecked().as_ptr(), self.old_layout);
                            fillet.ptr = None;
                        }
                    }
                }
            }

            let _guard = ReallocGuard {
                fillet: self as *mut _,
                old_layout,
                new_len,
            };

            // SAFETY: Caller responsible for ensuring `new_len < self.len()`.
            ptr::drop_in_place(ptr::slice_from_raw_parts_mut(
                old_ptr.byte_add(Self::DATA_OFFSET).cast::<T>().add(new_len),
                old_len - new_len,
            ));
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

/// Shrinks a `Fillet`'s allocation to match `actual_len` on drop.
///
/// After `grow`/`grow_nonempty`, the header reflects the allocated capacity.
/// `actual_len` tracks how many elements are initialized. On drop, the allocation
/// is shrunk to `actual_len` if it differs from the header.
struct ExtendGuard<T> {
    fillet: *mut Fillet<T>,
    actual_len: usize,
}

impl<T> Drop for ExtendGuard<T> {
    fn drop(&mut self) {
        unsafe {
            let fillet = &mut *self.fillet;
            if size_of::<T>() == 0 {
                fillet.len = self.actual_len;
                return;
            }
            // If actual_len already matches the header, shrink_uninit_nonempty is a
            // same-size realloc (no-op in practice), so skip it.
            if fillet.len() != self.actual_len {
                if let Some(nz) = NonZeroUsize::new(self.actual_len) {
                    fillet.shrink_uninit_nonempty(nz);
                } else {
                    fillet.dealloc_nonempty();
                }
            }
        }
    }
}

impl<T> Drop for Fillet<T> {
    fn drop(&mut self) {
        self.clear();
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

impl<T: PartialOrd> PartialOrd for Fillet<T> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}

impl<T: Ord> Ord for Fillet<T> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl<T: PartialEq<U>, U> PartialEq<[U]> for Fillet<T> {
    fn eq(&self, other: &[U]) -> bool {
        self.as_slice() == other
    }
}

impl<T: PartialEq<U>, U> PartialEq<Fillet<U>> for [T] {
    fn eq(&self, other: &Fillet<U>) -> bool {
        self == other.as_slice()
    }
}

impl<T: PartialEq<U>, U> PartialEq<&[U]> for Fillet<T> {
    fn eq(&self, other: &&[U]) -> bool {
        self.as_slice() == *other
    }
}

impl<T: PartialEq<U>, U> PartialEq<Fillet<U>> for &[T] {
    fn eq(&self, other: &Fillet<U>) -> bool {
        *self == other.as_slice()
    }
}

impl<T: PartialEq<U>, U, const N: usize> PartialEq<[U; N]> for Fillet<T> {
    fn eq(&self, other: &[U; N]) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: PartialEq<U>, U, const N: usize> PartialEq<Fillet<U>> for [T; N] {
    fn eq(&self, other: &Fillet<U>) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: PartialEq<U>, U, const N: usize> PartialEq<&[U; N]> for Fillet<T> {
    fn eq(&self, other: &&[U; N]) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: PartialEq<U>, U, const N: usize> PartialEq<Fillet<U>> for &[T; N] {
    fn eq(&self, other: &Fillet<U>) -> bool {
        self.as_slice() == other.as_slice()
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

        // [Kept, Kept, Hole, Hole, Unchecked, Unchecked]
        // |            ^write      ^read               |
        // |<-               old_len                  ->|
        //
        // On panic, the guard shifts Unchecked down to cover the Hole region
        // and adjusts the allocation.
        struct RetainGuard<T> {
            fillet: *mut Fillet<T>,
            data: *mut T,
            read: usize,
            write: usize,
            old_len: usize,
        }
        impl<T> Drop for RetainGuard<T> {
            #[cold]
            fn drop(&mut self) {
                let remaining = self.old_len - self.read;
                unsafe {
                    if remaining > 0 {
                        ptr::copy(
                            self.data.add(self.read),
                            self.data.add(self.write),
                            remaining,
                        );
                    }
                    let new_len = self.write + remaining;
                    let fillet = &mut *self.fillet;
                    if let Some(nz) = NonZeroUsize::new(new_len) {
                        fillet.shrink_uninit_nonempty(nz);
                    } else {
                        fillet.dealloc_nonempty();
                    }
                }
            }
        }

        // Fast path: scan the prefix where every element is kept.
        // No guard is needed until the first rejection.
        let s = self.as_mut_ptr();
        let mut read = 0;
        loop {
            if unsafe { !f(&*s.add(read)) } {
                break;
            }
            read += 1;
            if read == old_len {
                return;
            }
        }

        // Advance read past the first rejected element before dropping it,
        // so the guard never includes it in the Unchecked region.
        let mut g = RetainGuard {
            fillet: self as *mut _,
            data: s,
            read: read + 1,
            write: read,
            old_len,
        };
        unsafe { ptr::drop_in_place(s.add(read)) };

        while g.read < old_len {
            unsafe {
                let cur = s.add(g.read);
                if !f(&*cur) {
                    g.read += 1;
                    ptr::drop_in_place(cur);
                } else {
                    ptr::copy_nonoverlapping(cur, s.add(g.write), 1);
                    g.write += 1;
                    g.read += 1;
                }
            }
        }

        let write = g.write;
        core::mem::forget(g);

        unsafe {
            if let Some(nz) = NonZeroUsize::new(write) {
                self.shrink_uninit_nonempty(nz);
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
                let dst = self.grow_nonempty(new_len).as_mut_ptr();
                let mut guard = ExtendGuard {
                    fillet: self as *mut _,
                    actual_len: old_len,
                };
                // SAFETY: The range is already bounds checked in the heap array.
                let src = (*guard.fillet)
                    .ptr
                    .unwrap_unchecked()
                    .byte_add(Self::DATA_OFFSET)
                    .cast::<T>();
                for (j, i) in range.enumerate() {
                    dst.add(j)
                        .write(MaybeUninit::new(src.add(i).as_ref().clone()));
                    guard.actual_len += 1;
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
            unsafe { self.len += added };
            return;
        }

        let initial_len = self.len();

        let (lower, upper_opt) = iter.size_hint();

        // Exact size iterator.
        if upper_opt.is_some_and(|u| u == lower) {
            if lower != 0 {
                let uninit = unsafe { self.grow(NonZeroUsize::new_unchecked(initial_len + lower)) }
                    .as_mut_ptr();
                let mut guard = ExtendGuard {
                    fillet: self as *mut _,
                    actual_len: initial_len,
                };
                for item in iter {
                    if guard.actual_len - initial_len >= lower {
                        break;
                    }
                    unsafe {
                        uninit
                            .add(guard.actual_len - initial_len)
                            .write(MaybeUninit::new(item))
                    };
                    guard.actual_len += 1;
                }
            }
            return;
        }

        let mut capacity = initial_len;
        let mut growth = upper_opt.unwrap_or(lower.max(4));
        capacity += growth;
        let mut uninit = unsafe { self.grow(NonZeroUsize::new_unchecked(capacity)) }.as_mut_ptr();
        let mut remaining_uninit = capacity - initial_len;
        let mut guard = ExtendGuard {
            fillet: self as *mut _,
            actual_len: initial_len,
        };
        for item in iter {
            if remaining_uninit == 0 {
                growth *= 2;
                capacity += growth;
                let grown = unsafe { (*guard.fillet).grow_nonempty(capacity) };
                remaining_uninit = grown.len();
                uninit = grown.as_mut_ptr();
            }
            unsafe { uninit.write(MaybeUninit::new(item)) };
            unsafe { uninit = uninit.add(1) };
            remaining_uninit -= 1;
            guard.actual_len += 1;
        }
    }
}

impl<'a, T: Copy + 'a> Extend<&'a T> for Fillet<T> {
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        self.extend(iter.into_iter().copied());
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

impl<T> FilletIntoIter<T> {
    /// View the unconsumed elements as a slice.
    pub fn as_slice(&self) -> &[T] {
        if size_of::<T>() == 0 {
            return unsafe { slice::from_raw_parts(ptr::dangling::<T>(), self.end - self.start) };
        }

        unsafe {
            match self.inner.ptr {
                None => &[],
                Some(ptr) => {
                    slice::from_raw_parts(ptr.as_ptr().add(self.start), self.end - self.start)
                }
            }
        }
    }

    /// View the unconsumed elements as a mutable slice.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        if size_of::<T>() == 0 {
            return unsafe {
                slice::from_raw_parts_mut(ptr::dangling_mut::<T>(), self.end - self.start)
            };
        }

        unsafe {
            match self.inner.ptr {
                None => &mut [],
                Some(ptr) => {
                    slice::from_raw_parts_mut(ptr.as_ptr().add(self.start), self.end - self.start)
                }
            }
        }
    }
}

// SAFETY: FilletIntoIter owns [T], so is Send/Sync as long as T is.
unsafe impl<T: Send> Send for FilletIntoIter<T> {}
unsafe impl<T: Sync> Sync for FilletIntoIter<T> {}

impl<T: Debug> Debug for FilletIntoIter<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(self.as_slice(), f)
    }
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

impl<T> FusedIterator for FilletIntoIter<T> {}

impl<T> Drop for FilletIntoIter<T> {
    fn drop(&mut self) {
        // ZSTs do not have an allocation.
        if size_of::<T>() == 0 {
            let len = self.len();
            if len != 0 {
                // SAFETY: drop_in_place is safe on a dangling pointer for ZSTs
                unsafe {
                    ptr::drop_in_place(ptr::slice_from_raw_parts_mut(
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

                // Ensure dealloc runs even if drop_in_place panics.
                struct DeallocGuard {
                    ptr: *mut u8,
                    layout: Layout,
                }
                impl Drop for DeallocGuard {
                    fn drop(&mut self) {
                        unsafe { dealloc(self.ptr, self.layout) };
                    }
                }

                let _guard = DeallocGuard { ptr, layout };
                ptr::drop_in_place(ptr::slice_from_raw_parts_mut(
                    data_ptr.add(self.start),
                    self.end - self.start,
                ));
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

impl<'a, T> IntoIterator for &'a Fillet<T> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

impl<'a, T> IntoIterator for &'a mut Fillet<T> {
    type Item = &'a mut T;
    type IntoIter = slice::IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.as_mut_slice().iter_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use core::iter::repeat_n;
    use core::sync::atomic::{AtomicUsize, Ordering};

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

    /// `Eq` and `Hash` behavior for ZSTs.
    #[test]
    fn eq_and_hash_zst() {
        let f1: Fillet<()> = [(); 3].into();
        let f2: Fillet<()> = [(); 3].into();
        let f3: Fillet<()> = [(); 1].into();
        assert_eq!(f1, f2);
        assert_ne!(f1, f3);
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
        struct Dropper;
        static DROPS: AtomicUsize = AtomicUsize::new(0);
        impl Drop for Dropper {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::SeqCst);
            }
        }

        let f: Fillet<Dropper> = (0..3).map(|_| Dropper).collect();
        assert_eq!(DROPS.load(Ordering::SeqCst), 0);
        assert_eq!(f.len(), 3);
        drop(f);
        assert_eq!(DROPS.load(Ordering::SeqCst), 3);
    }

    /// [`drop`] behavior during construction from an array.
    ///
    /// [`drop`]: Fillet::drop
    #[test]
    fn drop_counts_from_array() {
        struct Dropper;
        static DROPS: AtomicUsize = AtomicUsize::new(0);
        impl Drop for Dropper {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::SeqCst);
            }
        }
        let arr = [Dropper, Dropper];
        let f: Fillet<Dropper> = arr.into();
        assert_eq!(DROPS.load(Ordering::SeqCst), 0);
        assert_eq!(f.len(), 2);
        drop(f);
        assert_eq!(DROPS.load(Ordering::SeqCst), 2);
    }

    /// [`drop`] behavior during construction from a shared slice of an array.
    ///
    /// [`drop`]: Fillet::drop
    #[test]
    fn drop_counts_from_slice_clone() {
        #[derive(Clone)]
        struct Dropper;
        static DROPS: AtomicUsize = AtomicUsize::new(0);
        impl Drop for Dropper {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::SeqCst);
            }
        }

        let array = [Dropper, Dropper];
        let f: Fillet<Dropper> = array.as_ref().into();
        assert_eq!(DROPS.load(Ordering::SeqCst), 0);
        assert_eq!(f.len(), 2);
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
        struct Dropper;
        static DROPS: AtomicUsize = AtomicUsize::new(0);
        impl Drop for Dropper {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::SeqCst);
            }
        }
        let boxed: Box<[Dropper]> = Box::from([Dropper, Dropper]);
        let f: Fillet<Dropper> = boxed.into();
        assert_eq!(DROPS.load(Ordering::SeqCst), 0);
        assert_eq!(f.len(), 2);
        drop(f);
        assert_eq!(DROPS.load(Ordering::SeqCst), 2);
    }

    /// [`drop`] behavior during construction from an owned [`Option`].
    ///
    /// [`drop`]: Fillet::drop
    #[test]
    fn drop_counts_from_option() {
        struct Dropper;
        static DROPS: AtomicUsize = AtomicUsize::new(0);
        impl Drop for Dropper {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::SeqCst);
            }
        }
        let o: Option<Dropper> = Some(Dropper);
        let f: Fillet<Dropper> = o.into();
        assert_eq!(DROPS.load(Ordering::SeqCst), 0);
        assert_eq!(f.len(), 1);
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
        struct Dropper;
        static DROPS: AtomicUsize = AtomicUsize::new(0);
        impl Drop for Dropper {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::SeqCst);
            }
        }
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
        struct Dropper;
        static DROPS: AtomicUsize = AtomicUsize::new(0);
        impl Drop for Dropper {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::SeqCst);
            }
        }
        let mut f: Fillet<Dropper> = [Dropper, Dropper, Dropper].into();
        assert_eq!(DROPS.load(Ordering::SeqCst), 0);
        assert_eq!(f.len(), 3);
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
        struct Dropper;
        static DROPS: AtomicUsize = AtomicUsize::new(0);
        impl Drop for Dropper {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::SeqCst);
            }
        }
        let mut f: Fillet<Dropper> = [Dropper, Dropper].into();
        assert_eq!(DROPS.load(Ordering::SeqCst), 0);
        assert_eq!(f.len(), 2);
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
        #[derive(Clone)]
        struct Dropper;
        static DROPS: AtomicUsize = AtomicUsize::new(0);
        impl Drop for Dropper {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::SeqCst);
            }
        }
        let mut f: Fillet<Dropper> = [Dropper, Dropper].into();
        f.extend_from_within(0..1);
        assert_eq!(f.len(), 3);
        f.extend_from_within(..);
        assert_eq!(f.len(), 6);
        assert_eq!(DROPS.load(Ordering::SeqCst), 0);
        assert_eq!(f.len(), 6);
        drop(f);
        assert_eq!(DROPS.load(Ordering::SeqCst), 6);
    }

    /// [`extend`] should not drop items.
    ///
    /// [`extend`]: Fillet::extend
    #[test]
    fn extend_drop() {
        struct Dropper;
        static DROPS: AtomicUsize = AtomicUsize::new(0);
        impl Drop for Dropper {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::SeqCst);
            }
        }
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
        #[derive(Clone)]
        struct Dropper;
        static DROPS: AtomicUsize = AtomicUsize::new(0);
        impl Drop for Dropper {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::SeqCst);
            }
        }

        let array = [Dropper, Dropper];
        let f: Fillet<Dropper> = Fillet::from(array.as_ref());
        assert_eq!(DROPS.load(Ordering::SeqCst), 0);
        assert_eq!(f.len(), 2);
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
        struct Dropper;
        static DROPS: AtomicUsize = AtomicUsize::new(0);
        impl Drop for Dropper {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::SeqCst);
            }
        }
        let f: Fillet<Dropper> = repeat_n((), 3).map(|_| Dropper).collect();
        assert_eq!(DROPS.load(Ordering::SeqCst), 0);
        assert_eq!(f.len(), 3);
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

        struct Dropper;
        static DROPS: AtomicUsize = AtomicUsize::new(0);
        impl Drop for Dropper {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::SeqCst);
            }
        }
        let mut f: Fillet<Dropper> = [Dropper, Dropper].into();
        assert_eq!(DROPS.load(Ordering::SeqCst), 0);
        assert_eq!(f.len(), 2);
        let item = f.pop();
        assert_eq!(DROPS.load(Ordering::SeqCst), 0); // Popped item not dropped yet
        assert_eq!(f.len(), 1);
        drop(item);
        assert_eq!(DROPS.load(Ordering::SeqCst), 1); // Popped item dropped
        assert_eq!(f.len(), 1);
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
        struct Dropper;
        static DROPS: AtomicUsize = AtomicUsize::new(0);
        impl Drop for Dropper {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::SeqCst);
            }
        }
        let mut f: Fillet<Dropper> = Fillet::EMPTY;
        f.push(Dropper);
        assert_eq!(DROPS.load(Ordering::SeqCst), 0);
        f.push(Dropper);
        assert_eq!(DROPS.load(Ordering::SeqCst), 0);
        assert_eq!(f.len(), 2);
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
        struct Dropper;
        static DROPS: AtomicUsize = AtomicUsize::new(0);
        impl Drop for Dropper {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::SeqCst);
            }
        }
        let mut f: Fillet<Dropper> = repeat_n((), 5).map(|_| Dropper).collect();
        assert_eq!(DROPS.load(Ordering::SeqCst), 0);
        assert_eq!(f.len(), 5);
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
        struct Dropper;
        static DROPS: AtomicUsize = AtomicUsize::new(0);
        impl Drop for Dropper {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::SeqCst);
            }
        }
        let mut f: Fillet<Dropper> = [Dropper, Dropper, Dropper, Dropper].into();
        f.retain(|_| false);
        assert_eq!(f.len(), 0);
        assert_eq!(DROPS.load(Ordering::SeqCst), 4);
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
        #[derive(Clone)]
        struct Dropper;
        static DROPS: AtomicUsize = AtomicUsize::new(0);
        impl Drop for Dropper {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::SeqCst);
            }
        }
        let mut f: Fillet<Dropper> = repeat_n(Dropper, 10).collect();
        f.retain(|_| false); // Drop all via read/drop.
        assert_eq!(f.len(), 0);
        assert_eq!(DROPS.load(Ordering::SeqCst), 10);
        drop(f);
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
        #[derive(Clone)]
        struct Dropper;
        static DROPS: AtomicUsize = AtomicUsize::new(0);
        impl Drop for Dropper {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::SeqCst);
            }
        }
        let mut f: Fillet<Dropper> = repeat_n(Dropper, 5).collect();
        f.extend_from_within(1..4);
        assert_eq!(f.len(), 8);
        assert_eq!(DROPS.load(Ordering::SeqCst), 0);
        assert_eq!(f.len(), 8);
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

    /// `IntoIterator` for `&Fillet`.
    #[test]
    fn into_iter_ref() {
        let f: Fillet<i32> = [1, 2, 3].into();
        let mut sum = 0;
        for &x in &f {
            sum += x;
        }
        assert_eq!(sum, 6);
        assert_eq!(f.len(), 3);
    }

    /// `IntoIterator` for `&mut Fillet`.
    #[test]
    fn into_iter_mut_ref() {
        let mut f: Fillet<i32> = [1, 2, 3].into();
        for x in &mut f {
            *x += 10;
        }
        assert_eq!(f, [11, 12, 13]);
    }

    /// `IntoIterator` for `&Fillet<()>` (ZST).
    #[test]
    fn into_iter_ref_zst() {
        let f: Fillet<()> = [(); 3].into();
        let mut count = 0;
        for _ in &f {
            count += 1;
        }
        assert_eq!(count, 3);
    }

    /// `Extend<&T>` where `T: Copy`.
    #[test]
    fn extend_ref_copy() {
        let mut f: Fillet<i32> = [1, 2].into();
        let more = [3, 4, 5];
        f.extend(more.iter());
        assert_eq!(f, [1, 2, 3, 4, 5]);
    }

    /// `Extend<&T>` from another `Fillet`.
    #[test]
    fn extend_ref_from_fillet() {
        let mut f: Fillet<i32> = [1].into();
        let other: Fillet<i32> = [2, 3].into();
        f.extend(&other);
        assert_eq!(f, [1, 2, 3]);
    }

    /// Cross-type [`PartialEq`] with slices and arrays.
    #[test]
    fn partial_eq_cross_type() {
        let f: Fillet<i32> = [1, 2, 3].into();
        assert_eq!(f, [1, 2, 3]);
        assert_eq!(f, &[1, 2, 3]);
        assert_eq!(f, [1, 2, 3].as_slice());
        assert_eq!([1, 2, 3], f);
        assert_eq!(&[1, 2, 3], f);

        let arr = [1, 2, 3];
        assert_eq!(f, &arr);
        assert_eq!(&arr, f);

        let e: Fillet<i32> = Fillet::EMPTY;
        let empty: &[i32] = &[];
        assert_eq!(e, [0i32; 0]);
        assert_eq!(e, empty);
    }

    /// [`PartialOrd`] and [`Ord`].
    #[test]
    fn ordering() {
        let a: Fillet<i32> = [1, 2].into();
        let b: Fillet<i32> = [1, 3].into();
        let c: Fillet<i32> = [1, 2].into();
        assert!(a < b);
        assert!(b > a);
        assert_eq!(a.cmp(&c), core::cmp::Ordering::Equal);
        assert_eq!(a.partial_cmp(&b), Some(core::cmp::Ordering::Less));

        let empty: Fillet<i32> = Fillet::EMPTY;
        assert!(empty < a);
    }

    /// `Send` and `Sync` for [`FilletIntoIter`].
    #[test]
    fn into_iter_send_sync() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}
        assert_send::<FilletIntoIter<i32>>();
        assert_sync::<FilletIntoIter<i32>>();
    }

    /// `Debug` for [`FilletIntoIter`].
    #[test]
    fn into_iter_debug() {
        use alloc::format;
        let f: Fillet<i32> = [1, 2, 3].into();
        let mut iter = f.into_iter();
        assert_eq!(format!("{iter:?}"), "[1, 2, 3]");
        iter.next();
        assert_eq!(format!("{iter:?}"), "[2, 3]");
    }

    /// [`as_slice`] on [`FilletIntoIter`].
    ///
    /// [`as_slice`]: FilletIntoIter::as_slice
    #[test]
    fn into_iter_as_slice() {
        let f: Fillet<i32> = [1, 2, 3].into();
        let mut iter = f.into_iter();
        assert_eq!(iter.as_slice(), &[1, 2, 3]);
        iter.next();
        assert_eq!(iter.as_slice(), &[2, 3]);
        iter.next_back();
        assert_eq!(iter.as_slice(), &[2]);
    }

    /// [`as_mut_slice`] on [`FilletIntoIter`].
    ///
    /// [`as_mut_slice`]: FilletIntoIter::as_mut_slice
    #[test]
    fn into_iter_as_mut_slice() {
        let f: Fillet<i32> = [1, 2, 3].into();
        let mut iter = f.into_iter();
        iter.as_mut_slice()[0] = 10;
        assert_eq!(iter.next(), Some(10));
    }

    /// [`as_slice`] on [`FilletIntoIter`] with ZST.
    ///
    /// [`as_slice`]: FilletIntoIter::as_slice
    #[test]
    fn into_iter_as_slice_zst() {
        let f: Fillet<()> = [(); 3].into();
        let mut iter = f.into_iter();
        assert_eq!(iter.as_slice().len(), 3);
        iter.next();
        assert_eq!(iter.as_slice().len(), 2);
    }

    /// [`as_slice`] on empty [`FilletIntoIter`].
    ///
    /// [`as_slice`]: FilletIntoIter::as_slice
    #[test]
    fn into_iter_as_slice_empty() {
        let f: Fillet<i32> = Fillet::EMPTY;
        let iter = f.into_iter();
        assert_eq!(iter.as_slice(), &[]);
    }

    /// [`pop`] should work with ZSTs, including correct drop counts.
    ///
    /// [`pop`]: Fillet::pop
    #[test]
    fn pop_zst() {
        #[derive(Clone)]
        struct Dropper;
        static DROPS: AtomicUsize = AtomicUsize::new(0);
        impl Drop for Dropper {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::SeqCst);
            }
        }

        let mut f: Fillet<Dropper> = [Dropper, Dropper, Dropper].into();
        assert_eq!(f.len(), 3);
        let item = f.pop();
        assert!(item.is_some());
        assert_eq!(f.len(), 2);
        assert_eq!(DROPS.load(Ordering::SeqCst), 0);
        drop(item);
        assert_eq!(DROPS.load(Ordering::SeqCst), 1);
        assert!(f.pop().is_some());
        assert!(f.pop().is_some());
        assert!(f.pop().is_none());
        assert_eq!(f.len(), 0);
        drop(f);
        assert_eq!(DROPS.load(Ordering::SeqCst), 3);
    }

    /// [`truncate`] should work with ZSTs, triggering the correct number of drops.
    ///
    /// [`truncate`]: Fillet::truncate
    #[test]
    fn truncate_zst() {
        struct Dropper;
        static DROPS: AtomicUsize = AtomicUsize::new(0);
        impl Drop for Dropper {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::SeqCst);
            }
        }
        let mut f: Fillet<Dropper> = repeat_n((), 5).map(|_| Dropper).collect();
        f.truncate(2);
        assert_eq!(f.len(), 2);
        assert_eq!(DROPS.load(Ordering::SeqCst), 3);
        drop(f);
        assert_eq!(DROPS.load(Ordering::SeqCst), 5);
    }

    /// [`retain`] keeping all elements with a non-ZST exercises the fast-path prefix scan.
    ///
    /// [`retain`]: Fillet::retain
    #[test]
    fn retain_keep_all() {
        let mut f: Fillet<i32> = [1, 2, 3, 4, 5].into();
        f.retain(|_| true);
        assert_eq!(f, [1, 2, 3, 4, 5]);
    }

    /// [`extend_from_within`] with a non-zero-start range and a `Clone + Drop` type.
    ///
    /// [`extend_from_within`]: Fillet::extend_from_within
    #[test]
    fn extend_from_within_nonzero_drop_type() {
        use alloc::string::String;
        let mut f: Fillet<String> = ["a".into(), "b".into(), "c".into(), "d".into()].into();
        f.extend_from_within(1..3);
        assert_eq!(f.len(), 6);
        assert_eq!(f[4], "b");
        assert_eq!(f[5], "c");
    }

    /// [`extend_from_within`] with a range that exceeds the length is clamped.
    ///
    /// [`extend_from_within`]: Fillet::extend_from_within
    #[test]
    fn extend_from_within_clamped() {
        let mut f = Fillet::from([1, 2, 3]);
        f.extend_from_within(1..100);
        assert_eq!(f, [1, 2, 3, 2, 3]);

        let mut g = Fillet::from([1, 2]);
        g.extend_from_within(5..10); // entirely out of bounds
        assert_eq!(g, [1, 2]);
    }

    /// [`extend_from_within`] on an empty `Fillet` is a no-op.
    ///
    /// [`extend_from_within`]: Fillet::extend_from_within
    #[test]
    fn extend_from_within_empty() {
        let mut f: Fillet<i32> = Fillet::EMPTY;
        f.extend_from_within(..);
        assert!(f.is_empty());
    }

    /// [`extend`] with an iterator whose `size_hint` overstates the count.
    ///
    /// `size_hint` has no safety contract, so the exact-size path must reconcile the
    /// actual yield count and shrink the allocation if fewer items arrive.
    ///
    /// [`extend`]: Fillet::extend
    #[test]
    fn extend_inaccurate_size_hint() {
        static DROPS: AtomicUsize = AtomicUsize::new(0);
        #[allow(dead_code)]
        struct Dropper(u32); // Non-ZST to exercise the exact-size path.
        impl Drop for Dropper {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::SeqCst);
            }
        }

        struct ShortIter(usize);
        impl Iterator for ShortIter {
            type Item = Dropper;
            fn next(&mut self) -> Option<Dropper> {
                if self.0 == 0 {
                    return None;
                }
                self.0 -= 1;
                Some(Dropper(self.0 as u32))
            }
            fn size_hint(&self) -> (usize, Option<usize>) {
                (10, Some(10))
            }
        }

        let mut f: Fillet<Dropper> = Fillet::EMPTY;
        f.extend(ShortIter(3));
        assert_eq!(f.len(), 3);
        drop(f);
        assert_eq!(DROPS.load(Ordering::SeqCst), 3);
    }

    /// [`extend_from_within`] with a range that does not start at zero.
    ///
    /// [`extend_from_within`]: Fillet::extend_from_within
    #[test]
    fn extend_from_within_nonzero_start() {
        let mut f = Fillet::from([10, 20, 30, 40]);
        f.extend_from_within(2..4);
        assert_eq!(f, [10, 20, 30, 40, 30, 40]);

        let mut g = Fillet::from([1, 2, 3, 4, 5]);
        g.extend_from_within(3..5);
        assert_eq!(g, [1, 2, 3, 4, 5, 4, 5]);
    }

    /// [`extend_from_within`] with a `Clone + Drop` type.
    ///
    /// The source elements must not be invalidated during cloning.
    ///
    /// [`extend_from_within`]: Fillet::extend_from_within
    #[test]
    fn extend_from_within_clone_drop() {
        use alloc::string::String;
        let mut f: Fillet<String> = ["hello".into(), "world".into()].into();
        f.extend_from_within(..);
        assert_eq!(f.len(), 4);
        assert_eq!(f[0], "hello");
        assert_eq!(f[1], "world");
        assert_eq!(f[2], "hello");
        assert_eq!(f[3], "world");
    }

    /// [`retain`] with a panicking predicate must drop each element exactly once.
    ///
    /// [`retain`]: Fillet::retain
    #[test]
    fn retain_panic_drop_count() {
        extern crate std;
        use std::panic::{AssertUnwindSafe, catch_unwind};

        static DROPS: AtomicUsize = AtomicUsize::new(0);
        #[allow(dead_code)]
        struct Dropper(u32); // Non-ZST to exercise the heap path.
        impl Drop for Dropper {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::SeqCst);
            }
        }

        let result = catch_unwind(AssertUnwindSafe(|| {
            let mut f: Fillet<Dropper> = (0..5).map(Dropper).collect();
            let mut count = 0;
            f.retain(|_| {
                count += 1;
                if count == 3 {
                    panic!("predicate panic");
                }
                count % 2 == 1
            });
        }));

        assert!(result.is_err());
        assert_eq!(
            DROPS.load(Ordering::SeqCst),
            5,
            "each element must be dropped exactly once"
        );
    }

    /// [`truncate`] must not double-drop if `T::drop()` panics.
    ///
    /// [`truncate`]: Fillet::truncate
    #[test]
    fn truncate_drop_panic() {
        extern crate std;
        use std::panic::{AssertUnwindSafe, catch_unwind};

        static DROPS: AtomicUsize = AtomicUsize::new(0);
        struct PanicDrop(u32);
        impl Drop for PanicDrop {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::SeqCst);
                if self.0 == 3 {
                    panic!("drop panic");
                }
            }
        }

        // f = [PanicDrop(0), PanicDrop(1), PanicDrop(2), PanicDrop(3), PanicDrop(4)]
        // truncate(2) should drop [2, 3, 4]. PanicDrop(3) panics in drop.
        // drop_in_place's internal guard still drops 4.
        // On unwind, Fillet::drop must only drop the survivors [0, 1].
        let result = catch_unwind(AssertUnwindSafe(|| {
            let mut f: Fillet<PanicDrop> = (0..5).map(PanicDrop).collect();
            f.truncate(2);
        }));

        assert!(result.is_err());
        assert_eq!(
            DROPS.load(Ordering::SeqCst),
            5,
            "each element must be dropped exactly once"
        );
    }

    /// [`truncate`] must not double-drop ZSTs if `T::drop()` panics.
    ///
    /// [`truncate`]: Fillet::truncate
    #[test]
    fn truncate_drop_panic_zst() {
        extern crate std;
        use std::panic::{AssertUnwindSafe, catch_unwind};

        static DROPS: AtomicUsize = AtomicUsize::new(0);
        static PANIC_AT: AtomicUsize = AtomicUsize::new(3);
        struct PanicDrop;
        impl Drop for PanicDrop {
            fn drop(&mut self) {
                let n = DROPS.fetch_add(1, Ordering::SeqCst);
                if n == PANIC_AT.load(Ordering::SeqCst) {
                    panic!("drop panic");
                }
            }
        }

        let result = catch_unwind(AssertUnwindSafe(|| {
            let mut f: Fillet<PanicDrop> = repeat_n((), 5).map(|_| PanicDrop).collect();
            f.truncate(2);
        }));

        assert!(result.is_err());
        assert_eq!(
            DROPS.load(Ordering::SeqCst),
            5,
            "each element must be dropped exactly once"
        );
    }

    /// [`FilletIntoIter`] must free its allocation even if `T::drop()` panics.
    #[test]
    fn into_iter_drop_panic_no_leak() {
        extern crate std;
        use std::panic::{AssertUnwindSafe, catch_unwind};

        static DROPS: AtomicUsize = AtomicUsize::new(0);
        struct PanicDrop(u32);
        impl Drop for PanicDrop {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::SeqCst);
                if self.0 == 1 {
                    panic!("drop panic");
                }
            }
        }

        // Consume partially, then drop the iterator.
        // PanicDrop(1) panics during iterator drop.
        // All elements must still be dropped and the allocation freed.
        let result = catch_unwind(AssertUnwindSafe(|| {
            let f: Fillet<PanicDrop> = (0..4).map(PanicDrop).collect();
            let mut iter = f.into_iter();
            let _ = iter.next(); // consumes PanicDrop(0)
            drop(iter); // drops [1, 2, 3]; PanicDrop(1) panics
        }));

        assert!(result.is_err());
        assert_eq!(
            DROPS.load(Ordering::SeqCst),
            4,
            "each element must be dropped exactly once"
        );
        // Allocation leak is detectable by Miri but not by drop counts.
        // Run under Miri to verify no leak.
    }

    /// [`extend`] exact-size path must not drop uninitialized slots if `iter.next()` panics.
    ///
    /// [`extend`]: Fillet::extend
    #[test]
    fn extend_exact_size_next_panic() {
        extern crate std;
        use std::panic::{AssertUnwindSafe, catch_unwind};

        static DROPS: AtomicUsize = AtomicUsize::new(0);
        #[allow(dead_code)]
        struct Dropper(u32);
        impl Drop for Dropper {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::SeqCst);
            }
        }

        struct PanicIter(u32);
        impl Iterator for PanicIter {
            type Item = Dropper;
            fn next(&mut self) -> Option<Dropper> {
                let i = self.0;
                self.0 += 1;
                if i == 3 {
                    panic!("next panic");
                }
                Some(Dropper(i))
            }
            fn size_hint(&self) -> (usize, Option<usize>) {
                (6, Some(6))
            }
        }

        let result = catch_unwind(AssertUnwindSafe(|| {
            let mut f: Fillet<Dropper> = Fillet::from_one(Dropper(100));
            f.extend(PanicIter(0)); // writes 3 items then panics
        }));

        assert!(result.is_err());
        // 1 pre-existing + 3 written by extend = 4 total must be dropped.
        assert_eq!(
            DROPS.load(Ordering::SeqCst),
            4,
            "only initialized elements must be dropped"
        );
    }

    /// [`extend`] amortized path must not drop uninitialized slots if `iter.next()` panics.
    ///
    /// [`extend`]: Fillet::extend
    #[test]
    fn extend_unknown_size_next_panic() {
        extern crate std;
        use std::panic::{AssertUnwindSafe, catch_unwind};

        static DROPS: AtomicUsize = AtomicUsize::new(0);
        #[allow(dead_code)]
        struct Dropper(u32);
        impl Drop for Dropper {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::SeqCst);
            }
        }

        struct PanicIter(u32);
        impl Iterator for PanicIter {
            type Item = Dropper;
            fn next(&mut self) -> Option<Dropper> {
                let i = self.0;
                self.0 += 1;
                if i == 3 {
                    panic!("next panic");
                }
                Some(Dropper(i))
            }
            // Unknown size — no upper bound.
            fn size_hint(&self) -> (usize, Option<usize>) {
                (0, None)
            }
        }

        let result = catch_unwind(AssertUnwindSafe(|| {
            let mut f: Fillet<Dropper> = Fillet::from_one(Dropper(100));
            f.extend(PanicIter(0));
        }));

        assert!(result.is_err());
        assert_eq!(
            DROPS.load(Ordering::SeqCst),
            4,
            "only initialized elements must be dropped"
        );
    }

    /// [`extend_from_within`] must not drop uninitialized slots if `T::clone()` panics.
    ///
    /// [`extend_from_within`]: Fillet::extend_from_within
    #[test]
    fn extend_from_within_clone_panic() {
        extern crate std;
        use std::panic::{AssertUnwindSafe, catch_unwind};

        static DROPS: AtomicUsize = AtomicUsize::new(0);
        static CLONES: AtomicUsize = AtomicUsize::new(0);
        #[allow(dead_code)]
        struct PanicClone(u32);
        impl Clone for PanicClone {
            fn clone(&self) -> Self {
                if CLONES.fetch_add(1, Ordering::SeqCst) == 2 {
                    panic!("clone panic");
                }
                PanicClone(self.0)
            }
        }
        impl Drop for PanicClone {
            fn drop(&mut self) {
                DROPS.fetch_add(1, Ordering::SeqCst);
            }
        }

        let result = catch_unwind(AssertUnwindSafe(|| {
            let mut f: Fillet<PanicClone> =
                [PanicClone(0), PanicClone(1), PanicClone(2), PanicClone(3)].into();
            f.extend_from_within(..); // panics on 3rd clone
        }));

        assert!(result.is_err());
        // 4 originals + 2 successful clones = 6 drops.
        assert_eq!(
            DROPS.load(Ordering::SeqCst),
            6,
            "only initialized elements must be dropped"
        );
    }
}
