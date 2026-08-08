//! Explicit bounded execution contexts for native `PhaseSmith` kernels.

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon::{ThreadPool, ThreadPoolBuildError, ThreadPoolBuilder};

/// Reusable, operation-owned worker pool with deterministic ordered mapping.
///
/// The context never installs or mutates Rayon's global pool. A one-thread
/// context executes closures directly, which also prevents nested pools when
/// Python already schedules independent native batches.
pub struct ExecutionContext {
    threads: usize,
    pool: Option<ThreadPool>,
}

impl ExecutionContext {
    /// Construct a context with an exact positive worker budget.
    ///
    /// # Errors
    ///
    /// Returns [`ThreadPoolBuildError`] when Rayon cannot build the bounded
    /// worker pool.
    pub fn new(threads: usize) -> Result<Self, ThreadPoolBuildError> {
        let pool = if threads <= 1 {
            None
        } else {
            Some(
                ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .thread_name(|index| format!("phasesmith-native-{index}"))
                    .build()?,
            )
        };
        Ok(Self {
            threads: threads.max(1),
            pool,
        })
    }

    /// Return a zero-allocation serial context.
    #[must_use]
    pub const fn serial() -> Self {
        Self {
            threads: 1,
            pool: None,
        }
    }

    /// Return the exact worker budget owned by this context.
    #[must_use]
    pub const fn threads(&self) -> usize {
        self.threads
    }

    /// Map independent indices while collecting results in increasing order.
    ///
    /// Work below `minimum_parallel_items` remains serial. Indexed Rayon
    /// collection preserves the input order independently of completion order.
    pub fn map_ordered<R, F>(
        &self,
        item_count: usize,
        minimum_parallel_items: usize,
        operation: F,
    ) -> Vec<R>
    where
        R: Send,
        F: Fn(usize) -> R + Send + Sync,
    {
        if let Some(pool) = &self.pool
            && item_count >= minimum_parallel_items
        {
            return pool.install(|| (0..item_count).into_par_iter().map(operation).collect());
        }
        (0..item_count).map(operation).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn ordered_mapping_is_identical_across_worker_counts() {
        let serial = ExecutionContext::serial();
        let parallel = ExecutionContext::new(3).expect("pool");
        let operation = |index| {
            thread::sleep(Duration::from_micros(((7 - index) % 4) as u64));
            index * index
        };
        assert_eq!(
            serial.map_ordered(8, 2, operation),
            parallel.map_ordered(8, 2, operation)
        );
        assert_eq!(parallel.threads(), 3);
    }

    #[test]
    fn threshold_keeps_small_work_serial_and_ordered() {
        let context = ExecutionContext::new(2).expect("pool");
        assert_eq!(context.map_ordered(3, 4, |index| index + 1), [1, 2, 3]);
    }
}
