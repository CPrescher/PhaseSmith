//! Explicit bounded execution contexts for native `PhaseSmith` kernels.

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon::{ThreadPool, ThreadPoolBuildError, ThreadPoolBuilder};

/// Reusable, operation-owned worker pool with deterministic ordered mapping.
///
/// The context never installs or mutates Rayon's global pool. A one-thread
/// context executes closures directly, which also prevents nested pools when
/// Python already schedules independent native batches.
#[derive(Clone)]
pub struct ExecutionContext {
    threads: usize,
    pool: Option<Arc<ThreadPool>>,
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
            Some(Arc::new(
                ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .thread_name(|index| format!("phasesmith-native-{index}"))
                    .build()?,
            ))
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

/// Default fixed worker budget used by scripts and applications.
pub const DEFAULT_EXECUTION_THREADS: usize = 2;
/// Default number of independent tasks required before parallel scheduling.
pub const DEFAULT_MINIMUM_PARALLEL_TASKS: usize = 2;

/// Invalid or unconstructable native execution policy.
#[derive(Debug)]
pub enum ExecutionPolicyError {
    /// A fixed worker budget was zero.
    InvalidThreadCount,
    /// The parallel scheduling threshold was zero.
    InvalidMinimumParallelTasks,
    /// Rayon could not construct the bounded worker pool.
    ThreadPool(ThreadPoolBuildError),
}

impl Display for ExecutionPolicyError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidThreadCount => {
                formatter.write_str("threads must be automatic or a positive integer")
            }
            Self::InvalidMinimumParallelTasks => {
                formatter.write_str("minimum_parallel_tasks must be a positive integer")
            }
            Self::ThreadPool(error) => Display::fmt(error, formatter),
        }
    }
}

impl Error for ExecutionPolicyError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::ThreadPool(error) => Some(error),
            Self::InvalidThreadCount | Self::InvalidMinimumParallelTasks => None,
        }
    }
}

/// Immutable worker-budget policy with one persistent bounded execution pool.
///
/// `requested_threads = None` selects the available logical CPU count. Fixed
/// budgets are capped to that count. Cloned execution contexts share the same
/// Rayon pool, so prepared phases and workflow operations do not rebuild or
/// nest worker pools.
#[derive(Clone)]
pub struct ExecutionPolicy {
    requested_threads: Option<usize>,
    minimum_parallel_tasks: usize,
    context: ExecutionContext,
}

impl std::fmt::Debug for ExecutionPolicy {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ExecutionPolicy")
            .field("requested_threads", &self.requested_threads)
            .field("minimum_parallel_tasks", &self.minimum_parallel_tasks)
            .field(
                "context",
                &format_args!("{} threads", self.resolved_budget()),
            )
            .finish()
    }
}

impl PartialEq for ExecutionPolicy {
    fn eq(&self, other: &Self) -> bool {
        self.requested_threads == other.requested_threads
            && self.minimum_parallel_tasks == other.minimum_parallel_tasks
    }
}

impl Eq for ExecutionPolicy {}

impl ExecutionPolicy {
    /// Construct and retain the bounded execution context for this policy.
    ///
    /// # Errors
    ///
    /// Returns [`ExecutionPolicyError`] for zero budgets/thresholds or when
    /// Rayon cannot build the resolved worker pool.
    pub fn new(
        requested_threads: Option<usize>,
        minimum_parallel_tasks: usize,
    ) -> Result<Self, ExecutionPolicyError> {
        if requested_threads == Some(0) {
            return Err(ExecutionPolicyError::InvalidThreadCount);
        }
        if minimum_parallel_tasks == 0 {
            return Err(ExecutionPolicyError::InvalidMinimumParallelTasks);
        }
        let available_threads = std::thread::available_parallelism().map_or(1, usize::from);
        let resolved_threads = requested_threads
            .unwrap_or(available_threads)
            .min(available_threads)
            .max(1);
        let context =
            ExecutionContext::new(resolved_threads).map_err(ExecutionPolicyError::ThreadPool)?;
        Ok(Self {
            requested_threads,
            minimum_parallel_tasks,
            context,
        })
    }

    /// Construct the bounded two-thread default policy.
    ///
    /// # Errors
    ///
    /// Returns [`ExecutionPolicyError`] when Rayon cannot build the pool.
    pub fn bounded_default() -> Result<Self, ExecutionPolicyError> {
        Self::new(
            Some(DEFAULT_EXECUTION_THREADS),
            DEFAULT_MINIMUM_PARALLEL_TASKS,
        )
    }

    /// Return the requested fixed worker count, or `None` for automatic mode.
    #[must_use]
    pub const fn requested_threads(&self) -> Option<usize> {
        self.requested_threads
    }

    /// Return the independent-task threshold for parallel scheduling.
    #[must_use]
    pub const fn minimum_parallel_tasks(&self) -> usize {
        self.minimum_parallel_tasks
    }

    /// Return the resolved logical-CPU budget retained by this policy.
    #[must_use]
    pub const fn resolved_budget(&self) -> usize {
        self.context.threads()
    }

    /// Return the worker count for a known number of independent tasks.
    #[must_use]
    pub fn worker_count(&self, task_count: usize) -> usize {
        if task_count < self.minimum_parallel_tasks {
            return 1;
        }
        self.resolved_budget().min(task_count).max(1)
    }

    /// Borrow the persistent bounded execution context.
    #[must_use]
    pub const fn context(&self) -> &ExecutionContext {
        &self.context
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

    #[test]
    fn policy_validates_resolves_and_reuses_its_context() {
        let available = thread::available_parallelism().map_or(1, usize::from);
        let policy = ExecutionPolicy::new(Some(available + 3), 3).expect("policy");
        assert_eq!(policy.requested_threads(), Some(available + 3));
        assert_eq!(policy.minimum_parallel_tasks(), 3);
        assert_eq!(policy.resolved_budget(), available);
        assert_eq!(policy.worker_count(2), 1);
        assert_eq!(policy.worker_count(3), available.min(3));
        assert_eq!(policy.context().threads(), available);

        let shared_context = policy.context().clone();
        assert_eq!(
            policy.context().map_ordered(4, 2, |index| index * 2),
            shared_context.map_ordered(4, 2, |index| index * 2)
        );
    }

    #[test]
    fn policy_rejects_zero_configuration_values() {
        assert!(matches!(
            ExecutionPolicy::new(Some(0), 2),
            Err(ExecutionPolicyError::InvalidThreadCount)
        ));
        assert!(matches!(
            ExecutionPolicy::new(Some(1), 0),
            Err(ExecutionPolicyError::InvalidMinimumParallelTasks)
        ));
    }
}
