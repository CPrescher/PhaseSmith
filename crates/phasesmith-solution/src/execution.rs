//! Call-scoped bounded execution; no process-global Rayon pool or numerical state.
use crate::{SolutionError, error};
use phasesmith_execution::ExecutionContext;
use std::{
    cell::{Cell, RefCell},
    sync::{
        Arc, OnceLock,
        atomic::{AtomicUsize, Ordering},
    },
};

thread_local! {
    static CURRENT: RefCell<Option<Arc<Workers>>> = const { RefCell::new(None) };
    static PARALLEL_ANCESTOR: Cell<bool> = const { Cell::new(false) };
    static TASK_DEPTH: Cell<usize> = const { Cell::new(0) };
}
struct Scope {
    previous: Option<Arc<Workers>>,
    ancestor: bool,
}
impl Scope {
    fn enter(next: Option<Arc<Workers>>) -> Self {
        let ancestor = PARALLEL_ANCESTOR.with(|p| p.replace(p.get() || next.is_some()));
        Self {
            previous: CURRENT.with(|c| c.replace(next)),
            ancestor,
        }
    }
}
impl Drop for Scope {
    fn drop(&mut self) {
        CURRENT.with(|c| c.replace(self.previous.take()));
        PARALLEL_ANCESTOR.with(|p| p.set(self.ancestor));
    }
}
struct Active(Arc<Workers>);
impl Active {
    fn enter(workers: Arc<Workers>) -> Self {
        TASK_DEPTH.with(|depth| {
            let n = depth.get();
            depth.set(n + 1);
            if n == 0 {
                let active = workers.active.fetch_add(1, Ordering::Relaxed) + 1;
                workers.maximum.fetch_max(active, Ordering::Relaxed);
            }
        });
        Self(workers)
    }
}
impl Drop for Active {
    fn drop(&mut self) {
        TASK_DEPTH.with(|depth| {
            let n = depth.get();
            depth.set(n - 1);
            if n == 1 {
                self.0.active.fetch_sub(1, Ordering::Relaxed);
            }
        });
    }
}
/// Actual execution of one bounded numerical call.
#[derive(Clone, Copy, Debug, Default)]
pub struct WorkerStats {
    /// Caller-requested positive worker limit.
    pub requested: usize,
    /// Worker limit capped to available logical parallelism.
    pub resolved: usize,
    /// Parallel ordered maps actually used.
    pub maps: usize,
    /// Tasks submitted to those maps.
    pub tasks: usize,
    /// Peak distinct active pool threads; one for a serial call.
    pub maximum_active: usize,
}
pub(crate) struct Workers {
    context: OnceLock<Result<ExecutionContext, String>>,
    resolved: usize,
    maps: AtomicUsize,
    tasks: AtomicUsize,
    active: AtomicUsize,
    maximum: AtomicUsize,
}
impl Workers {
    pub(crate) fn count(&self) -> usize {
        self.resolved
    }
    pub(crate) fn map<R: Send, F: Fn(usize) -> R + Send + Sync>(
        self: &Arc<Self>,
        n: usize,
        f: F,
    ) -> Result<Vec<R>, SolutionError> {
        if n < 2 || self.count() == 1 {
            return Ok((0..n).map(f).collect());
        }
        let context = self
            .context
            .get_or_init(|| ExecutionContext::new(self.resolved).map_err(|e| e.to_string()))
            .as_ref()
            .map_err(|e| error(e.clone()))?;
        self.maps.fetch_add(1, Ordering::Relaxed);
        self.tasks.fetch_add(n, Ordering::Relaxed);
        Ok(context.map_ordered(n, 2, |i| {
            let _scope = Scope::enter(Some(self.clone()));
            let _active = Active::enter(self.clone());
            f(i)
        }))
    }
}
pub(crate) fn current() -> Option<Arc<Workers>> {
    CURRENT.with(|c| c.borrow().clone())
}

/// Run a synchronous operation under one reusable, owned worker budget.
/// Pools are released after all tasks join; returned values retain no pool.
/// A serial nested scope is permitted. Nested parallel roots are rejected;
/// internal ordered maps share the existing pool without spawning another.
/// # Errors
/// Rejects nonpositive/over64 budgets, nested parallel roots and pool failures,
/// and propagates the operation's error. Panic unwinding restores the prior scope.
pub fn with_solution_workers<T>(
    workers: usize,
    operation: impl FnOnce() -> Result<T, SolutionError>,
) -> Result<(T, WorkerStats), SolutionError> {
    if !(1..=64).contains(&workers) {
        return Err(error("workers must lie in [1,64]"));
    }
    if workers > 1 && PARALLEL_ANCESTOR.with(Cell::get) {
        return Err(error("nested parallel solution roots require workers=1"));
    }
    let available = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    let resolved = workers.min(available);
    let context = Arc::new(Workers {
        context: OnceLock::new(),
        resolved,
        maps: AtomicUsize::new(0),
        tasks: AtomicUsize::new(0),
        active: AtomicUsize::new(0),
        maximum: AtomicUsize::new(0),
    });
    let _scope = Scope::enter((resolved > 1).then(|| context.clone()));
    let result = operation()?;
    Ok((
        result,
        WorkerStats {
            requested: workers,
            resolved,
            maps: context.maps.load(Ordering::Relaxed),
            tasks: context.tasks.load(Ordering::Relaxed),
            maximum_active: context.maximum.load(Ordering::Relaxed).max(1),
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validation_and_scope_restoration() {
        for invalid in [0, 65] {
            assert!(with_solution_workers(invalid, || Ok(())).is_err());
        }
        let ((), stats) = with_solution_workers(1, || {
            assert!(current().is_none());
            Ok(())
        })
        .unwrap();
        assert_eq!(stats.resolved, 1);
        let ((), stats) = with_solution_workers(4, || {
            if let Some(pool) = current() {
                assert!(pool.context.get().is_none());
                assert!(with_solution_workers(2, || Ok(())).is_err());
                with_solution_workers(1, || {
                    assert!(current().is_none());
                    assert!(with_solution_workers(2, || Ok(())).is_err());
                    Ok(())
                })?;
                assert!(Arc::ptr_eq(&pool, &current().unwrap()));
                let rows = pool.map(64, |i| {
                    let nested = current().unwrap();
                    assert!(Arc::ptr_eq(&pool, &nested));
                    nested.map(8, |j| i * 8 + j).unwrap()
                })?;
                assert_eq!(
                    rows.into_iter().flatten().collect::<Vec<_>>(),
                    (0..512).collect::<Vec<_>>()
                );
            }
            Ok(())
        })
        .unwrap();
        assert!(stats.maximum_active <= stats.resolved);
        assert!(current().is_none());
        let failed = with_solution_workers(2, || Err::<(), _>(error("test")));
        assert!(failed.is_err());
        assert!(current().is_none());
        let panicked = std::panic::catch_unwind(|| {
            let _ =
                with_solution_workers(2, || -> Result<(), SolutionError> { panic!("unwind test") });
        });
        assert!(panicked.is_err());
        assert!(current().is_none());
    }
    #[test]
    fn pool_is_lazy_and_released_after_the_call() {
        let (weak, _) = with_solution_workers(2, || {
            let Some(pool) = current() else {
                return Ok(None);
            };
            assert!(pool.context.get().is_none());
            let weak = Arc::downgrade(&pool);
            assert_eq!(pool.map(4, |i| i)?, [0, 1, 2, 3]);
            assert!(pool.context.get().is_some());
            Ok(Some(weak))
        })
        .unwrap();
        if let Some(weak) = weak {
            assert!(weak.upgrade().is_none());
        }
    }
    #[test]
    fn independent_roots_are_thread_local() {
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..2)
                .map(|_| {
                    scope.spawn(|| {
                        with_solution_workers(2, || {
                            if let Some(pool) = current() {
                                assert_eq!(
                                    pool.map(16, |i| i * i)?,
                                    (0..16).map(|i| i * i).collect::<Vec<_>>()
                                );
                            }
                            Ok(())
                        })
                        .unwrap()
                    })
                })
                .collect();
            for handle in handles {
                assert!(handle.join().is_ok());
            }
        });
        assert!(current().is_none());
    }
}
