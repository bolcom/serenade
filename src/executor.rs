//! Optional bounded pool for blocking session storage and prediction work.
use futures::channel::oneshot;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;

type Job = Box<dyn FnOnce() + Send + 'static>;

pub struct BlockingExecutor {
    sender: Option<mpsc::SyncSender<Job>>,
    workers: Vec<thread::JoinHandle<()>>,
}

impl BlockingExecutor {
    pub fn new(threads: usize, queue_capacity: usize) -> anyhow::Result<Self> {
        anyhow::ensure!(
            threads > 0 && queue_capacity > 0,
            "blocking threads and queue capacity must be positive"
        );
        let (sender, receiver) = mpsc::sync_channel::<Job>(queue_capacity);
        let receiver = Arc::new(Mutex::new(receiver));
        let mut pool = Self {
            sender: Some(sender),
            workers: Vec::new(),
        };
        for index in 0..threads {
            let receiver = receiver.clone();
            pool.workers.push(
                thread::Builder::new()
                    .name(format!("prediction-{}", index))
                    .spawn(move || loop {
                        // Release the receiver lock before running the job.
                        let job = receiver.lock().unwrap().recv();
                        match job {
                            Ok(job) => {
                                let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job));
                            }
                            Err(_) => break,
                        }
                    })?,
            );
        }
        Ok(pool)
    }

    /// Submission never blocks the HTTP worker. Full queues reject immediately.
    pub fn submit<F, T>(&self, work: F) -> anyhow::Result<oneshot::Receiver<T>>
    where
        F: FnOnce() -> T + Send + 'static,
        T: Send + 'static,
    {
        let (sender, receiver) = oneshot::channel();
        let job = Box::new(move || {
            let _ = sender.send(work());
        });
        self.sender
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("executor is shut down"))?
            .try_send(job)
            .map_err(|error| match error {
                mpsc::TrySendError::Full(_) => anyhow::anyhow!("prediction queue is full"),
                mpsc::TrySendError::Disconnected(_) => anyhow::anyhow!("executor is shut down"),
            })?;
        Ok(receiver)
    }
}

impl Drop for BlockingExecutor {
    fn drop(&mut self) {
        // Disconnect, drain accepted jobs, then join all workers.
        self.sender.take();
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounds_queue_and_drains_accepted_work() {
        let pool = BlockingExecutor::new(1, 1).unwrap();
        let (started_tx, started_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let first = pool
            .submit(move || {
                started_tx.send(()).unwrap();
                release_rx.recv().unwrap();
                1
            })
            .unwrap();
        started_rx.recv().unwrap();
        let second = pool.submit(|| 2).unwrap();
        assert!(pool.submit(|| 3).is_err());
        release_tx.send(()).unwrap();
        drop(pool);
        assert_eq!(futures::executor::block_on(first).unwrap(), 1);
        assert_eq!(futures::executor::block_on(second).unwrap(), 2);
    }
}
