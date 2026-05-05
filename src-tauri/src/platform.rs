//! Platform-specific utilities.
//!
//! FD-003 M3.5 #8 — macOS P-core binding via QoS class.

/// Elevate the calling thread to `QOS_CLASS_USER_INTERACTIVE` so the macOS
/// scheduler preferentially assigns it to P-cores (performance cores).
///
/// On Apple Silicon M-series chips, P-cores run at ≈2-3× the frequency of
/// E-cores.  ASR inference threads running on E-cores therefore suffer a
/// 50-70% throughput penalty.  This call pins the worker to the highest QoS
/// class so the scheduler keeps it on P-cores.
///
/// **Observability**: logs the *before* QoS class (as numeric value) so it
/// can be compared against expectations in production logs.
///
/// ## Safety
/// `pthread_set_qos_class_self_np` is a documented macOS POSIX extension.
/// The call is safe to make from any thread at any time; it only changes
/// the calling thread's scheduling hint, not any shared state.
///
/// No-op on non-macOS platforms.
#[allow(unused_variables)]
pub fn elevate_thread_qos(worker_name: &str) {
    #[cfg(target_os = "macos")]
    {
        use libc::{
            pthread_get_qos_class_np, pthread_self, pthread_set_qos_class_self_np,
            qos_class_t::QOS_CLASS_USER_INTERACTIVE,
        };

        // Safety: pthread_get/set_qos_class_np are documented macOS extensions
        // that only affect the calling thread's scheduling hint.
        unsafe {
            let mut before_class = libc::qos_class_t::QOS_CLASS_UNSPECIFIED;
            let mut relative_priority: libc::c_int = 0;
            pthread_get_qos_class_np(
                pthread_self(),
                &mut before_class,
                &mut relative_priority,
            );
            log::debug!(
                "QoS [{worker_name}] before={before_class:?} relative_priority={relative_priority}"
            );

            let ret = pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
            if ret != 0 {
                log::warn!("QoS [{worker_name}] pthread_set_qos_class_self_np failed: {ret}");
            } else {
                log::debug!("QoS [{worker_name}] elevated to QOS_CLASS_USER_INTERACTIVE");
            }
        }
    }
}
