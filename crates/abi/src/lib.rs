//! `extern "C"` boundary: JSON UTF-8 in and out, opaque handles, no callbacks.
#![allow(clippy::missing_safety_doc)]
use ducktape_core::effects::{Clock, Resume, Start, Step};
use ducktape_core::types::{ErrorKind, ProviderError};
use ducktape_core::Operation;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::slice;

/// Version of this ABI; bumped on any breaking change to exports or JSON.
pub const ABI_VERSION: u32 = 1;

/// Returns the ABI version.
#[no_mangle]
pub extern "C" fn ducktape_abi_version() -> u32 {
    ABI_VERSION
}

/// Allocates `len` bytes the host may write into; free with `ducktape_free`.
#[no_mangle]
pub extern "C" fn ducktape_alloc(len: usize) -> *mut u8 {
    if len == 0 {
        return std::ptr::null_mut();
    }
    let mut buf = Vec::<u8>::with_capacity(len);
    let ptr = buf.as_mut_ptr();
    std::mem::forget(buf);
    ptr
}

/// Frees a buffer of `len` bytes that came from `ducktape_alloc` or an out parameter.
#[no_mangle]
pub unsafe extern "C" fn ducktape_free(ptr: *mut u8, len: usize) {
    if !ptr.is_null() && len != 0 {
        drop(Vec::from_raw_parts(ptr, len, len));
    }
}

fn give(bytes: Vec<u8>, out: *mut *mut u8, out_len: *mut usize) {
    let mut bytes = bytes;
    bytes.shrink_to_fit();
    let len = bytes.len();
    let ptr = ducktape_alloc(len);
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, len);
        *out = ptr;
        *out_len = len;
    }
}

fn give_step(step: &Step, out: *mut *mut u8, out_len: *mut usize) {
    give(
        serde_json::to_vec(step).expect("step serializes"),
        out,
        out_len,
    );
}

fn parse_error(message: String) -> Step {
    Step::failed(ProviderError::new(ErrorKind::BadRequest, message))
}

fn guarded(f: impl FnOnce() -> Step) -> Step {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|panic| {
        let detail = panic
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| panic.downcast_ref::<String>().cloned())
            .unwrap_or_default();
        Step::failed(ProviderError::new(
            ErrorKind::Other("internal".to_owned()),
            format!("internal error: {detail}"),
        ))
    })
}

unsafe fn input<'a>(ptr: *const u8, len: usize) -> &'a [u8] {
    if ptr.is_null() || len == 0 {
        &[]
    } else {
        slice::from_raw_parts(ptr, len)
    }
}

/// Starts an operation from a `Start` document and writes the first `Step`; null handle on an invalid document.
#[no_mangle]
pub unsafe extern "C" fn ducktape_op_start(
    json: *const u8,
    len: usize,
    out: *mut *mut u8,
    out_len: *mut usize,
) -> *mut Operation {
    match serde_json::from_slice::<Start>(input(json, len)) {
        Ok(start) => {
            let mut op = None;
            let step = guarded(|| {
                let (started, step) = Operation::start(start);
                op = Some(started);
                step
            });
            give_step(&step, out, out_len);
            match op {
                Some(op) => Box::into_raw(Box::new(op)),
                None => std::ptr::null_mut(),
            }
        }
        Err(e) => {
            give_step(
                &parse_error(format!("invalid start document: {e}")),
                out,
                out_len,
            );
            std::ptr::null_mut()
        }
    }
}

/// Resumes an operation with a `Resume` document and writes the next `Step`.
#[no_mangle]
pub unsafe extern "C" fn ducktape_op_resume(
    op: *mut Operation,
    json: *const u8,
    len: usize,
    out: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    let Some(op) = op.as_mut() else { return -1 };
    match serde_json::from_slice::<Resume>(input(json, len)) {
        Ok(resume) => {
            give_step(&guarded(|| op.resume(resume)), out, out_len);
            0
        }
        Err(e) => {
            give_step(
                &parse_error(format!("invalid resume document: {e}")),
                out,
                out_len,
            );
            -2
        }
    }
}

/// Delivers raw body bytes of the open response with a `Clock` document and writes the next `Step`.
#[no_mangle]
pub unsafe extern "C" fn ducktape_op_feed(
    op: *mut Operation,
    bytes: *const u8,
    bytes_len: usize,
    clock_json: *const u8,
    clock_len: usize,
    out: *mut *mut u8,
    out_len: *mut usize,
) -> i32 {
    let Some(op) = op.as_mut() else { return -1 };
    match serde_json::from_slice::<Clock>(input(clock_json, clock_len)) {
        Ok(clock) => {
            let bytes = input(bytes, bytes_len);
            give_step(&guarded(|| op.feed(bytes, clock)), out, out_len);
            0
        }
        Err(e) => {
            give_step(
                &parse_error(format!("invalid clock document: {e}")),
                out,
                out_len,
            );
            -2
        }
    }
}

/// Stores a secret under `secret_ref`; the core copies the bytes and the caller wipes its own.
#[no_mangle]
pub unsafe extern "C" fn ducktape_op_set_secret(
    op: *mut Operation,
    secret_ref: *const u8,
    ref_len: usize,
    bytes: *const u8,
    len: usize,
) -> i32 {
    let Some(op) = op.as_mut() else { return -1 };
    let Ok(secret_ref) = std::str::from_utf8(input(secret_ref, ref_len)) else {
        return -2;
    };
    op.set_secret(secret_ref, input(bytes, len).to_vec());
    0
}

/// Frees an operation handle.
#[no_mangle]
pub unsafe extern "C" fn ducktape_op_free(op: *mut Operation) {
    if !op.is_null() {
        drop(Box::from_raw(op));
    }
}
