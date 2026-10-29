use ducktape_abi::{
    ducktape_abi_version, ducktape_free, ducktape_op_free, ducktape_op_resume, ducktape_op_start,
};
use ducktape_core::effects::{Effect, OperationResult, Step};

unsafe fn call_start(json: &str) -> (*mut ducktape_core::Operation, Step) {
    let mut out: *mut u8 = std::ptr::null_mut();
    let mut out_len = 0usize;
    let op = ducktape_op_start(json.as_ptr(), json.len(), &mut out, &mut out_len);
    let step = serde_json::from_slice(std::slice::from_raw_parts(out, out_len)).unwrap();
    ducktape_free(out, out_len);
    (op, step)
}

const CLOCK: &str =
    r#"{"now_ms": 1761696000000, "entropy": "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA="}"#;

#[test]
fn abi_version_is_one() {
    assert_eq!(ducktape_abi_version(), 1);
}

#[test]
fn auth_status_runs_through_the_abi() {
    let json = format!(
        r#"{{"clock": {CLOCK}, "provider": {{"ns": "openai"}}, "operation": {{"type": "auth_status"}}}}"#
    );
    unsafe {
        let (op, step) = call_start(&json);
        assert!(!op.is_null());
        assert!(matches!(
            step.effect,
            Effect::Done {
                result: OperationResult::AuthStatus { .. }
            }
        ));
        ducktape_op_free(op);
    }
}

#[test]
fn invalid_json_fails_without_a_handle() {
    unsafe {
        let (op, step) = call_start("{not json");
        assert!(op.is_null());
        assert!(matches!(step.effect, Effect::Failed { .. }));
    }
}

#[test]
fn unknown_namespace_fails_with_unsupported() {
    let json = format!(
        r#"{{"clock": {CLOCK}, "provider": {{"ns": "acme"}}, "operation": {{"type": "models"}}}}"#
    );
    unsafe {
        let (op, step) = call_start(&json);
        match step.effect {
            Effect::Failed { error } => assert_eq!(error.kind.as_str(), "unsupported"),
            other => panic!("{other:?}"),
        }
        let resume = format!(r#"{{"clock": {CLOCK}, "input": {{"type": "cancelled"}}}}"#);
        let mut out: *mut u8 = std::ptr::null_mut();
        let mut out_len = 0usize;
        assert_eq!(
            ducktape_op_resume(op, resume.as_ptr(), resume.len(), &mut out, &mut out_len),
            0
        );
        ducktape_free(out, out_len);
        ducktape_op_free(op);
    }
}
