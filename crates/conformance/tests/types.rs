use ducktape_conformance::{fixtures_dir, load_samples, roundtrip_def};

#[test]
fn type_samples_round_trip() {
    let samples = load_samples(&fixtures_dir().join("types"));
    assert!(!samples.is_empty());
    let mut failures = Vec::new();
    for (name, sample) in &samples {
        match roundtrip_def(&sample.def, &sample.value) {
            Ok(out) if out == sample.value => {}
            Ok(out) => failures.push(format!(
                "{name}: round-trip changed the value\n  in:  {}\n  out: {}",
                sample.value, out
            )),
            Err(e) => failures.push(format!("{name}: {e}")),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
