use ducktape_conformance::fixtures_dir;
use ducktape_conformance::replay::{normalize, run, WireFixture};
use std::path::PathBuf;

fn fixtures() -> Vec<PathBuf> {
    let mut out = Vec::new();
    for wire in std::fs::read_dir(fixtures_dir().join("wire")).expect("fixtures/wire") {
        let dir = wire.expect("entry").path();
        if !dir.is_dir() {
            continue;
        }
        for f in std::fs::read_dir(&dir).expect("wire dir") {
            let p = f.expect("entry").path();
            if p.extension().is_some_and(|x| x == "json") {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

#[test]
fn wire_fixtures_replay() {
    let update = std::env::var("UPDATE_FIXTURES").is_ok();
    let mut failures = Vec::new();
    let paths = fixtures();
    assert!(!paths.is_empty());
    for path in paths {
        let text = std::fs::read_to_string(&path).expect("read");
        let mut fixture: WireFixture =
            serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let actual = run(&fixture);
        if update {
            fixture.exchange = actual;
            std::fs::write(
                &path,
                serde_json::to_string_pretty(&fixture).expect("json") + "\n",
            )
            .expect("write");
            continue;
        }
        if actual.len() != fixture.exchange.len() {
            failures.push(format!(
                "{}: {} steps, expected {}\nlast: {}",
                path.display(),
                actual.len(),
                fixture.exchange.len(),
                serde_json::to_string(&actual.last()).unwrap()
            ));
            continue;
        }
        for (i, (a, e)) in actual.iter().zip(&fixture.exchange).enumerate() {
            if normalize(&a.step) != normalize(&e.step) || a.input != e.input {
                failures.push(format!(
                    "{} step {i}:\n  got:  {}\n  want: {}",
                    path.display(),
                    serde_json::to_string(&normalize(&a.step)).unwrap(),
                    serde_json::to_string(&normalize(&e.step)).unwrap()
                ));
                break;
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n\n"));
}
