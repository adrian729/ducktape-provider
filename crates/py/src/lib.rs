//! `ducktape_provider._core`: the effects state machine over native Python objects.
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use ducktape_core::effects::{Clock, Resume, Start, Step};
use ducktape_core::types::{ErrorKind, ProviderError};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use serde_json::{Map, Number, Value};
use std::panic::{catch_unwind, AssertUnwindSafe};

fn to_py<'py>(py: Python<'py>, value: &Value) -> PyResult<Bound<'py, PyAny>> {
    Ok(match value {
        Value::Null => py.None().into_bound(py),
        Value::Bool(b) => PyBool::new(py, *b).to_owned().into_any(),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_pyobject(py)?.into_any()
            } else if let Some(u) = n.as_u64() {
                u.into_pyobject(py)?.into_any()
            } else {
                PyFloat::new(py, n.as_f64().unwrap_or(f64::NAN)).into_any()
            }
        }
        Value::String(s) => PyString::new(py, s).into_any(),
        Value::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(to_py(py, item)?)?;
            }
            list.into_any()
        }
        Value::Object(map) => {
            let dict = PyDict::new(py);
            for (k, v) in map {
                dict.set_item(k, to_py(py, v)?)?;
            }
            dict.into_any()
        }
    })
}

fn from_py(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.is_none() {
        return Ok(Value::Null);
    }
    if let Ok(b) = obj.cast::<PyBool>() {
        return Ok(Value::Bool(b.is_true()));
    }
    if let Ok(i) = obj.cast::<PyInt>() {
        if let Ok(v) = i.extract::<i64>() {
            return Ok(Value::from(v));
        }
        return Ok(Value::from(i.extract::<u64>()?));
    }
    if let Ok(f) = obj.cast::<PyFloat>() {
        let n = Number::from_f64(f.value())
            .ok_or_else(|| PyValueError::new_err("non-finite floats are not JSON"))?;
        return Ok(Value::Number(n));
    }
    if let Ok(s) = obj.cast::<PyString>() {
        return Ok(Value::String(s.to_cow()?.into_owned()));
    }
    if let Ok(b) = obj.cast::<PyBytes>() {
        return Ok(Value::String(STANDARD.encode(b.as_bytes())));
    }
    if let Ok(d) = obj.cast::<PyDict>() {
        let mut map = Map::with_capacity(d.len());
        for (k, v) in d.iter() {
            let key = k
                .cast::<PyString>()
                .map_err(|_| PyTypeError::new_err("dict keys must be strings"))?
                .to_cow()?
                .into_owned();
            map.insert(key, from_py(&v)?);
        }
        return Ok(Value::Object(map));
    }
    if let Ok(l) = obj.cast::<PyList>() {
        return l.iter().map(|item| from_py(&item)).collect();
    }
    if let Ok(t) = obj.cast::<PyTuple>() {
        return t.iter().map(|item| from_py(&item)).collect();
    }
    Err(PyTypeError::new_err(format!(
        "cannot convert {} to JSON",
        obj.get_type().name()?
    )))
}

fn parse<T: serde::de::DeserializeOwned>(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<T> {
    serde_json::from_value(from_py(obj)?)
        .map_err(|e| PyValueError::new_err(format!("invalid {what}: {e}")))
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

fn step_to_py<'py>(py: Python<'py>, step: &Step) -> PyResult<Bound<'py, PyAny>> {
    let value = serde_json::to_value(step)
        .map_err(|e| PyValueError::new_err(format!("step serialization: {e}")))?;
    to_py(py, &value)
}

/// One running operation of the core.
#[pyclass(module = "ducktape_provider._core")]
pub struct Operation {
    inner: ducktape_core::Operation,
}

#[pymethods]
impl Operation {
    /// Continues the operation with the host's answer to the pending effect.
    fn resume<'py>(
        &mut self,
        py: Python<'py>,
        resume: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let resume: Resume = parse(resume, "resume document")?;
        let step = guarded(|| self.inner.resume(resume));
        step_to_py(py, &step)
    }

    /// Delivers raw body bytes of the open response.
    fn feed<'py>(
        &mut self,
        py: Python<'py>,
        bytes: &[u8],
        clock: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let clock: Clock = parse(clock, "clock document")?;
        let step = guarded(|| self.inner.feed(bytes, clock));
        step_to_py(py, &step)
    }

    /// Stores a secret the host resolved; the core keeps its own copy.
    fn set_secret(&mut self, secret_ref: &str, bytes: &[u8]) {
        self.inner.set_secret(secret_ref, bytes.to_vec());
    }
}

/// Opens an operation and returns it with its first step.
#[pyfunction]
fn start<'py>(
    py: Python<'py>,
    start: &Bound<'py, PyAny>,
) -> PyResult<(Operation, Bound<'py, PyAny>)> {
    let start: Start = parse(start, "start document")?;
    let mut op = None;
    let step = guarded(|| {
        let (started, step) = ducktape_core::Operation::start(start);
        op = Some(started);
        step
    });
    let Some(inner) = op else {
        return Err(PyValueError::new_err(
            "the core could not open the operation",
        ));
    };
    Ok((Operation { inner }, step_to_py(py, &step)?))
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("SCHEMA_VERSION", ducktape_core::SCHEMA_VERSION)?;
    m.add_class::<Operation>()?;
    m.add_function(wrap_pyfunction!(start, m)?)?;
    Ok(())
}
