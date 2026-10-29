//! Server-sent events parser over arbitrary byte chunks.

/// One complete event.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SseEvent {
    pub event: Option<String>,
    pub data: String,
}

/// Incremental SSE parser.
#[derive(Debug, Default)]
pub struct SseParser {
    buffer: Vec<u8>,
    event: Option<String>,
    data: Vec<String>,
}

impl SseParser {
    /// Feeds bytes and returns the events completed by them.
    pub fn feed(&mut self, bytes: &[u8]) -> Vec<SseEvent> {
        self.buffer.extend_from_slice(bytes);
        let mut out = Vec::new();
        while let Some(pos) = self.buffer.iter().position(|b| *b == b'\n') {
            let mut line: Vec<u8> = self.buffer.drain(..=pos).collect();
            line.pop();
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            if let Some(ev) = self.line(&line) {
                out.push(ev);
            }
        }
        out
    }

    /// Flushes a trailing event without a final blank line.
    pub fn finish(&mut self) -> Option<SseEvent> {
        if !self.buffer.is_empty() {
            let line = std::mem::take(&mut self.buffer);
            if let Some(ev) = self.line(&line) {
                return Some(ev);
            }
        }
        self.dispatch()
    }

    fn line(&mut self, line: &[u8]) -> Option<SseEvent> {
        if line.is_empty() {
            return self.dispatch();
        }
        let text = String::from_utf8_lossy(line);
        if text.starts_with(':') {
            return None;
        }
        let (field, value) = match text.split_once(':') {
            Some((f, v)) => (f, v.strip_prefix(' ').unwrap_or(v)),
            None => (text.as_ref(), ""),
        };
        match field {
            "event" => self.event = Some(value.to_owned()),
            "data" => self.data.push(value.to_owned()),
            _ => {}
        }
        None
    }

    fn dispatch(&mut self) -> Option<SseEvent> {
        if self.data.is_empty() && self.event.is_none() {
            return None;
        }
        let ev = SseEvent {
            event: self.event.take(),
            data: self.data.join("\n"),
        };
        self.data.clear();
        Some(ev)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_events_across_chunks() {
        let mut p = SseParser::default();
        assert!(p.feed(b"event: ping\ndata: {\"a\":").is_empty());
        let evs = p.feed(b" 1}\n\r\ndata: x\n\n");
        assert_eq!(evs.len(), 2);
        assert_eq!(evs[0].event.as_deref(), Some("ping"));
        assert_eq!(evs[0].data, "{\"a\": 1}");
        assert_eq!(evs[1].event, None);
        assert_eq!(evs[1].data, "x");
    }

    #[test]
    fn ignores_comments_and_joins_multiline_data() {
        let mut p = SseParser::default();
        let evs = p.feed(b": keepalive\ndata: a\ndata: b\n\n");
        assert_eq!(
            evs,
            vec![SseEvent {
                event: None,
                data: "a\nb".into()
            }]
        );
        assert_eq!(p.finish(), None);
    }
}
