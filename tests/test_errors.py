import http.client
import io
import unittest
import urllib.error
from email.message import Message

from http_test_utils import http_error

from ducktape_provider import errors


class _FailingBody(io.BytesIO):
    def __init__(self, failure: BaseException):
        super().__init__()
        self._failure = failure

    def read(self, size: int | None = -1, /) -> bytes:
        raise self._failure


class RaiseForHttpErrorTests(unittest.TestCase):
    def test_403_raises_auth_error(self):
        e = http_error("http://x", 403, b"forbidden")
        with self.assertRaises(errors.AuthError) as ctx:
            errors.raise_for_http_error("test", e)
        self.assertEqual(ctx.exception.status, 403)
        self.assertEqual(ctx.exception.body, "forbidden")

    def test_400_with_context_overflow_marker_raises_context_overflow_error(self):
        e = http_error("http://x", 400, b'{"error": "context_length_exceeded"}')
        with self.assertRaises(errors.ContextOverflowError) as ctx:
            errors.raise_for_http_error("test", e)
        self.assertEqual(ctx.exception.status, 400)

    def test_400_with_unrelated_body_raises_plain_api_error(self):
        e = http_error("http://x", 400, b"bad request, missing field")
        with self.assertRaises(errors.APIError) as ctx:
            errors.raise_for_http_error("test", e)
        self.assertIs(type(ctx.exception), errors.APIError)
        self.assertEqual(ctx.exception.status, 400)

    def test_413_raises_context_overflow_error_regardless_of_body(self):
        for body in (b"", b"unrelated gateway error"):
            with self.subTest(body=body):
                e = http_error("http://x", 413, body)
                with self.assertRaises(errors.ContextOverflowError) as ctx:
                    errors.raise_for_http_error("test", e)
                self.assertEqual(ctx.exception.status, 413)

    def test_404_raises_plain_api_error(self):
        e = http_error("http://x", 404, b"not found")
        with self.assertRaises(errors.APIError) as ctx:
            errors.raise_for_http_error("test", e)
        self.assertIs(type(ctx.exception), errors.APIError)
        self.assertEqual(ctx.exception.status, 404)

    def test_429_with_retry_after_header_sets_retry_after(self):
        e = http_error("http://x", 429, b"slow down", headers={"Retry-After": "12.5"})
        with self.assertRaises(errors.RateLimitError) as ctx:
            errors.raise_for_http_error("test", e)
        self.assertEqual(ctx.exception.retry_after, 12.5)

    def test_429_without_retry_after_header_leaves_retry_after_none(self):
        e = http_error("http://x", 429, b"slow down")
        with self.assertRaises(errors.RateLimitError) as ctx:
            errors.raise_for_http_error("test", e)
        self.assertIsNone(ctx.exception.retry_after)

    def test_429_with_unparseable_retry_after_leaves_retry_after_none(self):
        e = http_error(
            "http://x", 429, b"slow down", headers={"Retry-After": "not-a-number"}
        )
        with self.assertRaises(errors.RateLimitError) as ctx:
            errors.raise_for_http_error("test", e)
        self.assertIsNone(ctx.exception.retry_after)

    def test_429_with_non_finite_or_negative_retry_after_leaves_none(self):
        for value in ("nan", "inf", "-inf", "-1"):
            with self.subTest(value=value):
                e = http_error(
                    "http://x", 429, b"slow down", headers={"Retry-After": value}
                )
                with self.assertRaises(errors.RateLimitError) as ctx:
                    errors.raise_for_http_error("test", e)
                self.assertIsNone(ctx.exception.retry_after)

    def test_error_body_is_closed_after_reading(self):
        e = http_error("http://x", 500, b"boom")
        with self.assertRaises(errors.ServerError):
            errors.raise_for_http_error("test", e)
        self.assertTrue(e.fp.closed)

    def test_body_read_failure_still_classifies_by_status_and_closes(self):
        for failure in (
            TimeoutError(),
            http.client.IncompleteRead(b"part"),
            ConnectionResetError(),
        ):
            headers = Message()
            headers["Retry-After"] = "3"
            e = urllib.error.HTTPError(
                "http://x", 429, "error", headers, _FailingBody(failure)
            )
            with (
                self.subTest(failure=failure),
                self.assertRaises(errors.RateLimitError) as ctx,
            ):
                errors.raise_for_http_error("test", e)
            self.assertEqual(ctx.exception.body, "")
            self.assertEqual(ctx.exception.retry_after, 3.0)
            self.assertTrue(e.fp.closed)

    def test_oversized_body_is_read_only_up_to_the_cap_and_quoted_truncated(self):
        body = b'{"error": "prompt is too long"}' + b"x" * (1024 * 1024)
        e = http_error("http://x", 400, body)
        with self.assertRaises(errors.ContextOverflowError) as ctx:
            errors.raise_for_http_error("test", e)
        self.assertEqual(len(ctx.exception.body), errors._MAX_ERROR_BODY_BYTES)
        message = str(ctx.exception)
        self.assertLess(len(message), errors._MAX_MESSAGE_DETAIL_CHARS + 100)
        self.assertTrue(message.endswith("... [truncated]"))

    def test_short_body_is_quoted_whole(self):
        e = http_error("http://x", 404, b"no such model")
        with self.assertRaises(errors.APIError) as ctx:
            errors.raise_for_http_error("test", e)
        self.assertEqual(str(ctx.exception), "test chat failed: 404 no such model")

    def test_500_raises_server_error(self):
        e = http_error("http://x", 500, b"boom")
        with self.assertRaises(errors.ServerError) as ctx:
            errors.raise_for_http_error("test", e)
        self.assertEqual(ctx.exception.status, 500)


class RaiseForConnectionErrorTests(unittest.TestCase):
    def test_bare_timeout_error_raises_request_timeout_error(self):
        with self.assertRaises(errors.RequestTimeoutError):
            errors.raise_for_connection_error("test", TimeoutError())

    def test_url_error_with_timeout_reason_raises_request_timeout_error(self):
        e = urllib.error.URLError(TimeoutError())
        with self.assertRaises(errors.RequestTimeoutError):
            errors.raise_for_connection_error("test", e)

    def test_url_error_with_non_timeout_reason_raises_plain_api_error(self):
        e = urllib.error.URLError(OSError("connection refused"))
        with self.assertRaises(errors.APIError) as ctx:
            errors.raise_for_connection_error("test", e)
        self.assertIs(type(ctx.exception), errors.APIError)
        self.assertIn("connection refused", str(ctx.exception))


class RaiseForVendorErrorTests(unittest.TestCase):
    def test_mapped_status_picks_same_class_as_http_path(self):
        cases = [
            (401, errors.AuthError),
            (429, errors.RateLimitError),
            (529, errors.ServerError),
            (404, errors.APIError),
            (413, errors.ContextOverflowError),
        ]
        for status, expected in cases:
            with self.subTest(status=status), self.assertRaises(errors.APIError) as ctx:
                errors.raise_for_vendor_error("test", "boom", status=status)
            self.assertIs(type(ctx.exception), expected)
            self.assertEqual(ctx.exception.status, status)

    def test_413_raises_context_overflow_regardless_of_body(self):
        with self.assertRaises(errors.ContextOverflowError) as ctx:
            errors.raise_for_vendor_error("test", "boom", status=413, body="unrelated")
        self.assertEqual(ctx.exception.status, 413)

    def test_unknown_status_with_overflow_marker_raises_context_overflow(self):
        with self.assertRaises(errors.ContextOverflowError):
            errors.raise_for_vendor_error("test", "prompt is too long")

    def test_long_vendor_message_is_truncated_in_message_but_kept_in_body(self):
        message = "boom " * 10_000
        with self.assertRaises(errors.APIError) as ctx:
            errors.raise_for_vendor_error("test", message)
        self.assertTrue(str(ctx.exception).endswith("... [truncated]"))
        self.assertEqual(ctx.exception.body, message)

    def test_unknown_status_without_marker_raises_plain_api_error(self):
        with self.assertRaises(errors.APIError) as ctx:
            errors.raise_for_vendor_error("test", "boom", body='{"x": 1}')
        self.assertIs(type(ctx.exception), errors.APIError)
        self.assertEqual(ctx.exception.body, '{"x": 1}')

    def test_oversized_body_is_capped_like_an_http_error_body(self):
        body = "x" * (16 * 1024 * 1024)
        with self.assertRaises(errors.APIError) as ctx:
            errors.raise_for_vendor_error("test", "boom", body=body)
        self.assertEqual(len(ctx.exception.body), errors._MAX_ERROR_BODY_BYTES)

    def test_oversized_message_used_as_body_is_also_capped(self):
        message = "x" * (16 * 1024 * 1024)
        with self.assertRaises(errors.APIError) as ctx:
            errors.raise_for_vendor_error("test", message)
        self.assertEqual(len(ctx.exception.body), errors._MAX_ERROR_BODY_BYTES)


class RaiseForReadErrorTests(unittest.TestCase):
    def test_malformed_payload_raises_malformed_response_error(self):
        cause = ValueError("Expecting value")
        with self.assertRaises(errors.MalformedResponseError) as ctx:
            errors.raise_for_malformed_response("test", cause)
        self.assertIsInstance(ctx.exception, errors.APIError)
        self.assertIsNone(ctx.exception.status)
        self.assertIs(ctx.exception.__cause__, cause)
        self.assertIn("malformed", str(ctx.exception))

    def test_truncated_stream_is_a_plain_api_error_not_malformed(self):
        with self.assertRaises(errors.APIError) as ctx:
            errors.raise_for_truncated_stream("test", "message_stop")
        self.assertIs(type(ctx.exception), errors.APIError)

    def test_malformed_lookup_error_names_the_error_type(self):
        with self.assertRaises(errors.APIError) as ctx:
            errors.raise_for_malformed_response("test", KeyError("index"))
        self.assertIn("KeyError: 'index'", str(ctx.exception))

    def test_incomplete_read_raises_api_error(self):
        with self.assertRaises(errors.APIError) as ctx:
            errors.raise_for_connection_error("test", http.client.IncompleteRead(b""))
        self.assertIn("connection closed", str(ctx.exception))

    def test_bare_connection_reset_names_the_error(self):
        with self.assertRaises(errors.APIError) as ctx:
            errors.raise_for_connection_error("test", ConnectionResetError())
        self.assertIn("ConnectionResetError", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
