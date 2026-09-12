import unittest
import urllib.error

from http_test_utils import http_error

from ducktape_provider import errors


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


if __name__ == "__main__":
    unittest.main()
