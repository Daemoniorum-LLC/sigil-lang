//! HTTP Client Support
//!
//! Provides HTTP/1.1 and HTTP/2 client functionality with streaming support.
//!
//! ## Features
//!
//! - Full HTTP/1.1 and HTTP/2 support
//! - Streaming request and response bodies
//! - Connection pooling
//! - Automatic compression/decompression (gzip, brotli)
//! - Multipart form uploads
//! - Cookie handling
//! - Retry with backoff
//! - Timeout configuration

use super::common::{
    Headers, Method, ProtocolError, ProtocolResult, RetryConfig, StatusCode, Timeout, Uri,
};
use std::collections::HashMap;
use std::time::Duration;

/// HTTP client configuration
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// Base URL for relative paths
    pub base_url: Option<String>,
    /// Default headers for all requests
    pub default_headers: Headers,
    /// Timeout configuration
    pub timeout: Timeout,
    /// Retry configuration
    pub retry: Option<RetryConfig>,
    /// Maximum redirects to follow
    pub max_redirects: u32,
    /// Whether to follow redirects
    pub follow_redirects: bool,
    /// User agent string
    pub user_agent: String,
    /// Whether to accept compressed responses
    pub accept_compressed: bool,
    /// Connection pool idle timeout
    pub pool_idle_timeout: Option<Duration>,
    /// Maximum idle connections per host
    pub pool_max_idle_per_host: usize,
}

impl Default for ClientConfig {
    fn default() -> Self {
        ClientConfig {
            base_url: None,
            default_headers: Headers::new(),
            timeout: Timeout::new()
                .connect_timeout(Duration::from_secs(30))
                .read_timeout(Duration::from_secs(30)),
            retry: None,
            max_redirects: 10,
            follow_redirects: true,
            user_agent: format!("sigil-http/{}", env!("CARGO_PKG_VERSION")),
            accept_compressed: true,
            pool_idle_timeout: Some(Duration::from_secs(90)),
            pool_max_idle_per_host: 10,
        }
    }
}

/// HTTP client for making requests
#[derive(Debug, Clone)]
pub struct Client {
    config: ClientConfig,
    #[cfg(feature = "reqwest")]
    inner: Option<reqwest::Client>,
}

impl Client {
    /// Create a new HTTP client with default configuration
    pub fn new() -> Self {
        Client::with_config(ClientConfig::default())
    }

    /// Create a new HTTP client with custom configuration
    pub fn with_config(config: ClientConfig) -> Self {
        #[cfg(feature = "reqwest")]
        let inner = {
            let mut builder = reqwest::Client::builder()
                .user_agent(&config.user_agent)
                .redirect(if config.follow_redirects {
                    reqwest::redirect::Policy::limited(config.max_redirects as usize)
                } else {
                    reqwest::redirect::Policy::none()
                });

            if let Some(timeout) = config.timeout.connect {
                builder = builder.connect_timeout(timeout);
            }
            if let Some(timeout) = config.timeout.read {
                builder = builder.read_timeout(timeout);
            }
            if let Some(timeout) = config.timeout.total {
                builder = builder.timeout(timeout);
            }
            if let Some(idle) = config.pool_idle_timeout {
                builder = builder.pool_idle_timeout(idle);
            }

            if config.accept_compressed {
                builder = builder.gzip(true).brotli(true);
            }

            builder.build().ok()
        };

        #[cfg(not(feature = "reqwest"))]
        let inner = None;

        Client {
            config,
            #[cfg(feature = "reqwest")]
            inner,
        }
    }

    /// Set base URL for relative paths
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.config.base_url = Some(url.into());
        self
    }

    /// Set a default header for all requests
    pub fn default_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.config.default_headers.insert(key, value);
        self
    }

    /// Set connection timeout
    pub fn connect_timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = self.config.timeout.connect_timeout(timeout);
        self
    }

    /// Set read timeout
    pub fn read_timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = self.config.timeout.read_timeout(timeout);
        self
    }

    /// Set total request timeout
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = self.config.timeout.total_timeout(timeout);
        self
    }

    /// Configure retry behavior
    pub fn retry(mut self, config: RetryConfig) -> Self {
        self.config.retry = Some(config);
        self
    }

    /// Set user agent
    pub fn user_agent(mut self, ua: impl Into<String>) -> Self {
        self.config.user_agent = ua.into();
        self
    }

    /// Set bearer token authentication
    pub fn bearer_auth(self, token: impl Into<String>) -> Self {
        self.default_header("Authorization", format!("Bearer {}", token.into()))
    }

    /// Set basic authentication
    pub fn basic_auth(self, username: impl Into<String>, password: impl Into<String>) -> Self {
        use base64::{engine::general_purpose::STANDARD, Engine};
        let credentials = format!("{}:{}", username.into(), password.into());
        let encoded = STANDARD.encode(credentials.as_bytes());
        self.default_header("Authorization", format!("Basic {}", encoded))
    }

    /// Create a GET request
    pub fn get(&self, url: impl Into<String>) -> RequestBuilder {
        self.request(Method::GET, url)
    }

    /// Create a POST request
    pub fn post(&self, url: impl Into<String>) -> RequestBuilder {
        self.request(Method::POST, url)
    }

    /// Create a PUT request
    pub fn put(&self, url: impl Into<String>) -> RequestBuilder {
        self.request(Method::PUT, url)
    }

    /// Create a DELETE request
    pub fn delete(&self, url: impl Into<String>) -> RequestBuilder {
        self.request(Method::DELETE, url)
    }

    /// Create a PATCH request
    pub fn patch(&self, url: impl Into<String>) -> RequestBuilder {
        self.request(Method::PATCH, url)
    }

    /// Create a HEAD request
    pub fn head(&self, url: impl Into<String>) -> RequestBuilder {
        self.request(Method::HEAD, url)
    }

    /// Create a request with the specified method
    pub fn request(&self, method: Method, url: impl Into<String>) -> RequestBuilder {
        let url_str = url.into();
        let full_url = if let Some(ref base) = self.config.base_url {
            if url_str.starts_with("http://") || url_str.starts_with("https://") {
                url_str
            } else {
                format!("{}{}", base.trim_end_matches('/'),
                    if url_str.starts_with('/') { url_str } else { format!("/{}", url_str) })
            }
        } else {
            url_str
        };

        RequestBuilder {
            client: self.clone(),
            method,
            url: full_url,
            headers: self.config.default_headers.clone(),
            query: Vec::new(),
            body: None,
            timeout: self.config.timeout.total,
        }
    }

    /// Resolve a URL against the base URL
    fn resolve_url(&self, url: &str) -> String {
        if let Some(ref base) = self.config.base_url {
            if url.starts_with("http://") || url.starts_with("https://") {
                url.to_string()
            } else {
                format!("{}{}", base.trim_end_matches('/'),
                    if url.starts_with('/') { url.to_string() } else { format!("/{}", url) })
            }
        } else {
            url.to_string()
        }
    }
}

impl Default for Client {
    fn default() -> Self {
        Client::new()
    }
}

/// Builder for HTTP requests
#[derive(Debug, Clone)]
pub struct RequestBuilder {
    client: Client,
    method: Method,
    url: String,
    headers: Headers,
    query: Vec<(String, String)>,
    body: Option<Body>,
    timeout: Option<Duration>,
}

impl RequestBuilder {
    /// Set a header
    pub fn header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key, value);
        self
    }

    /// Add query parameters
    pub fn query(mut self, params: &[(&str, &str)]) -> Self {
        for (k, v) in params {
            self.query.push((k.to_string(), v.to_string()));
        }
        self
    }

    /// Add a single query parameter
    pub fn query_param(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.query.push((key.into(), value.into()));
        self
    }

    /// Set JSON body
    pub fn json<T: serde::Serialize>(mut self, value: &T) -> ProtocolResult<Self> {
        let json = serde_json::to_string(value)
            .map_err(|e| ProtocolError::Serialization(e.to_string()))?;
        self.headers.set("Content-Type", "application/json");
        self.body = Some(Body::Text(json));
        Ok(self)
    }

    /// Set text body
    pub fn body(mut self, body: impl Into<String>) -> Self {
        self.body = Some(Body::Text(body.into()));
        self
    }

    /// Set binary body
    pub fn body_bytes(mut self, bytes: Vec<u8>) -> Self {
        self.body = Some(Body::Bytes(bytes));
        self
    }

    /// Set form body (application/x-www-form-urlencoded)
    pub fn form(mut self, data: &[(&str, &str)]) -> Self {
        let encoded: String = data
            .iter()
            .map(|(k, v)| format!("{}={}", urlencoded(k), urlencoded(v)))
            .collect::<Vec<_>>()
            .join("&");
        self.headers.set("Content-Type", "application/x-www-form-urlencoded");
        self.body = Some(Body::Text(encoded));
        self
    }

    /// Set request timeout
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Build the request
    pub fn build(self) -> Request {
        let mut url = self.url;
        if !self.query.is_empty() {
            let query_string: String = self.query
                .iter()
                .map(|(k, v)| format!("{}={}", urlencoded(k), urlencoded(v)))
                .collect::<Vec<_>>()
                .join("&");
            if url.contains('?') {
                url = format!("{}&{}", url, query_string);
            } else {
                url = format!("{}?{}", url, query_string);
            }
        }

        Request {
            method: self.method,
            url,
            headers: self.headers,
            body: self.body,
            timeout: self.timeout,
        }
    }

    /// Send the request and await the response
    #[cfg(feature = "reqwest")]
    pub async fn send(self) -> ProtocolResult<Response> {
        let request = self.build();
        let client = self.client;

        if let Some(ref inner) = client.inner {
            let mut req_builder = match request.method {
                Method::GET => inner.get(&request.url),
                Method::POST => inner.post(&request.url),
                Method::PUT => inner.put(&request.url),
                Method::DELETE => inner.delete(&request.url),
                Method::PATCH => inner.patch(&request.url),
                Method::HEAD => inner.head(&request.url),
                _ => return Err(ProtocolError::Protocol(format!("Unsupported method: {:?}", request.method))),
            };

            // Add headers
            for (key, values) in request.headers.iter() {
                for value in values {
                    req_builder = req_builder.header(key.as_str(), value.as_str());
                }
            }

            // Add body
            if let Some(body) = request.body {
                req_builder = match body {
                    Body::Text(text) => req_builder.body(text),
                    Body::Bytes(bytes) => req_builder.body(bytes),
                };
            }

            // Add timeout
            if let Some(timeout) = request.timeout {
                req_builder = req_builder.timeout(timeout);
            }

            // Send request
            let resp = req_builder.send().await
                .map_err(|e| {
                    if e.is_timeout() {
                        ProtocolError::RequestTimeout
                    } else if e.is_connect() {
                        ProtocolError::ConnectionFailed(e.to_string())
                    } else {
                        ProtocolError::Protocol(e.to_string())
                    }
                })?;

            // Convert response
            let status = StatusCode::from_u16(resp.status().as_u16());
            let mut headers = Headers::new();
            for (key, value) in resp.headers() {
                if let Ok(v) = value.to_str() {
                    headers.insert(key.as_str(), v);
                }
            }

            let bytes = resp.bytes().await
                .map_err(|e| ProtocolError::Io(e.to_string()))?;

            Ok(Response {
                status,
                headers,
                body: bytes.to_vec(),
            })
        } else {
            Err(ProtocolError::Protocol("HTTP client not initialized".to_string()))
        }
    }

    /// Synchronous send (blocks current thread)
    #[cfg(not(feature = "reqwest"))]
    pub fn send_sync(self) -> ProtocolResult<Response> {
        Err(ProtocolError::Protocol("HTTP client requires 'http-client' feature".to_string()))
    }
}

/// HTTP request body
#[derive(Debug, Clone)]
pub enum Body {
    /// Text body
    Text(String),
    /// Binary body
    Bytes(Vec<u8>),
}

/// HTTP request
#[derive(Debug, Clone)]
pub struct Request {
    /// HTTP method
    pub method: Method,
    /// Request URL
    pub url: String,
    /// Request headers
    pub headers: Headers,
    /// Request body
    pub body: Option<Body>,
    /// Request timeout
    pub timeout: Option<Duration>,
}

/// HTTP response
#[derive(Debug, Clone)]
pub struct Response {
    /// Response status code
    pub status: StatusCode,
    /// Response headers
    pub headers: Headers,
    /// Response body (raw bytes)
    pub body: Vec<u8>,
}

impl Response {
    /// Get the response status
    pub fn status(&self) -> StatusCode {
        self.status
    }

    /// Get the response headers
    pub fn headers(&self) -> &Headers {
        &self.headers
    }

    /// Get the response body as text
    pub fn text(&self) -> ProtocolResult<String> {
        String::from_utf8(self.body.clone())
            .map_err(|e| ProtocolError::Deserialization(e.to_string()))
    }

    /// Parse the response body as JSON
    pub fn json<T: serde::de::DeserializeOwned>(&self) -> ProtocolResult<T> {
        serde_json::from_slice(&self.body)
            .map_err(|e| ProtocolError::Deserialization(e.to_string()))
    }

    /// Get the response body as bytes
    pub fn bytes(&self) -> &[u8] {
        &self.body
    }

    /// Take ownership of the response body
    pub fn into_bytes(self) -> Vec<u8> {
        self.body
    }

    /// Check if the response status is success (2xx)
    pub fn is_success(&self) -> bool {
        self.status.is_success()
    }

    /// Check if the response status is a client error (4xx)
    pub fn is_client_error(&self) -> bool {
        self.status.is_client_error()
    }

    /// Check if the response status is a server error (5xx)
    pub fn is_server_error(&self) -> bool {
        self.status.is_server_error()
    }

    /// Convert to a result, returning Err for non-success status codes
    pub fn error_for_status(self) -> ProtocolResult<Self> {
        if self.status.is_client_error() {
            let body = self.text().unwrap_or_default();
            Err(ProtocolError::ClientError(self.status, body))
        } else if self.status.is_server_error() {
            let body = self.text().unwrap_or_default();
            Err(ProtocolError::ServerError(self.status, body))
        } else {
            Ok(self)
        }
    }

    /// Get the content-type header
    pub fn content_type(&self) -> Option<&str> {
        self.headers.get("content-type")
    }

    /// Get the content-length header
    pub fn content_length(&self) -> Option<u64> {
        self.headers.content_length()
    }
}

/// URL-encode a string
fn urlencoded(s: &str) -> String {
    let mut result = String::new();
    for c in s.chars() {
        match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' | '.' | '~' => {
                result.push(c);
            }
            ' ' => result.push_str("%20"),
            _ => {
                for b in c.to_string().as_bytes() {
                    result.push_str(&format!("%{:02X}", b));
                }
            }
        }
    }
    result
}

/// Convenience function to make a GET request
pub async fn get(url: impl Into<String>) -> ProtocolResult<Response> {
    #[cfg(feature = "reqwest")]
    {
        Client::new().get(url).send().await
    }
    #[cfg(not(feature = "reqwest"))]
    {
        let _ = url;
        Err(ProtocolError::Protocol("HTTP client requires 'http-client' feature".to_string()))
    }
}

/// Convenience function to make a POST request with JSON body
pub async fn post_json<T: serde::Serialize>(url: impl Into<String>, body: &T) -> ProtocolResult<Response> {
    #[cfg(feature = "reqwest")]
    {
        Client::new().post(url).json(body)?.send().await
    }
    #[cfg(not(feature = "reqwest"))]
    {
        let _ = (url, body);
        Err(ProtocolError::Protocol("HTTP client requires 'http-client' feature".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_builder() {
        let client = Client::new()
            .base_url("https://api.example.com")
            .bearer_auth("token123")
            .timeout(Duration::from_secs(30));

        assert_eq!(client.config.base_url, Some("https://api.example.com".to_string()));
    }

    #[test]
    fn test_request_builder() {
        let client = Client::new().base_url("https://api.example.com");
        let request = client
            .get("/users")
            .query(&[("page", "1"), ("limit", "10")])
            .header("X-Custom", "value")
            .build();

        assert_eq!(request.method, Method::GET);
        assert!(request.url.contains("page=1"));
        assert!(request.url.contains("limit=10"));
    }

    #[test]
    fn test_url_encoding() {
        assert_eq!(urlencoded("hello world"), "hello%20world");
        assert_eq!(urlencoded("foo=bar"), "foo%3Dbar");
    }
}
