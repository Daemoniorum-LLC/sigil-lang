//! Common Protocol Types
//!
//! Shared abstractions used across all protocol implementations.

use std::collections::HashMap;
use std::fmt;
use std::time::Duration;

/// A parsed URI with all components
#[derive(Debug, Clone, PartialEq)]
pub struct Uri {
    /// The scheme (e.g., "https", "grpc", "ws")
    pub scheme: String,
    /// The host (e.g., "example.com")
    pub host: String,
    /// The port number (optional)
    pub port: Option<u16>,
    /// The path (e.g., "/api/v1/users")
    pub path: String,
    /// The query string (optional, without leading ?)
    pub query: Option<String>,
    /// The fragment (optional, without leading #)
    pub fragment: Option<String>,
    /// User info for authentication (optional, "user:pass")
    pub userinfo: Option<String>,
}

impl Uri {
    /// Parse a URI string into components
    pub fn parse(s: &str) -> Result<Self, ProtocolError> {
        #[cfg(feature = "url")]
        {
            use url::Url;
            let url = Url::parse(s).map_err(|e| ProtocolError::InvalidUri(e.to_string()))?;

            Ok(Uri {
                scheme: url.scheme().to_string(),
                host: url.host_str().unwrap_or("").to_string(),
                port: url.port(),
                path: url.path().to_string(),
                query: url.query().map(|s| s.to_string()),
                fragment: url.fragment().map(|s| s.to_string()),
                userinfo: if url.username().is_empty() {
                    None
                } else {
                    Some(format!(
                        "{}:{}",
                        url.username(),
                        url.password().unwrap_or("")
                    ))
                },
            })
        }

        #[cfg(not(feature = "url"))]
        {
            // Basic fallback parser
            let mut uri = Uri {
                scheme: String::new(),
                host: String::new(),
                port: None,
                path: String::from("/"),
                query: None,
                fragment: None,
                userinfo: None,
            };

            let s = s.trim();

            // Parse scheme
            if let Some(pos) = s.find("://") {
                uri.scheme = s[..pos].to_string();
                let rest = &s[pos + 3..];

                // Parse host and port
                let (authority, path_and_rest) = if let Some(pos) = rest.find('/') {
                    (&rest[..pos], &rest[pos..])
                } else {
                    (rest, "/")
                };

                // Parse userinfo
                let host_port = if let Some(pos) = authority.find('@') {
                    uri.userinfo = Some(authority[..pos].to_string());
                    &authority[pos + 1..]
                } else {
                    authority
                };

                // Parse host and port
                if let Some(pos) = host_port.rfind(':') {
                    uri.host = host_port[..pos].to_string();
                    if let Ok(port) = host_port[pos + 1..].parse() {
                        uri.port = Some(port);
                    }
                } else {
                    uri.host = host_port.to_string();
                }

                // Parse path, query, fragment
                let (path_query, fragment) = if let Some(pos) = path_and_rest.find('#') {
                    uri.fragment = Some(path_and_rest[pos + 1..].to_string());
                    (&path_and_rest[..pos], Some(&path_and_rest[pos + 1..]))
                } else {
                    (path_and_rest, None)
                };

                if let Some(pos) = path_query.find('?') {
                    uri.path = path_query[..pos].to_string();
                    uri.query = Some(path_query[pos + 1..].to_string());
                } else {
                    uri.path = path_query.to_string();
                }
            } else {
                return Err(ProtocolError::InvalidUri("Missing scheme".to_string()));
            }

            Ok(uri)
        }
    }

    /// Reconstruct the full URI string
    pub fn to_string(&self) -> String {
        let mut s = format!("{}://", self.scheme);

        if let Some(ref userinfo) = self.userinfo {
            s.push_str(userinfo);
            s.push('@');
        }

        s.push_str(&self.host);

        if let Some(port) = self.port {
            s.push(':');
            s.push_str(&port.to_string());
        }

        s.push_str(&self.path);

        if let Some(ref query) = self.query {
            s.push('?');
            s.push_str(query);
        }

        if let Some(ref fragment) = self.fragment {
            s.push('#');
            s.push_str(fragment);
        }

        s
    }

    /// Get the authority portion (host:port)
    pub fn authority(&self) -> String {
        if let Some(port) = self.port {
            format!("{}:{}", self.host, port)
        } else {
            self.host.clone()
        }
    }
}

impl fmt::Display for Uri {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

/// HTTP-style headers collection
#[derive(Debug, Clone, Default)]
pub struct Headers {
    inner: HashMap<String, Vec<String>>,
}

impl Headers {
    /// Create empty headers
    pub fn new() -> Self {
        Headers {
            inner: HashMap::new(),
        }
    }

    /// Insert a header value (appends if key exists)
    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<String>) -> &mut Self {
        let key = key.into().to_lowercase();
        let value = value.into();
        self.inner.entry(key).or_insert_with(Vec::new).push(value);
        self
    }

    /// Set a header value (replaces existing)
    pub fn set(&mut self, key: impl Into<String>, value: impl Into<String>) -> &mut Self {
        let key = key.into().to_lowercase();
        let value = value.into();
        self.inner.insert(key, vec![value]);
        self
    }

    /// Get the first value for a header
    pub fn get(&self, key: &str) -> Option<&str> {
        self.inner
            .get(&key.to_lowercase())
            .and_then(|v| v.first().map(|s| s.as_str()))
    }

    /// Get all values for a header
    pub fn get_all(&self, key: &str) -> Option<&[String]> {
        self.inner.get(&key.to_lowercase()).map(|v| v.as_slice())
    }

    /// Remove a header
    pub fn remove(&mut self, key: &str) -> Option<Vec<String>> {
        self.inner.remove(&key.to_lowercase())
    }

    /// Check if header exists
    pub fn contains(&self, key: &str) -> bool {
        self.inner.contains_key(&key.to_lowercase())
    }

    /// Get content-type header
    pub fn content_type(&self) -> Option<&str> {
        self.get("content-type")
    }

    /// Get content-length header
    pub fn content_length(&self) -> Option<u64> {
        self.get("content-length").and_then(|v| v.parse().ok())
    }

    /// Iterate over all headers
    pub fn iter(&self) -> impl Iterator<Item = (&String, &Vec<String>)> {
        self.inner.iter()
    }

    /// Number of header keys
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl IntoIterator for Headers {
    type Item = (String, Vec<String>);
    type IntoIter = std::collections::hash_map::IntoIter<String, Vec<String>>;

    fn into_iter(self) -> Self::IntoIter {
        self.inner.into_iter()
    }
}

impl<K: Into<String>, V: Into<String>> FromIterator<(K, V)> for Headers {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> Self {
        let mut headers = Headers::new();
        for (k, v) in iter {
            headers.insert(k, v);
        }
        headers
    }
}

/// HTTP status codes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StatusCode(u16);

impl StatusCode {
    // Informational
    pub const CONTINUE: StatusCode = StatusCode(100);
    pub const SWITCHING_PROTOCOLS: StatusCode = StatusCode(101);

    // Success
    pub const OK: StatusCode = StatusCode(200);
    pub const CREATED: StatusCode = StatusCode(201);
    pub const ACCEPTED: StatusCode = StatusCode(202);
    pub const NO_CONTENT: StatusCode = StatusCode(204);

    // Redirection
    pub const MOVED_PERMANENTLY: StatusCode = StatusCode(301);
    pub const FOUND: StatusCode = StatusCode(302);
    pub const SEE_OTHER: StatusCode = StatusCode(303);
    pub const NOT_MODIFIED: StatusCode = StatusCode(304);
    pub const TEMPORARY_REDIRECT: StatusCode = StatusCode(307);
    pub const PERMANENT_REDIRECT: StatusCode = StatusCode(308);

    // Client Error
    pub const BAD_REQUEST: StatusCode = StatusCode(400);
    pub const UNAUTHORIZED: StatusCode = StatusCode(401);
    pub const FORBIDDEN: StatusCode = StatusCode(403);
    pub const NOT_FOUND: StatusCode = StatusCode(404);
    pub const METHOD_NOT_ALLOWED: StatusCode = StatusCode(405);
    pub const CONFLICT: StatusCode = StatusCode(409);
    pub const GONE: StatusCode = StatusCode(410);
    pub const UNPROCESSABLE_ENTITY: StatusCode = StatusCode(422);
    pub const TOO_MANY_REQUESTS: StatusCode = StatusCode(429);

    // Server Error
    pub const INTERNAL_SERVER_ERROR: StatusCode = StatusCode(500);
    pub const NOT_IMPLEMENTED: StatusCode = StatusCode(501);
    pub const BAD_GATEWAY: StatusCode = StatusCode(502);
    pub const SERVICE_UNAVAILABLE: StatusCode = StatusCode(503);
    pub const GATEWAY_TIMEOUT: StatusCode = StatusCode(504);

    /// Create a status code from a number
    pub fn from_u16(code: u16) -> Self {
        StatusCode(code)
    }

    /// Get the numeric code
    pub fn as_u16(&self) -> u16 {
        self.0
    }

    /// Check if this is an informational status (1xx)
    pub fn is_informational(&self) -> bool {
        self.0 >= 100 && self.0 < 200
    }

    /// Check if this is a success status (2xx)
    pub fn is_success(&self) -> bool {
        self.0 >= 200 && self.0 < 300
    }

    /// Check if this is a redirection status (3xx)
    pub fn is_redirection(&self) -> bool {
        self.0 >= 300 && self.0 < 400
    }

    /// Check if this is a client error status (4xx)
    pub fn is_client_error(&self) -> bool {
        self.0 >= 400 && self.0 < 500
    }

    /// Check if this is a server error status (5xx)
    pub fn is_server_error(&self) -> bool {
        self.0 >= 500 && self.0 < 600
    }

    /// Get the reason phrase for this status code
    pub fn reason_phrase(&self) -> &'static str {
        match self.0 {
            100 => "Continue",
            101 => "Switching Protocols",
            200 => "OK",
            201 => "Created",
            202 => "Accepted",
            204 => "No Content",
            301 => "Moved Permanently",
            302 => "Found",
            303 => "See Other",
            304 => "Not Modified",
            307 => "Temporary Redirect",
            308 => "Permanent Redirect",
            400 => "Bad Request",
            401 => "Unauthorized",
            403 => "Forbidden",
            404 => "Not Found",
            405 => "Method Not Allowed",
            409 => "Conflict",
            410 => "Gone",
            422 => "Unprocessable Entity",
            429 => "Too Many Requests",
            500 => "Internal Server Error",
            501 => "Not Implemented",
            502 => "Bad Gateway",
            503 => "Service Unavailable",
            504 => "Gateway Timeout",
            _ => "Unknown",
        }
    }
}

impl fmt::Display for StatusCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.0, self.reason_phrase())
    }
}

/// Timeout configuration for protocol operations
#[derive(Debug, Clone)]
pub struct Timeout {
    /// Connection timeout
    pub connect: Option<Duration>,
    /// Read timeout
    pub read: Option<Duration>,
    /// Write timeout
    pub write: Option<Duration>,
    /// Total operation timeout
    pub total: Option<Duration>,
}

impl Timeout {
    /// Create a new timeout configuration
    pub fn new() -> Self {
        Timeout {
            connect: None,
            read: None,
            write: None,
            total: None,
        }
    }

    /// Set all timeouts to the same duration
    pub fn all(duration: Duration) -> Self {
        Timeout {
            connect: Some(duration),
            read: Some(duration),
            write: Some(duration),
            total: Some(duration),
        }
    }

    /// Set connection timeout
    pub fn connect_timeout(mut self, duration: Duration) -> Self {
        self.connect = Some(duration);
        self
    }

    /// Set read timeout
    pub fn read_timeout(mut self, duration: Duration) -> Self {
        self.read = Some(duration);
        self
    }

    /// Set write timeout
    pub fn write_timeout(mut self, duration: Duration) -> Self {
        self.write = Some(duration);
        self
    }

    /// Set total timeout
    pub fn total_timeout(mut self, duration: Duration) -> Self {
        self.total = Some(duration);
        self
    }
}

impl Default for Timeout {
    fn default() -> Self {
        Timeout::new()
    }
}

/// Backoff strategy for retries
#[derive(Debug, Clone)]
pub enum BackoffStrategy {
    /// Fixed delay between retries
    Fixed(Duration),
    /// Linear increase in delay
    Linear {
        initial: Duration,
        increment: Duration,
        max: Option<Duration>,
    },
    /// Exponential increase in delay
    Exponential {
        initial: Duration,
        factor: f64,
        max: Option<Duration>,
    },
}

impl BackoffStrategy {
    /// Calculate the delay for a given attempt number (0-indexed)
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        match self {
            BackoffStrategy::Fixed(d) => *d,
            BackoffStrategy::Linear {
                initial,
                increment,
                max,
            } => {
                let delay = *initial + (*increment * attempt);
                max.map(|m| delay.min(m)).unwrap_or(delay)
            }
            BackoffStrategy::Exponential {
                initial,
                factor,
                max,
            } => {
                let multiplier = factor.powi(attempt as i32);
                let delay = initial.mul_f64(multiplier);
                max.map(|m| delay.min(m)).unwrap_or(delay)
            }
        }
    }
}

/// Retry configuration
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of attempts (including initial)
    pub max_attempts: u32,
    /// Backoff strategy
    pub backoff: BackoffStrategy,
    /// Status codes to retry on
    pub retry_on_status: Vec<StatusCode>,
    /// Whether to retry on connection errors
    pub retry_on_connection_error: bool,
    /// Whether to retry on timeout
    pub retry_on_timeout: bool,
}

impl RetryConfig {
    /// Create a new retry configuration
    pub fn new(max_attempts: u32) -> Self {
        RetryConfig {
            max_attempts,
            backoff: BackoffStrategy::Exponential {
                initial: Duration::from_millis(100),
                factor: 2.0,
                max: Some(Duration::from_secs(30)),
            },
            retry_on_status: vec![
                StatusCode::SERVICE_UNAVAILABLE,
                StatusCode::GATEWAY_TIMEOUT,
                StatusCode::TOO_MANY_REQUESTS,
            ],
            retry_on_connection_error: true,
            retry_on_timeout: true,
        }
    }

    /// Set the backoff strategy
    pub fn backoff(mut self, strategy: BackoffStrategy) -> Self {
        self.backoff = strategy;
        self
    }

    /// Set status codes to retry on
    pub fn retry_on(mut self, codes: Vec<StatusCode>) -> Self {
        self.retry_on_status = codes;
        self
    }

    /// Check if a status code should be retried
    pub fn should_retry_status(&self, status: StatusCode) -> bool {
        self.retry_on_status.contains(&status)
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        RetryConfig::new(3)
    }
}

/// Protocol error types
#[derive(Debug, Clone)]
pub enum ProtocolError {
    /// Invalid URI format
    InvalidUri(String),
    /// Connection failed
    ConnectionFailed(String),
    /// Connection timeout
    ConnectionTimeout,
    /// Read timeout
    ReadTimeout,
    /// Write timeout
    WriteTimeout,
    /// Request timeout
    RequestTimeout,
    /// TLS/SSL error
    TlsError(String),
    /// Protocol-specific error
    Protocol(String),
    /// Serialization error
    Serialization(String),
    /// Deserialization error
    Deserialization(String),
    /// Authentication error
    Authentication(String),
    /// Authorization error
    Authorization(String),
    /// Rate limited
    RateLimited { retry_after: Option<Duration> },
    /// Resource not found
    NotFound(String),
    /// Server error
    ServerError(StatusCode, String),
    /// Client error
    ClientError(StatusCode, String),
    /// IO error
    Io(String),
    /// Channel closed
    ChannelClosed,
    /// Operation cancelled
    Cancelled,
    /// Other error
    Other(String),
}

impl fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProtocolError::InvalidUri(msg) => write!(f, "Invalid URI: {}", msg),
            ProtocolError::ConnectionFailed(msg) => write!(f, "Connection failed: {}", msg),
            ProtocolError::ConnectionTimeout => write!(f, "Connection timeout"),
            ProtocolError::ReadTimeout => write!(f, "Read timeout"),
            ProtocolError::WriteTimeout => write!(f, "Write timeout"),
            ProtocolError::RequestTimeout => write!(f, "Request timeout"),
            ProtocolError::TlsError(msg) => write!(f, "TLS error: {}", msg),
            ProtocolError::Protocol(msg) => write!(f, "Protocol error: {}", msg),
            ProtocolError::Serialization(msg) => write!(f, "Serialization error: {}", msg),
            ProtocolError::Deserialization(msg) => write!(f, "Deserialization error: {}", msg),
            ProtocolError::Authentication(msg) => write!(f, "Authentication error: {}", msg),
            ProtocolError::Authorization(msg) => write!(f, "Authorization error: {}", msg),
            ProtocolError::RateLimited { retry_after } => {
                if let Some(d) = retry_after {
                    write!(f, "Rate limited, retry after {:?}", d)
                } else {
                    write!(f, "Rate limited")
                }
            }
            ProtocolError::NotFound(msg) => write!(f, "Not found: {}", msg),
            ProtocolError::ServerError(code, msg) => write!(f, "Server error ({}): {}", code, msg),
            ProtocolError::ClientError(code, msg) => write!(f, "Client error ({}): {}", code, msg),
            ProtocolError::Io(msg) => write!(f, "IO error: {}", msg),
            ProtocolError::ChannelClosed => write!(f, "Channel closed"),
            ProtocolError::Cancelled => write!(f, "Operation cancelled"),
            ProtocolError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for ProtocolError {}

/// Result type for protocol operations
pub type ProtocolResult<T> = Result<T, ProtocolError>;

/// HTTP Methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Method {
    GET,
    POST,
    PUT,
    DELETE,
    PATCH,
    HEAD,
    OPTIONS,
    CONNECT,
    TRACE,
}

impl Method {
    /// Get the method name as a string
    pub fn as_str(&self) -> &'static str {
        match self {
            Method::GET => "GET",
            Method::POST => "POST",
            Method::PUT => "PUT",
            Method::DELETE => "DELETE",
            Method::PATCH => "PATCH",
            Method::HEAD => "HEAD",
            Method::OPTIONS => "OPTIONS",
            Method::CONNECT => "CONNECT",
            Method::TRACE => "TRACE",
        }
    }

    /// Check if the method is idempotent
    pub fn is_idempotent(&self) -> bool {
        matches!(
            self,
            Method::GET
                | Method::HEAD
                | Method::PUT
                | Method::DELETE
                | Method::OPTIONS
                | Method::TRACE
        )
    }

    /// Check if the method is safe (no side effects)
    pub fn is_safe(&self) -> bool {
        matches!(
            self,
            Method::GET | Method::HEAD | Method::OPTIONS | Method::TRACE
        )
    }
}

impl fmt::Display for Method {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl TryFrom<&str> for Method {
    type Error = ProtocolError;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        match s.to_uppercase().as_str() {
            "GET" => Ok(Method::GET),
            "POST" => Ok(Method::POST),
            "PUT" => Ok(Method::PUT),
            "DELETE" => Ok(Method::DELETE),
            "PATCH" => Ok(Method::PATCH),
            "HEAD" => Ok(Method::HEAD),
            "OPTIONS" => Ok(Method::OPTIONS),
            "CONNECT" => Ok(Method::CONNECT),
            "TRACE" => Ok(Method::TRACE),
            _ => Err(ProtocolError::Protocol(format!(
                "Unknown HTTP method: {}",
                s
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uri_parsing() {
        let uri =
            Uri::parse("https://user:pass@example.com:8080/path?query=value#fragment").unwrap();
        assert_eq!(uri.scheme, "https");
        assert_eq!(uri.host, "example.com");
        assert_eq!(uri.port, Some(8080));
        assert_eq!(uri.path, "/path");
        assert_eq!(uri.query, Some("query=value".to_string()));
        assert_eq!(uri.fragment, Some("fragment".to_string()));
        assert_eq!(uri.userinfo, Some("user:pass".to_string()));
    }

    #[test]
    fn test_headers() {
        let mut headers = Headers::new();
        headers.insert("Content-Type", "application/json");
        headers.insert("X-Custom", "value1");
        headers.insert("X-Custom", "value2");

        assert_eq!(headers.get("content-type"), Some("application/json"));
        assert_eq!(headers.get_all("x-custom").map(|v| v.len()), Some(2));
    }

    #[test]
    fn test_status_code() {
        assert!(StatusCode::OK.is_success());
        assert!(StatusCode::NOT_FOUND.is_client_error());
        assert!(StatusCode::INTERNAL_SERVER_ERROR.is_server_error());
        assert!(StatusCode::MOVED_PERMANENTLY.is_redirection());
    }

    #[test]
    fn test_backoff_strategy() {
        let exp = BackoffStrategy::Exponential {
            initial: Duration::from_millis(100),
            factor: 2.0,
            max: Some(Duration::from_secs(10)),
        };

        assert_eq!(exp.delay_for_attempt(0), Duration::from_millis(100));
        assert_eq!(exp.delay_for_attempt(1), Duration::from_millis(200));
        assert_eq!(exp.delay_for_attempt(2), Duration::from_millis(400));
        assert_eq!(exp.delay_for_attempt(10), Duration::from_secs(10)); // Capped at max
    }
}
