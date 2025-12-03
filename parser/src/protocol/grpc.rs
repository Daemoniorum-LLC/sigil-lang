//! gRPC Support
//!
//! High-performance RPC framework with Protocol Buffers support.
//!
//! ## Features
//!
//! - Unary, server streaming, client streaming, and bidirectional streaming
//! - Connection multiplexing
//! - TLS/mTLS support
//! - Load balancing
//! - Interceptors/middleware
//! - Deadline propagation
//! - Compression

use super::common::{Headers, ProtocolError, ProtocolResult, Timeout, Uri};
use std::collections::HashMap;
use std::time::Duration;

/// gRPC status codes (based on google.rpc.Code)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum Code {
    /// Not an error; returned on success
    Ok = 0,
    /// The operation was cancelled
    Cancelled = 1,
    /// Unknown error
    Unknown = 2,
    /// Client specified an invalid argument
    InvalidArgument = 3,
    /// Deadline expired before operation could complete
    DeadlineExceeded = 4,
    /// Some requested entity was not found
    NotFound = 5,
    /// Entity already exists
    AlreadyExists = 6,
    /// Permission denied
    PermissionDenied = 7,
    /// Resource exhausted (e.g., rate limiting)
    ResourceExhausted = 8,
    /// Operation rejected because precondition failed
    FailedPrecondition = 9,
    /// Operation was aborted
    Aborted = 10,
    /// Operation was outside valid range
    OutOfRange = 11,
    /// Operation not implemented
    Unimplemented = 12,
    /// Internal error
    Internal = 13,
    /// Service unavailable
    Unavailable = 14,
    /// Data loss
    DataLoss = 15,
    /// Request lacks valid authentication credentials
    Unauthenticated = 16,
}

impl Code {
    /// Convert from i32
    pub fn from_i32(code: i32) -> Self {
        match code {
            0 => Code::Ok,
            1 => Code::Cancelled,
            2 => Code::Unknown,
            3 => Code::InvalidArgument,
            4 => Code::DeadlineExceeded,
            5 => Code::NotFound,
            6 => Code::AlreadyExists,
            7 => Code::PermissionDenied,
            8 => Code::ResourceExhausted,
            9 => Code::FailedPrecondition,
            10 => Code::Aborted,
            11 => Code::OutOfRange,
            12 => Code::Unimplemented,
            13 => Code::Internal,
            14 => Code::Unavailable,
            15 => Code::DataLoss,
            16 => Code::Unauthenticated,
            _ => Code::Unknown,
        }
    }

    /// Get the status code description
    pub fn description(&self) -> &'static str {
        match self {
            Code::Ok => "The operation completed successfully",
            Code::Cancelled => "The operation was cancelled",
            Code::Unknown => "Unknown error",
            Code::InvalidArgument => "Client specified an invalid argument",
            Code::DeadlineExceeded => "Deadline expired before operation could complete",
            Code::NotFound => "Some requested entity was not found",
            Code::AlreadyExists => "The entity already exists",
            Code::PermissionDenied => "Permission denied",
            Code::ResourceExhausted => "Resource exhausted",
            Code::FailedPrecondition => "Precondition failed",
            Code::Aborted => "The operation was aborted",
            Code::OutOfRange => "Operation was out of range",
            Code::Unimplemented => "Operation not implemented",
            Code::Internal => "Internal error",
            Code::Unavailable => "Service unavailable",
            Code::DataLoss => "Data loss",
            Code::Unauthenticated => "Request lacks valid authentication credentials",
        }
    }

    /// Check if this is a successful status
    pub fn is_ok(&self) -> bool {
        matches!(self, Code::Ok)
    }

    /// Check if this error is retryable
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Code::Unavailable | Code::DeadlineExceeded | Code::ResourceExhausted | Code::Aborted
        )
    }
}

impl std::fmt::Display for Code {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// gRPC status with code and message
#[derive(Debug, Clone)]
pub struct Status {
    /// The status code
    pub code: Code,
    /// Human-readable error message
    pub message: String,
    /// Optional error details (serialized protobuf)
    pub details: Vec<u8>,
}

impl Status {
    /// Create a new status
    pub fn new(code: Code, message: impl Into<String>) -> Self {
        Status {
            code,
            message: message.into(),
            details: Vec::new(),
        }
    }

    /// Create an OK status
    pub fn ok() -> Self {
        Status::new(Code::Ok, "")
    }

    /// Create a cancelled status
    pub fn cancelled(message: impl Into<String>) -> Self {
        Status::new(Code::Cancelled, message)
    }

    /// Create an invalid argument status
    pub fn invalid_argument(message: impl Into<String>) -> Self {
        Status::new(Code::InvalidArgument, message)
    }

    /// Create a not found status
    pub fn not_found(message: impl Into<String>) -> Self {
        Status::new(Code::NotFound, message)
    }

    /// Create an already exists status
    pub fn already_exists(message: impl Into<String>) -> Self {
        Status::new(Code::AlreadyExists, message)
    }

    /// Create a permission denied status
    pub fn permission_denied(message: impl Into<String>) -> Self {
        Status::new(Code::PermissionDenied, message)
    }

    /// Create a resource exhausted status
    pub fn resource_exhausted(message: impl Into<String>) -> Self {
        Status::new(Code::ResourceExhausted, message)
    }

    /// Create an internal error status
    pub fn internal(message: impl Into<String>) -> Self {
        Status::new(Code::Internal, message)
    }

    /// Create an unavailable status
    pub fn unavailable(message: impl Into<String>) -> Self {
        Status::new(Code::Unavailable, message)
    }

    /// Create an unauthenticated status
    pub fn unauthenticated(message: impl Into<String>) -> Self {
        Status::new(Code::Unauthenticated, message)
    }

    /// Create an unimplemented status
    pub fn unimplemented(message: impl Into<String>) -> Self {
        Status::new(Code::Unimplemented, message)
    }

    /// Add error details
    pub fn with_details(mut self, details: Vec<u8>) -> Self {
        self.details = details;
        self
    }

    /// Check if this is an OK status
    pub fn is_ok(&self) -> bool {
        self.code.is_ok()
    }

    /// Convert to a result
    pub fn to_result<T>(self, value: T) -> Result<T, Self> {
        if self.is_ok() {
            Ok(value)
        } else {
            Err(self)
        }
    }
}

impl std::fmt::Display for Status {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for Status {}

impl From<Status> for ProtocolError {
    fn from(status: Status) -> Self {
        ProtocolError::Protocol(format!("gRPC error: {}", status))
    }
}

/// gRPC metadata (headers/trailers)
#[derive(Debug, Clone, Default)]
pub struct Metadata {
    inner: HashMap<String, Vec<String>>,
}

impl Metadata {
    /// Create empty metadata
    pub fn new() -> Self {
        Metadata {
            inner: HashMap::new(),
        }
    }

    /// Insert a metadata value (ASCII)
    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<String>) -> &mut Self {
        let key = key.into().to_lowercase();
        self.inner
            .entry(key)
            .or_insert_with(Vec::new)
            .push(value.into());
        self
    }

    /// Insert binary metadata (must have -bin suffix)
    pub fn insert_bin(&mut self, key: impl Into<String>, value: Vec<u8>) -> &mut Self {
        let mut key = key.into().to_lowercase();
        if !key.ends_with("-bin") {
            key.push_str("-bin");
        }
        let encoded = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &value);
        self.inner.entry(key).or_insert_with(Vec::new).push(encoded);
        self
    }

    /// Get a metadata value
    pub fn get(&self, key: &str) -> Option<&str> {
        self.inner
            .get(&key.to_lowercase())
            .and_then(|v| v.first().map(|s| s.as_str()))
    }

    /// Get binary metadata
    pub fn get_bin(&self, key: &str) -> Option<Vec<u8>> {
        let mut key = key.to_lowercase();
        if !key.ends_with("-bin") {
            key.push_str("-bin");
        }
        self.get(&key).and_then(|v| {
            base64::Engine::decode(&base64::engine::general_purpose::STANDARD, v).ok()
        })
    }

    /// Get all values for a key
    pub fn get_all(&self, key: &str) -> Option<&[String]> {
        self.inner.get(&key.to_lowercase()).map(|v| v.as_slice())
    }

    /// Iterate over all metadata
    pub fn iter(&self) -> impl Iterator<Item = (&String, &Vec<String>)> {
        self.inner.iter()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

/// gRPC channel configuration
#[derive(Debug, Clone)]
pub struct ChannelConfig {
    /// Connection timeout
    pub connect_timeout: Duration,
    /// Request timeout (deadline)
    pub timeout: Option<Duration>,
    /// Keep-alive configuration
    pub keep_alive: Option<KeepAliveConfig>,
    /// Maximum message size for sending
    pub max_send_message_size: Option<usize>,
    /// Maximum message size for receiving
    pub max_recv_message_size: Option<usize>,
    /// Initial connection window size
    pub initial_connection_window_size: Option<u32>,
    /// Initial stream window size
    pub initial_stream_window_size: Option<u32>,
    /// Compression algorithm
    pub compression: Option<Compression>,
    /// TLS configuration
    pub tls: Option<TlsConfig>,
}

impl Default for ChannelConfig {
    fn default() -> Self {
        ChannelConfig {
            connect_timeout: Duration::from_secs(20),
            timeout: None,
            keep_alive: Some(KeepAliveConfig::default()),
            max_send_message_size: Some(4 * 1024 * 1024), // 4MB
            max_recv_message_size: Some(4 * 1024 * 1024), // 4MB
            initial_connection_window_size: None,
            initial_stream_window_size: None,
            compression: None,
            tls: None,
        }
    }
}

/// Keep-alive configuration
#[derive(Debug, Clone)]
pub struct KeepAliveConfig {
    /// Interval between keep-alive pings
    pub interval: Duration,
    /// Timeout waiting for keep-alive response
    pub timeout: Duration,
    /// Send keep-alive even without active calls
    pub while_idle: bool,
}

impl Default for KeepAliveConfig {
    fn default() -> Self {
        KeepAliveConfig {
            interval: Duration::from_secs(20),
            timeout: Duration::from_secs(20),
            while_idle: false,
        }
    }
}

/// Compression algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compression {
    /// No compression
    None,
    /// Gzip compression
    Gzip,
    /// Deflate compression
    Deflate,
    /// Zstandard compression
    Zstd,
}

impl Compression {
    /// Get the algorithm name for the grpc-encoding header
    pub fn as_str(&self) -> &'static str {
        match self {
            Compression::None => "identity",
            Compression::Gzip => "gzip",
            Compression::Deflate => "deflate",
            Compression::Zstd => "zstd",
        }
    }
}

/// TLS configuration
#[derive(Debug, Clone)]
pub struct TlsConfig {
    /// Path to CA certificate(s)
    pub ca_cert: Option<String>,
    /// Client certificate path (for mTLS)
    pub client_cert: Option<String>,
    /// Client key path (for mTLS)
    pub client_key: Option<String>,
    /// Server name for SNI
    pub domain_name: Option<String>,
}

impl TlsConfig {
    /// Create a new TLS config
    pub fn new() -> Self {
        TlsConfig {
            ca_cert: None,
            client_cert: None,
            client_key: None,
            domain_name: None,
        }
    }

    /// Set CA certificate
    pub fn ca_cert(mut self, path: impl Into<String>) -> Self {
        self.ca_cert = Some(path.into());
        self
    }

    /// Set client certificate and key (for mTLS)
    pub fn client_identity(mut self, cert: impl Into<String>, key: impl Into<String>) -> Self {
        self.client_cert = Some(cert.into());
        self.client_key = Some(key.into());
        self
    }

    /// Set domain name for SNI
    pub fn domain_name(mut self, name: impl Into<String>) -> Self {
        self.domain_name = Some(name.into());
        self
    }
}

impl Default for TlsConfig {
    fn default() -> Self {
        TlsConfig::new()
    }
}

/// gRPC channel (connection to a server)
#[derive(Debug, Clone)]
pub struct Channel {
    /// The target URI
    uri: String,
    /// Channel configuration
    config: ChannelConfig,
    #[cfg(feature = "tonic")]
    inner: Option<tonic::transport::Channel>,
}

impl Channel {
    /// Create a channel builder
    pub fn builder(uri: impl Into<String>) -> ChannelBuilder {
        ChannelBuilder {
            uri: uri.into(),
            config: ChannelConfig::default(),
        }
    }

    /// Connect to a gRPC server
    #[cfg(feature = "tonic")]
    pub async fn connect(uri: impl Into<String>) -> ProtocolResult<Self> {
        Self::builder(uri).connect().await
    }

    /// Connect to a gRPC server (stub for non-tonic builds)
    #[cfg(not(feature = "tonic"))]
    pub async fn connect(uri: impl Into<String>) -> ProtocolResult<Self> {
        let _ = uri;
        Err(ProtocolError::Protocol(
            "gRPC requires 'grpc' feature".to_string(),
        ))
    }

    /// Get the target URI
    pub fn uri(&self) -> &str {
        &self.uri
    }

    /// Get the inner tonic channel
    #[cfg(feature = "tonic")]
    pub fn inner(&self) -> Option<&tonic::transport::Channel> {
        self.inner.as_ref()
    }
}

/// Builder for gRPC channels
#[derive(Debug, Clone)]
pub struct ChannelBuilder {
    uri: String,
    config: ChannelConfig,
}

impl ChannelBuilder {
    /// Set connection timeout
    pub fn connect_timeout(mut self, timeout: Duration) -> Self {
        self.config.connect_timeout = timeout;
        self
    }

    /// Set request timeout (deadline)
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.config.timeout = Some(timeout);
        self
    }

    /// Configure keep-alive
    pub fn keep_alive(mut self, config: KeepAliveConfig) -> Self {
        self.config.keep_alive = Some(config);
        self
    }

    /// Set maximum send message size
    pub fn max_send_message_size(mut self, size: usize) -> Self {
        self.config.max_send_message_size = Some(size);
        self
    }

    /// Set maximum receive message size
    pub fn max_recv_message_size(mut self, size: usize) -> Self {
        self.config.max_recv_message_size = Some(size);
        self
    }

    /// Set compression algorithm
    pub fn compression(mut self, compression: Compression) -> Self {
        self.config.compression = Some(compression);
        self
    }

    /// Configure TLS
    pub fn tls(mut self, config: TlsConfig) -> Self {
        self.config.tls = Some(config);
        self
    }

    /// Build and connect the channel
    #[cfg(feature = "tonic")]
    pub async fn connect(self) -> ProtocolResult<Channel> {
        use tonic::transport::Endpoint;

        let mut endpoint = Endpoint::from_shared(self.uri.clone())
            .map_err(|e| ProtocolError::InvalidUri(e.to_string()))?
            .connect_timeout(self.config.connect_timeout);

        if let Some(timeout) = self.config.timeout {
            endpoint = endpoint.timeout(timeout);
        }

        if let Some(keep_alive) = &self.config.keep_alive {
            endpoint = endpoint
                .keep_alive_timeout(keep_alive.timeout)
                .keep_alive_while_idle(keep_alive.while_idle);
        }

        if let Some(tls) = &self.config.tls {
            let mut tls_config = tonic::transport::ClientTlsConfig::new();

            if let Some(ref domain) = tls.domain_name {
                tls_config = tls_config.domain_name(domain);
            }

            endpoint = endpoint
                .tls_config(tls_config)
                .map_err(|e| ProtocolError::TlsError(e.to_string()))?;
        }

        let inner = endpoint
            .connect()
            .await
            .map_err(|e| ProtocolError::ConnectionFailed(e.to_string()))?;

        Ok(Channel {
            uri: self.uri,
            config: self.config,
            inner: Some(inner),
        })
    }

    /// Build and connect (stub for non-tonic builds)
    #[cfg(not(feature = "tonic"))]
    pub async fn connect(self) -> ProtocolResult<Channel> {
        Err(ProtocolError::Protocol(
            "gRPC requires 'grpc' feature".to_string(),
        ))
    }
}

/// gRPC request wrapper
#[derive(Debug)]
pub struct Request<T> {
    /// Request metadata
    pub metadata: Metadata,
    /// Request message
    pub message: T,
}

impl<T> Request<T> {
    /// Create a new request
    pub fn new(message: T) -> Self {
        Request {
            metadata: Metadata::new(),
            message,
        }
    }

    /// Add metadata
    pub fn metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Get the message
    pub fn into_inner(self) -> T {
        self.message
    }

    /// Get a reference to the message
    pub fn get_ref(&self) -> &T {
        &self.message
    }

    /// Get a mutable reference to the message
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.message
    }

    /// Map the message
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> Request<U> {
        Request {
            metadata: self.metadata,
            message: f(self.message),
        }
    }
}

/// gRPC response wrapper
#[derive(Debug)]
pub struct Response<T> {
    /// Response metadata (headers)
    pub metadata: Metadata,
    /// Response message
    pub message: T,
}

impl<T> Response<T> {
    /// Create a new response
    pub fn new(message: T) -> Self {
        Response {
            metadata: Metadata::new(),
            message,
        }
    }

    /// Add metadata
    pub fn metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Get the message
    pub fn into_inner(self) -> T {
        self.message
    }

    /// Get a reference to the message
    pub fn get_ref(&self) -> &T {
        &self.message
    }

    /// Get a mutable reference to the message
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.message
    }

    /// Map the message
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> Response<U> {
        Response {
            metadata: self.metadata,
            message: f(self.message),
        }
    }
}

/// Load balancing strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadBalancer {
    /// Round-robin between endpoints
    RoundRobin,
    /// Pick-first (use first available endpoint)
    PickFirst,
    /// Random selection
    Random,
    /// Least connections
    LeastConnections,
}

/// Interceptor for adding cross-cutting concerns
pub trait Interceptor: Send + Sync {
    /// Called before a request is sent
    fn intercept_request(&self, request: &mut Metadata) -> Result<(), Status>;

    /// Called after a response is received
    fn intercept_response(&self, response: &mut Metadata) -> Result<(), Status> {
        let _ = response;
        Ok(())
    }
}

/// Bearer token interceptor
pub struct BearerAuthInterceptor {
    token: String,
}

impl BearerAuthInterceptor {
    /// Create a new bearer auth interceptor
    pub fn new(token: impl Into<String>) -> Self {
        BearerAuthInterceptor {
            token: token.into(),
        }
    }
}

impl Interceptor for BearerAuthInterceptor {
    fn intercept_request(&self, request: &mut Metadata) -> Result<(), Status> {
        request.insert("authorization", format!("Bearer {}", self.token));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_status_codes() {
        assert!(Code::Ok.is_ok());
        assert!(!Code::Internal.is_ok());
        assert!(Code::Unavailable.is_retryable());
        assert!(!Code::InvalidArgument.is_retryable());
    }

    #[test]
    fn test_status() {
        let status = Status::not_found("Resource not found");
        assert_eq!(status.code, Code::NotFound);
        assert_eq!(status.message, "Resource not found");
    }

    #[test]
    fn test_metadata() {
        let mut meta = Metadata::new();
        meta.insert("x-request-id", "123");
        assert_eq!(meta.get("x-request-id"), Some("123"));
    }

    #[test]
    fn test_channel_builder() {
        let builder = Channel::builder("http://localhost:50051")
            .connect_timeout(Duration::from_secs(10))
            .compression(Compression::Gzip);

        assert_eq!(builder.config.connect_timeout, Duration::from_secs(10));
        assert_eq!(builder.config.compression, Some(Compression::Gzip));
    }

    #[test]
    fn test_request_response() {
        let req = Request::new("hello").metadata("x-custom", "value");
        assert_eq!(req.get_ref(), &"hello");
        assert_eq!(req.metadata.get("x-custom"), Some("value"));

        let resp = Response::new(42);
        assert_eq!(resp.into_inner(), 42);
    }
}
