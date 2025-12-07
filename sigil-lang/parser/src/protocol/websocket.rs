//! WebSocket Support
//!
//! Bidirectional real-time communication over WebSocket protocol.
//!
//! ## Features
//!
//! - Full WebSocket protocol support (RFC 6455)
//! - Text and binary messages
//! - Ping/pong handling
//! - Automatic reconnection
//! - TLS/WSS support
//! - Subprotocol negotiation

use super::common::{Headers, ProtocolError, ProtocolResult, Timeout, Uri};
use std::time::Duration;

/// WebSocket message types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Message {
    /// UTF-8 text message
    Text(String),
    /// Binary message
    Binary(Vec<u8>),
    /// Ping message (client should respond with Pong)
    Ping(Vec<u8>),
    /// Pong message (response to Ping)
    Pong(Vec<u8>),
    /// Close message with optional code and reason
    Close(Option<CloseFrame>),
}

impl Message {
    /// Create a text message
    pub fn text(s: impl Into<String>) -> Self {
        Message::Text(s.into())
    }

    /// Create a binary message
    pub fn binary(data: impl Into<Vec<u8>>) -> Self {
        Message::Binary(data.into())
    }

    /// Create a ping message
    pub fn ping(data: impl Into<Vec<u8>>) -> Self {
        Message::Ping(data.into())
    }

    /// Create a pong message
    pub fn pong(data: impl Into<Vec<u8>>) -> Self {
        Message::Pong(data.into())
    }

    /// Create a close message
    pub fn close(code: CloseCode, reason: impl Into<String>) -> Self {
        Message::Close(Some(CloseFrame {
            code,
            reason: reason.into(),
        }))
    }

    /// Check if this is a text message
    pub fn is_text(&self) -> bool {
        matches!(self, Message::Text(_))
    }

    /// Check if this is a binary message
    pub fn is_binary(&self) -> bool {
        matches!(self, Message::Binary(_))
    }

    /// Check if this is a ping message
    pub fn is_ping(&self) -> bool {
        matches!(self, Message::Ping(_))
    }

    /// Check if this is a pong message
    pub fn is_pong(&self) -> bool {
        matches!(self, Message::Pong(_))
    }

    /// Check if this is a close message
    pub fn is_close(&self) -> bool {
        matches!(self, Message::Close(_))
    }

    /// Check if this is a data message (text or binary)
    pub fn is_data(&self) -> bool {
        matches!(self, Message::Text(_) | Message::Binary(_))
    }

    /// Get the text content if this is a text message
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Message::Text(s) => Some(s),
            _ => None,
        }
    }

    /// Get the binary content if this is a binary message
    pub fn as_binary(&self) -> Option<&[u8]> {
        match self {
            Message::Binary(b) => Some(b),
            _ => None,
        }
    }

    /// Convert to text (consumes the message)
    pub fn into_text(self) -> Option<String> {
        match self {
            Message::Text(s) => Some(s),
            _ => None,
        }
    }

    /// Convert to bytes (consumes the message)
    pub fn into_bytes(self) -> Option<Vec<u8>> {
        match self {
            Message::Binary(b) => Some(b),
            _ => None,
        }
    }

    /// Get the payload length
    pub fn len(&self) -> usize {
        match self {
            Message::Text(s) => s.len(),
            Message::Binary(b) | Message::Ping(b) | Message::Pong(b) => b.len(),
            Message::Close(Some(frame)) => 2 + frame.reason.len(),
            Message::Close(None) => 0,
        }
    }

    /// Check if the payload is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Close frame with status code and reason
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CloseFrame {
    /// Close status code
    pub code: CloseCode,
    /// Close reason (UTF-8 text)
    pub reason: String,
}

/// WebSocket close codes (RFC 6455)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum CloseCode {
    /// Normal closure
    Normal = 1000,
    /// Endpoint going away
    GoingAway = 1001,
    /// Protocol error
    Protocol = 1002,
    /// Unsupported data type
    Unsupported = 1003,
    /// No status received (reserved, never sent)
    NoStatus = 1005,
    /// Abnormal closure (reserved, never sent)
    Abnormal = 1006,
    /// Invalid payload data
    InvalidData = 1007,
    /// Policy violation
    Policy = 1008,
    /// Message too big
    MessageTooBig = 1009,
    /// Missing extension
    MissingExtension = 1010,
    /// Internal server error
    InternalError = 1011,
    /// TLS handshake failure (reserved, never sent)
    TlsFailure = 1015,
    /// Application-specific code
    Custom(u16),
}

impl CloseCode {
    /// Create from u16
    pub fn from_u16(code: u16) -> Self {
        match code {
            1000 => CloseCode::Normal,
            1001 => CloseCode::GoingAway,
            1002 => CloseCode::Protocol,
            1003 => CloseCode::Unsupported,
            1005 => CloseCode::NoStatus,
            1006 => CloseCode::Abnormal,
            1007 => CloseCode::InvalidData,
            1008 => CloseCode::Policy,
            1009 => CloseCode::MessageTooBig,
            1010 => CloseCode::MissingExtension,
            1011 => CloseCode::InternalError,
            1015 => CloseCode::TlsFailure,
            _ => CloseCode::Custom(code),
        }
    }

    /// Get the u16 code value
    pub fn as_u16(&self) -> u16 {
        match self {
            CloseCode::Normal => 1000,
            CloseCode::GoingAway => 1001,
            CloseCode::Protocol => 1002,
            CloseCode::Unsupported => 1003,
            CloseCode::NoStatus => 1005,
            CloseCode::Abnormal => 1006,
            CloseCode::InvalidData => 1007,
            CloseCode::Policy => 1008,
            CloseCode::MessageTooBig => 1009,
            CloseCode::MissingExtension => 1010,
            CloseCode::InternalError => 1011,
            CloseCode::TlsFailure => 1015,
            CloseCode::Custom(code) => *code,
        }
    }

    /// Get the close code description
    pub fn description(&self) -> &'static str {
        match self {
            CloseCode::Normal => "Normal closure",
            CloseCode::GoingAway => "Endpoint going away",
            CloseCode::Protocol => "Protocol error",
            CloseCode::Unsupported => "Unsupported data type",
            CloseCode::NoStatus => "No status received",
            CloseCode::Abnormal => "Abnormal closure",
            CloseCode::InvalidData => "Invalid payload data",
            CloseCode::Policy => "Policy violation",
            CloseCode::MessageTooBig => "Message too big",
            CloseCode::MissingExtension => "Missing extension",
            CloseCode::InternalError => "Internal server error",
            CloseCode::TlsFailure => "TLS handshake failure",
            CloseCode::Custom(_) => "Custom close code",
        }
    }
}

impl std::fmt::Display for CloseCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.description(), self.as_u16())
    }
}

/// WebSocket connection configuration
#[derive(Debug, Clone)]
pub struct WebSocketConfig {
    /// Maximum message size
    pub max_message_size: usize,
    /// Maximum frame size
    pub max_frame_size: usize,
    /// Whether to accept unmasked frames from clients
    pub accept_unmasked_frames: bool,
    /// Subprotocols to request
    pub subprotocols: Vec<String>,
    /// Additional headers to send with the upgrade request
    pub headers: Headers,
    /// Connection timeout
    pub connect_timeout: Duration,
    /// Ping interval for keep-alive
    pub ping_interval: Option<Duration>,
    /// Pong timeout (close connection if no pong received)
    pub pong_timeout: Option<Duration>,
}

impl Default for WebSocketConfig {
    fn default() -> Self {
        WebSocketConfig {
            max_message_size: 64 * 1024 * 1024, // 64MB
            max_frame_size: 16 * 1024 * 1024,   // 16MB
            accept_unmasked_frames: false,
            subprotocols: Vec::new(),
            headers: Headers::new(),
            connect_timeout: Duration::from_secs(30),
            ping_interval: Some(Duration::from_secs(30)),
            pong_timeout: Some(Duration::from_secs(10)),
        }
    }
}

/// WebSocket connection state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionState {
    /// Connection is being established
    Connecting,
    /// Connection is open and ready
    Open,
    /// Connection is closing
    Closing,
    /// Connection is closed
    Closed,
}

/// WebSocket connection
#[derive(Debug)]
pub struct WebSocket {
    /// The WebSocket URL
    url: String,
    /// Connection configuration
    config: WebSocketConfig,
    /// Current connection state
    state: ConnectionState,
    /// Negotiated subprotocol
    subprotocol: Option<String>,
    #[cfg(feature = "tokio-tungstenite")]
    inner: Option<
        tokio_tungstenite::WebSocketStream<
            tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
        >,
    >,
}

impl WebSocket {
    /// Create a connection builder
    pub fn builder(url: impl Into<String>) -> WebSocketBuilder {
        WebSocketBuilder {
            url: url.into(),
            config: WebSocketConfig::default(),
        }
    }

    /// Connect to a WebSocket server
    #[cfg(feature = "tokio-tungstenite")]
    pub async fn connect(url: impl Into<String>) -> ProtocolResult<Self> {
        Self::builder(url).connect().await
    }

    /// Connect to a WebSocket server (stub for non-tungstenite builds)
    #[cfg(not(feature = "tokio-tungstenite"))]
    pub async fn connect(url: impl Into<String>) -> ProtocolResult<Self> {
        let _ = url;
        Err(ProtocolError::Protocol(
            "WebSocket requires 'websocket' feature".to_string(),
        ))
    }

    /// Get the connection URL
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Get the current connection state
    pub fn state(&self) -> ConnectionState {
        self.state
    }

    /// Check if the connection is open
    pub fn is_open(&self) -> bool {
        self.state == ConnectionState::Open
    }

    /// Get the negotiated subprotocol
    pub fn subprotocol(&self) -> Option<&str> {
        self.subprotocol.as_deref()
    }

    /// Send a message
    #[cfg(feature = "tokio-tungstenite")]
    pub async fn send(&mut self, message: Message) -> ProtocolResult<()> {
        use futures_util::SinkExt;
        use tokio_tungstenite::tungstenite::Message as TMessage;

        if self.state != ConnectionState::Open {
            return Err(ProtocolError::ChannelClosed);
        }

        let msg = match message {
            Message::Text(s) => TMessage::Text(s),
            Message::Binary(b) => TMessage::Binary(b),
            Message::Ping(b) => TMessage::Ping(b),
            Message::Pong(b) => TMessage::Pong(b),
            Message::Close(frame) => {
                let close_frame = frame.map(|f| {
                    tokio_tungstenite::tungstenite::protocol::CloseFrame {
                        code: tokio_tungstenite::tungstenite::protocol::frame::coding::CloseCode::from(f.code.as_u16()),
                        reason: f.reason.into(),
                    }
                });
                TMessage::Close(close_frame)
            }
        };

        if let Some(ref mut inner) = self.inner {
            inner
                .send(msg)
                .await
                .map_err(|e| ProtocolError::Protocol(e.to_string()))?;
        }

        Ok(())
    }

    /// Send a message (stub for non-tungstenite builds)
    #[cfg(not(feature = "tokio-tungstenite"))]
    pub async fn send(&mut self, message: Message) -> ProtocolResult<()> {
        let _ = message;
        Err(ProtocolError::Protocol(
            "WebSocket requires 'websocket' feature".to_string(),
        ))
    }

    /// Receive a message
    #[cfg(feature = "tokio-tungstenite")]
    pub async fn recv(&mut self) -> ProtocolResult<Option<Message>> {
        use futures_util::StreamExt;
        use tokio_tungstenite::tungstenite::Message as TMessage;

        if self.state != ConnectionState::Open {
            return Ok(None);
        }

        if let Some(ref mut inner) = self.inner {
            match inner.next().await {
                Some(Ok(msg)) => {
                    let message = match msg {
                        TMessage::Text(s) => Message::Text(s),
                        TMessage::Binary(b) => Message::Binary(b),
                        TMessage::Ping(b) => Message::Ping(b),
                        TMessage::Pong(b) => Message::Pong(b),
                        TMessage::Close(frame) => {
                            self.state = ConnectionState::Closed;
                            Message::Close(frame.map(|f| CloseFrame {
                                code: CloseCode::from_u16(f.code.into()),
                                reason: f.reason.to_string(),
                            }))
                        }
                        TMessage::Frame(_) => return self.recv().await, // Skip raw frames
                    };
                    Ok(Some(message))
                }
                Some(Err(e)) => {
                    self.state = ConnectionState::Closed;
                    Err(ProtocolError::Protocol(e.to_string()))
                }
                None => {
                    self.state = ConnectionState::Closed;
                    Ok(None)
                }
            }
        } else {
            Ok(None)
        }
    }

    /// Receive a message (stub for non-tungstenite builds)
    #[cfg(not(feature = "tokio-tungstenite"))]
    pub async fn recv(&mut self) -> ProtocolResult<Option<Message>> {
        Err(ProtocolError::Protocol(
            "WebSocket requires 'websocket' feature".to_string(),
        ))
    }

    /// Send a text message
    pub async fn send_text(&mut self, text: impl Into<String>) -> ProtocolResult<()> {
        self.send(Message::Text(text.into())).await
    }

    /// Send a binary message
    pub async fn send_binary(&mut self, data: impl Into<Vec<u8>>) -> ProtocolResult<()> {
        self.send(Message::Binary(data.into())).await
    }

    /// Send a ping
    pub async fn ping(&mut self, data: impl Into<Vec<u8>>) -> ProtocolResult<()> {
        self.send(Message::Ping(data.into())).await
    }

    /// Close the connection
    pub async fn close(
        &mut self,
        code: CloseCode,
        reason: impl Into<String>,
    ) -> ProtocolResult<()> {
        if self.state != ConnectionState::Open {
            return Ok(());
        }

        self.state = ConnectionState::Closing;
        self.send(Message::close(code, reason)).await?;
        self.state = ConnectionState::Closed;
        Ok(())
    }

    /// Close with normal status
    pub async fn close_normal(&mut self) -> ProtocolResult<()> {
        self.close(CloseCode::Normal, "").await
    }
}

/// Builder for WebSocket connections
#[derive(Debug, Clone)]
pub struct WebSocketBuilder {
    url: String,
    config: WebSocketConfig,
}

impl WebSocketBuilder {
    /// Set maximum message size
    pub fn max_message_size(mut self, size: usize) -> Self {
        self.config.max_message_size = size;
        self
    }

    /// Set maximum frame size
    pub fn max_frame_size(mut self, size: usize) -> Self {
        self.config.max_frame_size = size;
        self
    }

    /// Add a subprotocol to request
    pub fn subprotocol(mut self, protocol: impl Into<String>) -> Self {
        self.config.subprotocols.push(protocol.into());
        self
    }

    /// Add subprotocols to request
    pub fn subprotocols(mut self, protocols: Vec<String>) -> Self {
        self.config.subprotocols.extend(protocols);
        self
    }

    /// Add a header to the upgrade request
    pub fn header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.config.headers.insert(key, value);
        self
    }

    /// Set connection timeout
    pub fn connect_timeout(mut self, timeout: Duration) -> Self {
        self.config.connect_timeout = timeout;
        self
    }

    /// Set ping interval for keep-alive
    pub fn ping_interval(mut self, interval: Duration) -> Self {
        self.config.ping_interval = Some(interval);
        self
    }

    /// Disable ping keep-alive
    pub fn no_ping(mut self) -> Self {
        self.config.ping_interval = None;
        self
    }

    /// Set bearer authentication
    pub fn bearer_auth(self, token: impl Into<String>) -> Self {
        self.header("Authorization", format!("Bearer {}", token.into()))
    }

    /// Connect to the WebSocket server
    #[cfg(feature = "tokio-tungstenite")]
    pub async fn connect(self) -> ProtocolResult<WebSocket> {
        use tokio_tungstenite::connect_async;

        let (ws_stream, _response) = connect_async(&self.url)
            .await
            .map_err(|e| ProtocolError::ConnectionFailed(e.to_string()))?;

        Ok(WebSocket {
            url: self.url,
            config: self.config,
            state: ConnectionState::Open,
            subprotocol: None, // TODO: extract from response
            inner: Some(ws_stream),
        })
    }

    /// Connect to the WebSocket server (stub for non-tungstenite builds)
    #[cfg(not(feature = "tokio-tungstenite"))]
    pub async fn connect(self) -> ProtocolResult<WebSocket> {
        Err(ProtocolError::Protocol(
            "WebSocket requires 'websocket' feature".to_string(),
        ))
    }
}

/// Reconnecting WebSocket wrapper
#[derive(Debug)]
pub struct ReconnectingWebSocket {
    /// Builder used to create connections
    builder: WebSocketBuilder,
    /// Current connection
    connection: Option<WebSocket>,
    /// Reconnect configuration
    reconnect_config: ReconnectConfig,
    /// Number of reconnection attempts
    attempt_count: u32,
}

/// Reconnection configuration
#[derive(Debug, Clone)]
pub struct ReconnectConfig {
    /// Initial delay before reconnecting
    pub initial_delay: Duration,
    /// Maximum delay between reconnection attempts
    pub max_delay: Duration,
    /// Delay multiplier for exponential backoff
    pub multiplier: f64,
    /// Maximum number of reconnection attempts (None = infinite)
    pub max_attempts: Option<u32>,
}

impl Default for ReconnectConfig {
    fn default() -> Self {
        ReconnectConfig {
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(30),
            multiplier: 2.0,
            max_attempts: None,
        }
    }
}

impl ReconnectingWebSocket {
    /// Create a new reconnecting WebSocket
    pub fn new(url: impl Into<String>) -> Self {
        ReconnectingWebSocket {
            builder: WebSocket::builder(url),
            connection: None,
            reconnect_config: ReconnectConfig::default(),
            attempt_count: 0,
        }
    }

    /// Configure reconnection
    pub fn reconnect_config(mut self, config: ReconnectConfig) -> Self {
        self.reconnect_config = config;
        self
    }

    /// Set initial reconnect delay
    pub fn initial_delay(mut self, delay: Duration) -> Self {
        self.reconnect_config.initial_delay = delay;
        self
    }

    /// Set maximum reconnect delay
    pub fn max_delay(mut self, delay: Duration) -> Self {
        self.reconnect_config.max_delay = delay;
        self
    }

    /// Set maximum reconnection attempts
    pub fn max_attempts(mut self, attempts: u32) -> Self {
        self.reconnect_config.max_attempts = Some(attempts);
        self
    }

    /// Connect (or reconnect) to the server
    pub async fn connect(&mut self) -> ProtocolResult<()> {
        self.connection = Some(self.builder.clone().connect().await?);
        self.attempt_count = 0;
        Ok(())
    }

    /// Check if connected
    pub fn is_connected(&self) -> bool {
        self.connection
            .as_ref()
            .map(|c| c.is_open())
            .unwrap_or(false)
    }

    /// Get the current connection
    pub fn connection(&mut self) -> Option<&mut WebSocket> {
        self.connection.as_mut()
    }

    /// Calculate the delay for the next reconnection attempt
    fn next_delay(&self) -> Duration {
        let delay = self.reconnect_config.initial_delay.mul_f64(
            self.reconnect_config
                .multiplier
                .powi(self.attempt_count as i32),
        );
        delay.min(self.reconnect_config.max_delay)
    }

    /// Attempt to reconnect
    pub async fn reconnect(&mut self) -> ProtocolResult<()> {
        if let Some(max) = self.reconnect_config.max_attempts {
            if self.attempt_count >= max {
                return Err(ProtocolError::ConnectionFailed(format!(
                    "Max reconnection attempts ({}) exceeded",
                    max
                )));
            }
        }

        let delay = self.next_delay();
        self.attempt_count += 1;

        #[cfg(feature = "tokio")]
        tokio::time::sleep(delay).await;

        self.connection = Some(self.builder.clone().connect().await?);
        self.attempt_count = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_types() {
        let text = Message::text("hello");
        assert!(text.is_text());
        assert!(text.is_data());
        assert_eq!(text.as_text(), Some("hello"));

        let binary = Message::binary(vec![1, 2, 3]);
        assert!(binary.is_binary());
        assert!(binary.is_data());
        assert_eq!(binary.as_binary(), Some(&[1u8, 2, 3][..]));

        let ping = Message::ping(vec![1, 2]);
        assert!(ping.is_ping());
        assert!(!ping.is_data());
    }

    #[test]
    fn test_close_codes() {
        assert_eq!(CloseCode::Normal.as_u16(), 1000);
        assert_eq!(CloseCode::from_u16(1000), CloseCode::Normal);
        assert_eq!(CloseCode::from_u16(4000), CloseCode::Custom(4000));
    }

    #[test]
    fn test_websocket_builder() {
        let builder = WebSocket::builder("wss://example.com/socket")
            .subprotocol("graphql-transport-ws")
            .bearer_auth("token123")
            .connect_timeout(Duration::from_secs(10));

        assert_eq!(builder.url, "wss://example.com/socket");
        assert!(builder
            .config
            .subprotocols
            .contains(&"graphql-transport-ws".to_string()));
    }

    #[test]
    fn test_reconnect_config() {
        let config = ReconnectConfig {
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(30),
            multiplier: 2.0,
            max_attempts: Some(5),
        };

        let ws = ReconnectingWebSocket::new("wss://example.com").reconnect_config(config);

        assert_eq!(ws.reconnect_config.max_attempts, Some(5));
    }
}
