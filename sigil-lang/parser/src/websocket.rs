//! Native WebSocket client implementation for Sigil.
//!
//! Implements RFC 6455 WebSocket protocol without external dependencies.
//! Supports both ws:// (plain) and wss:// (TLS) connections.

use std::io::{Read, Write, BufRead, BufReader};
use std::net::TcpStream;

#[cfg(feature = "websocket")]
use native_tls::TlsConnector;

/// WebSocket operation error
#[derive(Debug)]
pub struct WebSocketError {
    pub message: String,
}

impl WebSocketError {
    pub fn new(msg: impl Into<String>) -> Self {
        Self { message: msg.into() }
    }
}

impl std::fmt::Display for WebSocketError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WebSocket error: {}", self.message)
    }
}

impl std::error::Error for WebSocketError {}

/// WebSocket frame opcodes (RFC 6455 Section 5.2)
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u8)]
pub enum Opcode {
    Continuation = 0x0,
    Text = 0x1,
    Binary = 0x2,
    Close = 0x8,
    Ping = 0x9,
    Pong = 0xA,
}

impl Opcode {
    fn from_u8(val: u8) -> Option<Self> {
        match val {
            0x0 => Some(Opcode::Continuation),
            0x1 => Some(Opcode::Text),
            0x2 => Some(Opcode::Binary),
            0x8 => Some(Opcode::Close),
            0x9 => Some(Opcode::Ping),
            0xA => Some(Opcode::Pong),
            _ => None,
        }
    }
}

/// A WebSocket message
#[derive(Debug, Clone)]
pub enum Message {
    Text(String),
    Binary(Vec<u8>),
    Close,
    Ping(Vec<u8>),
    Pong(Vec<u8>),
}

/// Stream that can be either plain TCP or TLS-wrapped
enum Stream {
    Plain(TcpStream),
    #[cfg(feature = "websocket")]
    Tls(native_tls::TlsStream<TcpStream>),
}

impl Read for Stream {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Stream::Plain(s) => s.read(buf),
            #[cfg(feature = "websocket")]
            Stream::Tls(s) => s.read(buf),
        }
    }
}

impl Write for Stream {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            Stream::Plain(s) => s.write(buf),
            #[cfg(feature = "websocket")]
            Stream::Tls(s) => s.write(buf),
        }
    }

    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            Stream::Plain(s) => s.flush(),
            #[cfg(feature = "websocket")]
            Stream::Tls(s) => s.flush(),
        }
    }
}

/// Native WebSocket client
pub struct WebSocket {
    stream: Stream,
}

impl WebSocket {
    /// Connect to a WebSocket server
    ///
    /// Supports ws:// and wss:// URLs
    pub fn connect(url: &str) -> Result<Self, WebSocketError> {
        // Parse URL
        let (secure, host, port, path) = Self::parse_url(url)?;

        // Establish TCP connection
        let addr = format!("{}:{}", host, port);
        let tcp_stream = TcpStream::connect(&addr)
            .map_err(|e| WebSocketError::new(format!("TCP connection failed: {}", e)))?;

        // Set timeouts
        tcp_stream.set_read_timeout(Some(std::time::Duration::from_secs(30))).ok();
        tcp_stream.set_write_timeout(Some(std::time::Duration::from_secs(30))).ok();

        // Wrap in TLS if secure
        let stream = if secure {
            #[cfg(feature = "websocket")]
            {
                let connector = TlsConnector::new()
                    .map_err(|e| WebSocketError::new(format!("TLS setup failed: {}", e)))?;
                let tls_stream = connector.connect(&host, tcp_stream)
                    .map_err(|e| WebSocketError::new(format!("TLS handshake failed: {}", e)))?;
                Stream::Tls(tls_stream)
            }
            #[cfg(not(feature = "websocket"))]
            {
                return Err(WebSocketError::new("TLS support not compiled in"));
            }
        } else {
            Stream::Plain(tcp_stream)
        };

        let mut ws = WebSocket { stream };

        // Perform WebSocket handshake
        ws.handshake(&host, port, &path)?;

        Ok(ws)
    }

    /// Parse WebSocket URL into components
    fn parse_url(url: &str) -> Result<(bool, String, u16, String), WebSocketError> {
        let (secure, rest) = if url.starts_with("wss://") {
            (true, &url[6..])
        } else if url.starts_with("ws://") {
            (false, &url[5..])
        } else {
            return Err(WebSocketError::new("URL must start with ws:// or wss://"));
        };

        // Split host:port from path
        let (host_port, path) = match rest.find('/') {
            Some(idx) => (&rest[..idx], &rest[idx..]),
            None => (rest, "/"),
        };

        // Parse host and port
        let (host, port) = match host_port.find(':') {
            Some(idx) => {
                let port_str = &host_port[idx + 1..];
                let port = port_str.parse::<u16>()
                    .map_err(|_| WebSocketError::new("Invalid port number"))?;
                (host_port[..idx].to_string(), port)
            }
            None => (host_port.to_string(), if secure { 443 } else { 80 }),
        };

        Ok((secure, host, port, path.to_string()))
    }

    /// Perform WebSocket upgrade handshake (RFC 6455 Section 4)
    fn handshake(&mut self, host: &str, port: u16, path: &str) -> Result<(), WebSocketError> {
        // Generate random 16-byte key and base64 encode
        let key_bytes: [u8; 16] = rand::random();
        let key = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, key_bytes);

        // Build HTTP upgrade request
        let host_header = if port == 80 || port == 443 {
            host.to_string()
        } else {
            format!("{}:{}", host, port)
        };

        let request = format!(
            "GET {} HTTP/1.1\r\n\
             Host: {}\r\n\
             Upgrade: websocket\r\n\
             Connection: Upgrade\r\n\
             Sec-WebSocket-Key: {}\r\n\
             Sec-WebSocket-Version: 13\r\n\
             \r\n",
            path, host_header, key
        );

        // Send request
        self.stream.write_all(request.as_bytes())
            .map_err(|e| WebSocketError::new(format!("Failed to send handshake: {}", e)))?;
        self.stream.flush()
            .map_err(|e| WebSocketError::new(format!("Failed to flush handshake: {}", e)))?;

        // Read response
        let mut reader = BufReader::new(&mut self.stream);
        let mut response_line = String::new();
        reader.read_line(&mut response_line)
            .map_err(|e| WebSocketError::new(format!("Failed to read response: {}", e)))?;

        // Check status
        if !response_line.starts_with("HTTP/1.1 101") {
            return Err(WebSocketError::new(format!("Handshake failed: {}", response_line.trim())));
        }

        // Read and validate headers
        let expected_accept = Self::compute_accept_key(&key);
        let mut found_accept = false;

        loop {
            let mut line = String::new();
            reader.read_line(&mut line)
                .map_err(|e| WebSocketError::new(format!("Failed to read headers: {}", e)))?;

            let line = line.trim();
            if line.is_empty() {
                break; // End of headers
            }

            if let Some((name, value)) = line.split_once(':') {
                let name = name.trim().to_lowercase();
                let value = value.trim();

                if name == "sec-websocket-accept" {
                    if value != expected_accept {
                        return Err(WebSocketError::new("Invalid Sec-WebSocket-Accept"));
                    }
                    found_accept = true;
                }
            }
        }

        if !found_accept {
            return Err(WebSocketError::new("Missing Sec-WebSocket-Accept header"));
        }

        Ok(())
    }

    /// Compute the expected Sec-WebSocket-Accept value (RFC 6455 Section 4.2.2)
    fn compute_accept_key(key: &str) -> String {
        use sha1::{Sha1, Digest};

        // Concatenate with magic GUID
        let magic = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
        let combined = format!("{}{}", key, magic);

        // SHA-1 hash
        let mut hasher = Sha1::new();
        hasher.update(combined.as_bytes());
        let hash = hasher.finalize();

        // Base64 encode
        base64::Engine::encode(&base64::engine::general_purpose::STANDARD, hash)
    }

    /// Send a text message
    pub fn send_text(&mut self, text: &str) -> Result<(), WebSocketError> {
        self.send_frame(Opcode::Text, text.as_bytes())
    }

    /// Send a binary message
    pub fn send_binary(&mut self, data: &[u8]) -> Result<(), WebSocketError> {
        self.send_frame(Opcode::Binary, data)
    }

    /// Send a close frame
    pub fn send_close(&mut self) -> Result<(), WebSocketError> {
        self.send_frame(Opcode::Close, &[])
    }

    /// Send a WebSocket frame (RFC 6455 Section 5.2)
    ///
    /// Client frames MUST be masked per spec
    fn send_frame(&mut self, opcode: Opcode, payload: &[u8]) -> Result<(), WebSocketError> {
        let mut frame = Vec::with_capacity(14 + payload.len());

        // First byte: FIN + opcode
        frame.push(0x80 | (opcode as u8)); // FIN bit set

        // Second byte: MASK + payload length
        let len = payload.len();
        if len < 126 {
            frame.push(0x80 | len as u8); // Mask bit set
        } else if len < 65536 {
            frame.push(0x80 | 126);
            frame.push((len >> 8) as u8);
            frame.push(len as u8);
        } else {
            frame.push(0x80 | 127);
            for i in (0..8).rev() {
                frame.push((len >> (i * 8)) as u8);
            }
        }

        // Masking key (4 random bytes)
        let mask: [u8; 4] = rand::random();
        frame.extend_from_slice(&mask);

        // Masked payload
        for (i, &byte) in payload.iter().enumerate() {
            frame.push(byte ^ mask[i % 4]);
        }

        self.stream.write_all(&frame)
            .map_err(|e| WebSocketError::new(format!("Failed to send frame: {}", e)))?;
        self.stream.flush()
            .map_err(|e| WebSocketError::new(format!("Failed to flush frame: {}", e)))?;

        Ok(())
    }

    /// Receive a message
    pub fn receive(&mut self) -> Result<Message, WebSocketError> {
        loop {
            let (opcode, payload) = self.receive_frame()?;

            match opcode {
                Opcode::Text => {
                    let text = String::from_utf8(payload)
                        .map_err(|e| WebSocketError::new(format!("Invalid UTF-8: {}", e)))?;
                    return Ok(Message::Text(text));
                }
                Opcode::Binary => {
                    return Ok(Message::Binary(payload));
                }
                Opcode::Close => {
                    return Ok(Message::Close);
                }
                Opcode::Ping => {
                    // Respond with pong
                    self.send_frame(Opcode::Pong, &payload)?;
                    // Continue to receive actual message
                }
                Opcode::Pong => {
                    // Ignore pong, continue receiving
                }
                Opcode::Continuation => {
                    // For simplicity, treat as text (full fragmentation support would need state)
                    let text = String::from_utf8_lossy(&payload).to_string();
                    return Ok(Message::Text(text));
                }
            }
        }
    }

    /// Receive a WebSocket frame
    fn receive_frame(&mut self) -> Result<(Opcode, Vec<u8>), WebSocketError> {
        // Read first two bytes
        let mut header = [0u8; 2];
        self.read_exact(&mut header)?;

        let _fin = (header[0] & 0x80) != 0;
        let opcode = Opcode::from_u8(header[0] & 0x0F)
            .ok_or_else(|| WebSocketError::new("Invalid opcode"))?;

        let masked = (header[1] & 0x80) != 0;
        let mut len = (header[1] & 0x7F) as usize;

        // Extended payload length
        if len == 126 {
            let mut ext = [0u8; 2];
            self.read_exact(&mut ext)?;
            len = ((ext[0] as usize) << 8) | (ext[1] as usize);
        } else if len == 127 {
            let mut ext = [0u8; 8];
            self.read_exact(&mut ext)?;
            len = 0;
            for &b in &ext {
                len = (len << 8) | (b as usize);
            }
        }

        // Read masking key if present (server frames usually aren't masked)
        let mask = if masked {
            let mut m = [0u8; 4];
            self.read_exact(&mut m)?;
            Some(m)
        } else {
            None
        };

        // Read payload
        let mut payload = vec![0u8; len];
        if len > 0 {
            self.read_exact(&mut payload)?;
        }

        // Unmask if needed
        if let Some(mask) = mask {
            for (i, byte) in payload.iter_mut().enumerate() {
                *byte ^= mask[i % 4];
            }
        }

        Ok((opcode, payload))
    }

    /// Read exact number of bytes
    fn read_exact(&mut self, buf: &mut [u8]) -> Result<(), WebSocketError> {
        let mut total = 0;
        while total < buf.len() {
            match self.stream.read(&mut buf[total..]) {
                Ok(0) => return Err(WebSocketError::new("Connection closed")),
                Ok(n) => total += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(WebSocketError::new(format!("Read error: {}", e))),
            }
        }
        Ok(())
    }

    /// Close the WebSocket connection gracefully
    pub fn close(&mut self) -> Result<(), WebSocketError> {
        // Send close frame
        let _ = self.send_close();
        Ok(())
    }
}

/// Connect to a WebSocket server, send a message, receive response, and close
///
/// This is a convenience function for simple request-response patterns
pub fn send_and_receive(url: &str, message: &str) -> Result<String, WebSocketError> {
    let mut ws = WebSocket::connect(url)?;
    ws.send_text(message)?;

    let response = match ws.receive()? {
        Message::Text(t) => t,
        Message::Binary(b) => String::from_utf8_lossy(&b).to_string(),
        Message::Close => String::new(),
        Message::Ping(_) | Message::Pong(_) => {
            // Try to receive actual message after ping/pong
            match ws.receive()? {
                Message::Text(t) => t,
                Message::Binary(b) => String::from_utf8_lossy(&b).to_string(),
                _ => String::new(),
            }
        }
    };

    ws.close()?;
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_url() {
        let (secure, host, port, path) = WebSocket::parse_url("ws://example.com/path").unwrap();
        assert!(!secure);
        assert_eq!(host, "example.com");
        assert_eq!(port, 80);
        assert_eq!(path, "/path");

        let (secure, host, port, path) = WebSocket::parse_url("wss://example.com:8443/api").unwrap();
        assert!(secure);
        assert_eq!(host, "example.com");
        assert_eq!(port, 8443);
        assert_eq!(path, "/api");
    }

    #[test]
    fn test_compute_accept_key() {
        // Test vector from RFC 6455
        let key = "dGhlIHNhbXBsZSBub25jZQ==";
        let expected = "s3pPLMBiTxaQ9kYGzzhZRbK+xOo=";
        assert_eq!(WebSocket::compute_accept_key(key), expected);
    }
}
